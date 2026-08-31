"""Phase 5 training interface (§17).

    python -m alphabet train-decoder --config configs/alphabet_dataset.yaml

- GPU when available; mixed precision ONLY where numerically safe
  (model forward under autocast; CTC loss always float32)
- checkpointing + resume + best checkpoint + early stopping
- warmup + cosine LR schedule, gradient clipping
- balanced sampling decided from MEASURED letter counts (§18)
- train-time temporal augmentation only; val/test untouched (§19)
- calibration (temperature + abstention thresholds) fit on VALIDATION only
  (§14); test data untouched until final evaluation (§16)

Reuses the frozen Phase 2-4 infrastructure: config loader, temporal encoder,
Phase 3 representation cache. No second training framework.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from alphabet.config import DecoderConfig, ablation_config_hash
from alphabet.dataset import (
    AlphabetSequenceDataset, SampleMeta, balanced_sampling_weights,
    collate_batch, load_manifest_entries,
)
from alphabet.evaluate import evaluate_model
from alphabet.losses import MultitaskLoss
from alphabet.phonetics import (
    PhoneVocabulary, VisemeVocabulary, text_to_phone_sequence,
    text_to_viseme_sequence,
)
from alphabet.targets import TargetType, classify_target
from alphabet.vocab import DEFAULT_VOCABULARY, LetterVocabulary

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(dec_cfg: DecoderConfig, temporal_config, vocab: LetterVocabulary):
    """Multi-task model with auxiliary heads ONLY when explicitly enabled."""
    from alphabet.heads import MultiTaskAlphabetModel
    from temporal.encoder import TemporalEncoderConfig

    if isinstance(temporal_config, TemporalEncoderConfig):
        tcfg = temporal_config
    else:
        tcfg = TemporalEncoderConfig(**asdict(temporal_config))
    phone_v = viseme_v = None
    if dec_cfg.auxiliary.phone_enabled:
        phone_v = PhoneVocabulary().size
    if dec_cfg.auxiliary.viseme_enabled:
        viseme_v = VisemeVocabulary().size
    return MultiTaskAlphabetModel(
        temporal_config=tcfg,
        letter_vocab_size=vocab.size,
        phone_vocab_size=phone_v,
        viseme_vocab_size=viseme_v,
        letter_head_hidden=dec_cfg.training.letter_head_hidden,
    )


def _pad_id_sequences(seqs):
    """Pad lists of id sequences -> [B, S_max] + lengths (pad = blank id 0)."""
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    s_max = int(lengths.max()) if len(seqs) else 0
    targets = torch.zeros(len(seqs), s_max, dtype=torch.long)
    for i, s in enumerate(seqs):
        if s:
            targets[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return targets, lengths


def make_collate_with_aux(aux_kinds, vocab: LetterVocabulary):
    """Collate that additionally encodes derived phone/viseme targets if used.

    aux_kinds: subset of {"phone", "viseme"} for the enabled auxiliary heads.
    """
    base = collate_batch

    def collate(batch):
        out = base(batch)
        labels = [m.label for m in out["metas"]]
        if "phone" in aux_kinds:
            seqs = [PhoneVocabulary().encode(text_to_phone_sequence(t)) for t in labels]
            out["phone_targets"], out["phone_target_lengths"] = _pad_id_sequences(seqs)
        if "viseme" in aux_kinds:
            seqs = [VisemeVocabulary().encode(text_to_viseme_sequence(t)) for t in labels]
            out["viseme_targets"], out["viseme_target_lengths"] = _pad_id_sequences(seqs)
        return out

    return collate


def warmup_cosine_lambda(step: int, total_steps: int, warmup_steps: int,
                         min_ratio: float):
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_ratio + (1.0 - min_ratio) * cos
def build_datasets(dec_cfg: DecoderConfig, phase2_cfg, vocab: LetterVocabulary,
                   limit_train: Optional[int] = None,
                   limit_val: Optional[int] = None,
                   limit_test: Optional[int] = None):
    """Manifest-driven datasets over the ablation-specific representation cache.

    Split integrity (§16): entries come pre-split in the manifest; the decoder
    never sees a recording in two splits. No speaker/session/frame leakage.
    """
    from alphabet.augment import SequenceAugmenter

    root = Path(phase2_cfg.dataset_root)
    cache_dir = Path(dec_cfg.repr_cache_dir) / dec_cfg.ablation
    expected_hash = ablation_config_hash(phase2_cfg.config_hash, dec_cfg.ablation)
    train_aug = SequenceAugmenter() if dec_cfg.training.augment else None

    def _ds(split, limit, augment):
        entries = load_manifest_entries(root, split=split)
        if limit is not None:
            entries = entries[:limit]
        return AlphabetSequenceDataset(
            entries, dataset_root=root, repr_cache_dir=cache_dir,
            expected_config_hash=expected_hash, vocab=vocab, augment=augment,
            seed=dec_cfg.training.seed,
        )

    return {
        "train": _ds("train", limit_train, train_aug),
        "val": _ds("val", limit_val, None),
        "test": _ds("test", limit_test, None),
    }


def make_train_loader(dataset, dec_cfg, collate_fn=collate_batch):
    """DataLoader with optional measured balanced sampling (§18)."""
    tcfg = dec_cfg.training
    metas: List[SampleMeta] = [s[0] for s in dataset.samples]
    counts: Dict[str, int] = {}
    for m in metas:
        counts[m.label] = counts.get(m.label, 0) + 1
    ratio = (max(counts.values()) / min(counts.values())) if counts else 1.0
    use_balanced = (tcfg.balanced_sampling == "on" or
                    (tcfg.balanced_sampling == "auto"
                     and ratio >= tcfg.auto_imbalance_ratio))
    sampler = None
    shuffle = True
    if use_balanced and len(metas) > 1:
        weights = balanced_sampling_weights(metas)
        sampler = WeightedRandomSampler(weights, num_samples=len(metas),
                                        replacement=True)
        shuffle = False
    logger.info("letter counts=%s imbalance_ratio=%.2f balanced_sampling=%s",
                dict(sorted(counts.items())), ratio, use_balanced)
    return DataLoader(dataset, batch_size=tcfg.batch_size, shuffle=shuffle,
                      sampler=sampler, num_workers=tcfg.num_workers,
                      collate_fn=collate_fn, drop_last=False)


def make_eval_loader(dataset, dec_cfg, collate_fn=collate_batch):
    return DataLoader(dataset, batch_size=dec_cfg.training.batch_size,
                      shuffle=False, num_workers=dec_cfg.training.num_workers,
                      collate_fn=collate_fn)


def _move_batch(batch, device):
    out = {
        "fused": batch["fused"].to(device),
        "mask": batch["mask"].to(device),
        "input_lengths": batch["input_lengths"],
        "targets": batch["targets"],
        "target_lengths": batch["target_lengths"],
        "metas": batch["metas"],
    }
    for k in ("phone_targets", "phone_target_lengths",
              "viseme_targets", "viseme_target_lengths"):
        if k in batch:
            out[k] = batch[k]
    return out


def run_train_epoch(model, loader, loss_fn, device, optimizer=None,
                    scheduler=None, grad_clip: float = 0.0, amp: bool = False):
    """One epoch; averaged loss parts returned. Backward runs iff optimizer given."""
    model.train(optimizer is not None)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))
    sums: Dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        if amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(batch["fused"], batch["mask"])
        else:
            outputs = model(batch["fused"], batch["mask"])
        # CTC loss is always float32 (losses.py casts internally) — safe under AMP.
        parts = loss_fn(
            outputs, batch["mask"], batch["targets"], batch["target_lengths"],
            phone_targets=batch.get("phone_targets"),
            phone_target_lengths=batch.get("phone_target_lengths"),
            viseme_targets=batch.get("viseme_targets"),
            viseme_target_lengths=batch.get("viseme_target_lengths"),
        )
        loss = parts["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {loss.item()}")
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v.item())
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


def compute_val_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            outputs = model(batch["fused"], batch["mask"])
            parts = loss_fn(outputs, batch["mask"], batch["targets"],
                            batch["target_lengths"])
            total += float(parts["total"].item())
            n += 1
    return total / max(n, 1)


def collect_aggregated(model, loader, device, vocab):
    """Aggregated letter posteriors + labels over a loader (for calibration)."""
    from alphabet.uncertainty import aggregate_letter_posterior
    model.eval()
    rows, labels = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["fused"].to(device), batch["mask"].to(device))
            post = aggregate_letter_posterior(
                outputs["letter_logits"], batch["mask"].to(device))
            rows.append(post.cpu())
            labels.extend(vocab.letters.index(m.label) for m in batch["metas"])
    if not rows:
        return torch.zeros(0, vocab.num_letters), []
    return torch.cat(rows), labels

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val,
                    dec_cfg, config_hash, temporal_config=None):
    """Checkpoint carries everything needed to rebuild the model exactly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_val": best_val,
        "vocab": DEFAULT_VOCABULARY.to_dict(),
        "decoder_config": dec_cfg.to_dict(),
        "temporal_config": asdict(temporal_config) if temporal_config else None,
        "config_hash": config_hash,
    }, path)


def load_model_from_checkpoint(path, device):
    """Rebuild model + vocab from a checkpoint (for inference / evaluation)."""
    from alphabet.heads import MultiTaskAlphabetModel
    from temporal.encoder import TemporalEncoderConfig

    ck = torch.load(path, map_location=device, weights_only=False)
    vocab = LetterVocabulary(letters=tuple(ck["vocab"]["letters"]))
    stored = ck["decoder_config"]
    dcfg = DecoderConfig(
        dataset_root=stored["dataset_root"],
        repr_cache_dir=stored["repr_cache_dir"],
        ablation=stored["ablation"],
        auxiliary=AuxiliaryConfig(**stored["auxiliary"]),
        training=TrainingConfig(**stored["training"]),
        uncertainty=UncertaintyConfig(**stored["uncertainty"]),
    )
    tcfg = TemporalEncoderConfig(**(ck.get("temporal_config") or {}))
    model = build_model(dcfg, tcfg, vocab)
    model.load_state_dict(ck["model"])
    model.to(device)
    model.eval()
    return model, vocab, ck

def train_decoder(dec_cfg, phase2_cfg,
                  limit_train=None, limit_val=None, limit_test=None,
                  max_steps=None, device=None, resume=True,
                  run_test=False) -> Dict:
    """Full Phase 5 training run; returns a JSON-serializable summary."""
    from alphabet.calibration import fit_temperature, select_abstention_thresholds

    set_seed(dec_cfg.training.seed)
    dev = resolve_device(device)
    vocab = DEFAULT_VOCABULARY
    ckpt_dir = Path(dec_cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(dec_cfg, phase2_cfg, vocab,
                              limit_train, limit_val, limit_test)
    if len(datasets["train"]) == 0:
        raise RuntimeError(
            "no training recordings available (empty manifest / missing "
            "representation cache); run ingest + preprocess + cache-reprs first")
    aux_kinds = []
    if dec_cfg.auxiliary.phone_enabled:
        aux_kinds.append("phone")
    if dec_cfg.auxiliary.viseme_enabled:
        aux_kinds.append("viseme")
    collate = make_collate_with_aux(aux_kinds, vocab)
    train_loader = make_train_loader(datasets["train"], dec_cfg, collate)
    val_loader = make_eval_loader(datasets["val"], dec_cfg, collate)

    model = build_model(dec_cfg, phase2_cfg.temporal, vocab).to(dev)
    loss_fn = MultitaskLoss(
        lambda_letter=1.0,
        lambda_phone=dec_cfg.auxiliary.lambda_phone if "phone" in aux_kinds else 0.0,
        lambda_viseme=dec_cfg.auxiliary.lambda_viseme if "viseme" in aux_kinds else 0.0,
        blank_index=vocab.blank_index,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=dec_cfg.training.lr,
                                  weight_decay=dec_cfg.training.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = dec_cfg.training.epochs * steps_per_epoch
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    warmup_steps = int(dec_cfg.training.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: warmup_cosine_lambda(
            step, total_steps, warmup_steps, dec_cfg.training.min_lr_ratio))

    start_epoch, best_val = 0, float("inf")
    last_path, best_path = ckpt_dir / "last.ckpt", ckpt_dir / "best.ckpt"
    history_path = ckpt_dir / "history.json"
    history = []
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        if ck.get("optimizer"):
            optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scheduler"):
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_val", float("inf"))
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        logger.info("resumed from %s at epoch %d", last_path, start_epoch)

    config_hash = ablation_config_hash(phase2_cfg.config_hash, dec_cfg.ablation)
    patience_left = dec_cfg.training.early_stopping_patience
    step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, dec_cfg.training.epochs):
        train_parts = run_train_epoch(
            model, train_loader, loss_fn, dev, optimizer=optimizer,
            scheduler=scheduler, grad_clip=dec_cfg.training.grad_clip,
            amp=dec_cfg.training.amp)
        step += steps_per_epoch
        val_loss = (compute_val_loss(model, val_loader, loss_fn, dev)
                    if len(datasets["val"]) else float("nan"))
        history.append({"epoch": epoch, "step": step,
                        **{k: v for k, v in train_parts.items()},
                        "val_loss": val_loss})
        logger.info("epoch %d train_total=%.4f val=%.4f lr=%.2e",
                    epoch, train_parts["total"], val_loss,
                    optimizer.param_groups[0]["lr"])
        improved = (val_loss == val_loss) and (val_loss < best_val)  # NaN-safe
        if improved:
            best_val = val_loss
            patience_left = dec_cfg.training.early_stopping_patience
            save_checkpoint(best_path, model, optimizer, scheduler, epoch,
                            best_val, dec_cfg, config_hash, phase2_cfg.temporal)
        else:
            patience_left -= 1
        save_checkpoint(last_path, model, optimizer, scheduler, epoch,
                        best_val, dec_cfg, config_hash, phase2_cfg.temporal)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if max_steps is not None and step >= max_steps:
            break
        if patience_left <= 0:
            logger.info("early stopping at epoch %d (best val %.4f)",
                        epoch, best_val)
            break

    # ---- calibration on VALIDATION only (§14); test untouched (§16) ----
    calibration: Dict = {}
    if len(datasets["val"]):
        _load_best_if_exists(best_path, model, dev)
        post_rows, labels = collect_aggregated(model, val_loader, dev, vocab)
        temperature = (dec_cfg.uncertainty.temperature
                       if dec_cfg.uncertainty.temperature is not None
                       else (fit_temperature(post_rows.log(), torch.tensor(labels))
                             if post_rows.shape[0] else 1.0))
        ev = evaluate_model(model, val_loader, vocab, temperature)
        recs = ev["records"]
        thresholds = (select_abstention_thresholds(
            [r.confidence for r in recs], [r.margin for r in recs],
            [r.correct for r in recs],
            coverage_target=dec_cfg.uncertainty.coverage_target)
            if recs else {"confidence_threshold": 0.0, "margin_threshold": 0.0})
        calibration = {"temperature": temperature, **thresholds}
        (ckpt_dir / "calibration.json").write_text(
            json.dumps(_jsonable(calibration), indent=2), encoding="utf-8")

    summary = {
        "epochs_run": len(history),
        "best_val_loss": best_val,
        "parameter_counts": model.parameter_counts(),
        "calibration": calibration,
        "device": str(dev),
        "aux_heads": aux_kinds,
        "ablation": dec_cfg.ablation,
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        "n_test": len(datasets["test"]),
    }
    (ckpt_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    if run_test and len(datasets["test"]):
        from alphabet.evaluate import write_failure_records
        test_eval = evaluate_model(
            model, make_eval_loader(datasets["test"], dec_cfg, collate), vocab,
            calibration.get("temperature", 1.0),
            thresholds=(calibration if calibration else None))
        write_failure_records(test_eval["records"],
                              ckpt_dir / "test_failures.jsonl")
        (ckpt_dir / "test_metrics.json").write_text(
            json.dumps(_jsonable(test_eval["metrics"]), indent=2),
            encoding="utf-8")
    return summary


def _load_best_if_exists(path, model, device):
    p = Path(path)
    if not p.exists():
        return None
    ck = torch.load(p, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    return ck


def _jsonable(obj):
    import numpy as _np
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (_np.floating, _np.integer)):
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    return obj

