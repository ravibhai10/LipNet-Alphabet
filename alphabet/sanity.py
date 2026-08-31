"""Phase 5 sanity tests: real-data pipeline check (§25) + tiny overfit (§26).

REAL-DATA SANITY (§25):
    Uses a tiny subset of REAL recordings (manifest + Phase 3 cache):
    Phase 2 -> Phase 3 (cached fused features) -> Phase 4 encoder -> letter
    decoder. Verifies: no NaN, correct shapes, correct target encoding,
    correct CTC lengths, forward pass, backward pass, decoding.
    If the dataset has no recordings, it reports a BLOCKER instead of
    fabricating a pass.

TINY-SUBSET OVERFIT (§26):
    Purpose is NOT performance — it verifies data -> target -> model -> loss
    -> gradients are connected correctly by driving training loss ~0 on a
    tiny controlled set. Because the dataset currently contains NO real
    recordings, the harness defaults to clearly-labelled SYNTHETIC fused
    sequences (per-letter temporal prototypes); the same harness runs on
    real cached features automatically as soon as they exist
    (``use_synthetic=False``), so no code path changes when data arrives.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from alphabet.config import DecoderConfig, ablation_config_hash
from alphabet.decoding import ClassMappingGreedyCTCDecoder
from alphabet.losses import MultitaskLoss
from alphabet.targets import collate_targets
from alphabet.train import (
    build_model, set_seed, resolve_device, warmup_cosine_lambda,
)
from alphabet.vocab import DEFAULT_VOCABULARY, LetterVocabulary

logger = logging.getLogger(__name__)


def dataset_has_recordings(dataset_root: Path) -> bool:
    root = Path(dataset_root)
    manifest = root / "manifest.jsonl"
    recordings = root / "recordings"
    if not manifest.exists() or not recordings.exists():
        return False
    return any(recordings.iterdir())


def make_synthetic_fused(n_per_letter: int = 2, t_frames: int = 32,
                         dim: int = 192, letters: str = "ABCDEFGH",
                         seed: int = 0):
    """Synthetic fused sequences with per-letter temporal prototypes.

    Structure per recording: [silence] [letter prototype ridge] [silence] —
    the same temporal shape CTC must learn on real data. Clearly synthetic;
    NEVER presented as real-data results.
    """
    rng = np.random.default_rng(seed)
    sequences, labels = [], []
    for li, letter in enumerate(letters):
        proto = rng.normal(0.0, 1.0, size=(dim,)).astype(np.float32)
        for _ in range(n_per_letter):
            seq = np.zeros((t_frames, dim), dtype=np.float32)
            start = int(t_frames * 0.25)
            end = int(t_frames * 0.75)
            ramp = np.linspace(0.0, 1.0, end - start, dtype=np.float32)
            envelope = np.minimum(ramp, ramp[::-1])  # smooth rise/fall
            seq[start:end] = proto[None, :] * envelope[:, None]
            seq += rng.normal(0.0, 0.05, size=seq.shape).astype(np.float32)
            sequences.append(seq)
            labels.append(letter)
    return sequences, labels
def real_data_sanity(dec_cfg: DecoderConfig, phase2_cfg, n_max: int = 8,
                     device: Optional[str] = None) -> Dict:
    """§25 sanity on a tiny subset of REAL recordings (or a blocker report)."""
    from alphabet.dataset import AlphabetSequenceDataset, collate_batch, \
        load_manifest_entries, cache_key

    checks: List[Dict] = []

    def check(name, ok, info=""):
        checks.append({"check": name, "ok": bool(ok), "info": info})

    root = Path(phase2_cfg.dataset_root)
    if not dataset_has_recordings(root):
        return {
            "status": "BLOCKED",
            "reason": ("dataset contains no recordings "
                       f"({root}); ingest + preprocess real recordings first"),
            "checks": checks,
        }
    cache_dir = Path(dec_cfg.repr_cache_dir) / dec_cfg.ablation
    entries = load_manifest_entries(root, split="train")[:n_max]
    if not entries:
        return {"status": "BLOCKED", "reason": "no train-split entries in manifest",
                "checks": checks}

    vocab = DEFAULT_VOCABULARY
    ds = AlphabetSequenceDataset(
        entries, dataset_root=root, repr_cache_dir=cache_dir,
        expected_config_hash=ablation_config_hash(
            phase2_cfg.config_hash, dec_cfg.ablation),
        vocab=vocab)
    dev = resolve_device(device)
    model = build_model(dec_cfg, phase2_cfg.temporal, vocab).to(dev)
    loss_fn = MultitaskLoss(blank_index=vocab.blank_index)
    batch = collate_batch([ds[i] for i in range(len(ds))])

    fused = batch["fused"].to(dev)
    mask = batch["mask"].to(dev)
    check("fused shape [B,T,192]", fused.ndim == 3 and fused.shape[-1] == 192,
          str(tuple(fused.shape)))
    check("mask shape matches [B,T]", mask.shape == fused.shape[:2])
    check("no NaN/Inf in inputs", bool(torch.isfinite(fused).all()))
    check("target ids valid (1..26, no blank)",
          bool(((batch["targets"] == 0) |
                ((batch["targets"] >= 1) & (batch["targets"] <= 26))).all()),
          str(batch["targets"].tolist()))
    check("target lengths == label lengths",
          bool((batch["target_lengths"] ==
                torch.tensor([len(v) for v in
                              [vocab.encode(m.label) for m in batch["metas"]]])).all()))
    check("input_lengths == mask sums",
          bool((batch["input_lengths"] == mask.sum(-1).cpu()).all()))

    outputs = model(fused, mask)
    check("letter logits [B,T,27]",
          tuple(outputs["letter_logits"].shape) ==
          (fused.shape[0], fused.shape[1], vocab.size),
          str(tuple(outputs["letter_logits"].shape)))
    check("no NaN/Inf in logits", bool(torch.isfinite(outputs["letter_logits"]).all()))
    loss = loss_fn(outputs, mask, batch["targets"], batch["target_lengths"])["total"]
    check("CTC loss finite", bool(torch.isfinite(loss)), f"{float(loss):.4f}")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check("backward produced gradients", len(grads) > 0,
          f"{len(grads)} grad tensors")
    check("no NaN/Inf in gradients",
          bool(all(torch.isfinite(g).all() for g in grads)))
    decoded = ClassMappingGreedyCTCDecoder(
        vocab.index_to_letter, vocab.blank_index).decode_batch(
            outputs["letter_logits"].detach(), batch["input_lengths"])
    check("greedy decoding runs, output is A-Z or empty",
          all(set(d.text) <= set(vocab.letters) for d in decoded),
          str([d.text for d in decoded]))
    failed = [c for c in checks if not c["ok"]]
    return {"status": "PASS" if not failed else "FAIL",
            "n_recordings": len(ds), "loss": float(loss.item()),
            "checks": checks, "failed": failed}

def tiny_overfit(dec_cfg: DecoderConfig, phase2_cfg,
                 n_per_letter: int = 2, t_frames: int = 32,
                 letters: str = "ABCDEFGH", steps: int = 300,
                 lr: float = 5e-3, seed: int = 0,
                 use_synthetic: bool = True,
                 device: Optional[str] = None) -> Dict:
    """§26: overfit a tiny controlled subset; verifies the wiring end-to-end.

    Returns {status, data_source, final_loss, final_accuracy, loss_curve}.
    Success criterion: training letter accuracy reaches 1.0 (loss ~0) —
    proving data -> target -> model -> loss -> gradient connectivity.
    """
    set_seed(seed)
    dev = resolve_device(device)
    vocab = DEFAULT_VOCABULARY

    if use_synthetic or not dataset_has_recordings(Path(phase2_cfg.dataset_root)):
        sequences, labels = make_synthetic_fused(
            n_per_letter=n_per_letter, t_frames=t_frames, dim=192,
            letters=letters, seed=seed)
        data_source = "synthetic_prototypes (no real recordings in dataset)"
    else:
        from alphabet.dataset import AlphabetSequenceDataset, \
            load_manifest_entries
        entries = load_manifest_entries(Path(phase2_cfg.dataset_root),
                                        split="train")
        take = entries[: n_per_letter * len(letters)]
        ds = AlphabetSequenceDataset(
            take, dataset_root=Path(phase2_cfg.dataset_root),
            repr_cache_dir=Path(dec_cfg.repr_cache_dir) / dec_cfg.ablation,
            expected_config_hash=ablation_config_hash(
                phase2_cfg.config_hash, dec_cfg.ablation),
            vocab=vocab)
        sequences = [ds[i]["fused"] for i in range(len(ds))]
        labels = [ds[i]["meta"].label for i in range(len(ds))]
        data_source = "real recordings (Phase 3 cache)"

    batch = {
        "fused": torch.from_numpy(np.stack(sequences)).to(dev),
        "mask": torch.ones(len(sequences), t_frames, dtype=torch.bool).to(dev),
        "targets": None, "target_lengths": None,
        "metas": None,
    }
    tb = collate_targets(vocab, labels)
    batch["targets"] = tb.targets.to(dev)
    batch["target_lengths"] = tb.target_lengths.to(dev)

    # Small model: overfit wiring test, not the production configuration.
    from dataclasses import replace as dc_replace
    tcfg = dc_replace(phase2_cfg.temporal, num_layers=2, ffn_dim=256)
    model = build_model(dec_cfg, tcfg, vocab).to(dev)
    loss_fn = MultitaskLoss(blank_index=vocab.blank_index)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    decoder = ClassMappingGreedyCTCDecoder(vocab.index_to_letter,
                                           vocab.blank_index)
    loss_curve, final_acc = [], 0.0
    for step in range(steps):
        outputs = model(batch["fused"], batch["mask"])
        parts = loss_fn(outputs, batch["mask"], batch["targets"],
                        batch["target_lengths"])
        optimizer.zero_grad(set_to_none=True)
        parts["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_curve.append(float(parts["total"].item()))
        if not np.isfinite(loss_curve[-1]):
            return {"status": "FAIL", "reason": "non-finite loss",
                    "data_source": data_source,
                    "loss_curve_tail": loss_curve[-10:]}
        if (step + 1) % 25 == 0 or step == steps - 1:
            with torch.no_grad():
                out = model(batch["fused"], batch["mask"])
            dec = decoder.decode_batch(out["letter_logits"], batch["mask"].sum(-1).cpu())
            preds = [d.text for d in dec]
            final_acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
            logger.info("overfit step %d loss=%.4f acc=%.2f", step,
                        loss_curve[-1], final_acc)
        if final_acc == 1.0 and loss_curve[-1] < 0.05:
            break

    ok = final_acc == 1.0 and loss_curve[-1] < 0.1
    return {
        "status": "PASS" if ok else "FAIL",
        "data_source": data_source,
        "n_sequences": len(sequences),
        "steps_run": len(loss_curve),
        "final_loss": loss_curve[-1],
        "final_accuracy": final_acc,
        "loss_first5": loss_curve[:5],
        "loss_last5": loss_curve[-5:],
    }

