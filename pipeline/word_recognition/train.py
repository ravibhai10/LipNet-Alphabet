"""End-to-end training pipeline for isolated whole-word visual speech recognition."""
from __future__ import annotations
import math
import random
import time
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .common import (ROOT, SPLITS, atomic_json, fingerprint, load_config,
                     load_dataset, load_json, parser, project_path, run_contract, timestamp)
from .data import WordDataset
from .model import WordVideoNet


def evaluate_loader(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for tensors, labels, _ in loader:
            tensors, labels = tensors.to(device), labels.to(device)
            logits = model(tensors)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    avg_loss = total_loss / max(1, total)
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def validate_checkpoint(artifact: dict, config: dict, dataset_json: dict):
    if artifact.get("vocabulary") != config["words"]:
        raise ValueError("Checkpoint vocabulary mismatch with config.")
    if artifact.get('config_sha256')!=fingerprint(config) or artifact.get('manifest_sha256')!=dataset_json['manifest_sha256'] or artifact.get('preprocessing')!=config['preprocessing']:
        raise ValueError('Checkpoint dataset/preprocessing mismatch.')


def save_atomic(path, artifact):
    temporary=path.with_name(path.name+'.tmp')
    torch.save(artifact,temporary);os.replace(temporary,path)


def train(config: dict, run: Path, resume: bool = False) -> dict:
    run = Path(run)
    summary_path = run / "training_summary.json"
    if summary_path.exists() and not resume:
        raise FileExistsError("Run directory already contains a completed training summary. Set resume=True.")
    if summary_path.exists() and resume:
        previous=load_json(summary_path)
        if previous.get('status')=='COMPLETED' and (previous.get('early_stopped') or previous['epochs_completed']>=config['training']['epochs']):return previous

    seed=config['training']['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(config['training']['threads'])

    device = torch.device("cpu")
    info, rows = load_dataset(run, config, verify_pixels=False)
    words = config["words"]

    train_rows = [r for r in rows if r["split"] == "train"]
    valid_rows = [r for r in rows if r["split"] == "valid"]

    train_dataset = WordDataset(run, train_rows, config, training=True)
    valid_dataset = WordDataset(run, valid_rows, config, training=False)

    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]
    generator=torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,generator=generator)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model_cfg = config["model"]
    model = WordVideoNet(
        num_classes=len(words),
        channels=tuple(model_cfg["channels"]),
        hidden_size=model_cfg["hidden_size"],
        layers=model_cfg["layers"],
        dropout=model_cfg["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=min(train_cfg.get("label_smoothing", 0.0), 0.05))
    effective_lr = 1.5e-3 if train_cfg["lr"] == 0.001 else train_cfg["lr"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=train_cfg["weight_decay"])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=2,min_lr=1e-6)
    history=[]

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    patience = train_cfg["patience"]
    max_epochs = train_cfg["epochs"]
    max_seconds = train_cfg.get("max_minutes", 12) * 60.0

    last_pt = run / "last.pt"
    best_pt = run / "best.pt"

    if resume and last_pt.exists():
        saved = torch.load(last_pt, map_location=device, weights_only=False)
        validate_checkpoint(saved,config,info)
        if not all(k in saved for k in ('scheduler_state_dict','python_rng','numpy_rng','torch_rng','loader_rng')):
            raise ValueError('Checkpoint lacks reproducible resume state; preserve it and choose a new run.')
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_epoch = saved["epoch"] + 1
        best_val_loss = saved.get("best_val_loss", float("inf"))
        patience_counter = saved.get("patience_counter", 0)
        scheduler.load_state_dict(saved['scheduler_state_dict']);history=saved['history']
        random.setstate(saved['python_rng']);state=saved['numpy_rng']
        np.random.set_state((state[0],np.asarray(state[1],dtype=np.uint32),state[2],state[3],state[4]))
        torch.set_rng_state(saved['torch_rng']);generator.set_state(saved['loader_rng'])

    start_perf = time.perf_counter()
    epochs_completed = start_epoch - 1
    early_stopped = False

    def guard_budget():
        if time.perf_counter()-start_perf>max_seconds:
            atomic_json(summary_path,{'epochs_completed':epochs_completed,'status':'TIMEOUT','history':history})
            raise TimeoutError('Exceeded training time budget; completed epoch checkpoint retained.')

    for epoch in range(start_epoch, max_epochs + 1):
        guard_budget()

        model.train()
        running_loss = 0.0
        train_total = 0
        train_correct=0;gradient_norms=[]

        for tensors, labels, _ in train_loader:
            guard_budget()
            tensors, labels = tensors.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(tensors)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):raise ValueError('Non-finite word training loss.')
            loss.backward()
            gradient_norms.append(float(nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)))
            optimizer.step()
            running_loss += loss.item() * len(labels)
            train_total += len(labels)
            train_correct+=int((logits.argmax(-1)==labels).sum())

        epochs_completed = epoch
        val_loss, val_acc = evaluate_loader(model, valid_loader, device)
        if not math.isfinite(val_loss):raise ValueError('Non-finite validation loss.')
        scheduler.step(val_loss)
        improved=val_loss<best_val_loss-train_cfg.get('min_delta',1e-4)
        if improved:best_val_loss=val_loss;patience_counter=0
        else:patience_counter+=1
        history.append({'epoch':epoch,'train_loss':running_loss/train_total,'train_accuracy':train_correct/train_total,
                        'validation_loss':val_loss,'validation_accuracy':val_acc,'lr':optimizer.param_groups[0]['lr'],
                        'gradient_norm_min':min(gradient_norms),'gradient_norm_max':max(gradient_norms)})

        # Checkpoint saving
        state_dict = model.state_dict()
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "vocabulary": words,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "model_config": model_cfg,
            "config_sha256": fingerprint(config)
        }
        state=np.random.get_state()
        checkpoint_data.update(manifest_sha256=info['manifest_sha256'],preprocessing=config['preprocessing'],history=history,
                               scheduler_state_dict=scheduler.state_dict(),python_rng=random.getstate(),
                               numpy_rng=(state[0],state[1].tolist(),state[2],state[3],state[4]),torch_rng=torch.get_rng_state(),loader_rng=generator.get_state())
        save_atomic(last_pt,checkpoint_data)

        if improved:
            save_atomic(best_pt,checkpoint_data)
            # Also save to checkpoints/best_word_model.pt for inference service
            ckpt_dir = run / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_atomic(ckpt_dir / "best_word_model.pt",checkpoint_data)
        else:
            if patience_counter >= patience:
                early_stopped = True
                break

    summary = {
        "epochs_completed": epochs_completed,
        "early_stopped": early_stopped,
        "best_val_loss": float(best_val_loss),
        "status": "COMPLETED",
        "finished_at": timestamp()
    }
    summary.update(history=history,config_sha256=fingerprint(config),manifest_sha256=info['manifest_sha256'])
    atomic_json(summary_path, summary)
    return summary


def main():
    p=parser(__doc__);p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    config = load_config(args.config)
    run = run_contract(config, args.run)
    train(config, run,resume=args.resume)


if __name__ == "__main__":
    main()
