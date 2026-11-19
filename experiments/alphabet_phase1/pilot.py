"""Two synthetic optimizer steps for compute/memory only. No dataset training."""
import argparse
import json
from pathlib import Path
import time
from .resources import guard, snapshot, process_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', choices=['reference','bilstm','tcn','attention'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Existing pilot receipt is protected')
    report = {'scope':'Synthetic microbatch-2 compute pilot, NOT recognition training/evaluation',
              'architecture':args.architecture,'dataset_samples':0,'checkpoint_saved':False,'full_training_started':False}
    try:
        report['before_import'] = guard(512)
        import torch
        from .models import build
        torch.set_num_threads(1)
        torch.manual_seed(42)
        guard(256)
        model = build(args.architecture).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        inputs = torch.rand(2,30,1,140,46)
        targets = torch.tensor([0,1])  # Software-only A/B index targets, not source annotations.
        times = []
        for _ in range(2):
            guard(256)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            times.append(time.perf_counter() - start)
        report.update(status='passed',optimizer_steps=2,microbatch=2,input_shape=list(inputs.shape),
            output_shape=list(logits.shape),parameters=sum(p.numel() for p in model.parameters()),
            step_seconds=times,seconds_per_synthetic_clip=sum(times)/4,process_memory=process_memory(),
            resources_after=snapshot(),torch_version=torch.__version__,threads=1,
            limitation='Cold two-step CPU proxy; excludes real decoding/crops, validation, sustained thermals and accuracy.')
    except Exception as exc:
        report.update(status='stopped',error=f'{type(exc).__name__}: {exc}',resources_after=snapshot())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(report))
    return 0 if report['status']=='passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
