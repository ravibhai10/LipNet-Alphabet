"""Read-only, memory-mapped official AV-HuBERT loading for small Windows RAM.

Uses PyTorch's meta/assign loading path while retaining Fairseq's config,
task-state, model-state upgrades and pruning. No optimizer tensors are copied.
The normal loader remains the default; this is an explicit offline CPU option.
"""
from pathlib import Path
import torch


def load_encoder(path):
    from omegaconf import OmegaConf
    from fairseq import checkpoint_utils, tasks
    from fairseq.dataclass.utils import convert_namespace_to_omegaconf
    path=Path(path)
    # Trusted local official checkpoint only, never an uploaded pickle.
    import hashlib
    with __import__('pathlib').Path(path).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!='dee8b0651dbea3a21cda0f2c52b956e3a878d5f6ee503d427d6a4044bb95c9cc':
        raise ValueError('Unregistered encoder pickle rejected; only the preserved trusted encoder is supported.')
    state=torch.load(str(path),map_location='cpu',mmap=True,weights_only=False)
    # Fairseq's compatibility upgrade needs the small history/extra metadata.
    # Only the large optimizer tensors are unnecessary for frozen inference.
    state.pop('last_optimizer_state',None)
    if state.get('cfg') is not None:
        state['cfg']=OmegaConf.create(state['cfg'],flags={'allow_objects':True})
        OmegaConf.set_struct(state['cfg'],True)
    state=checkpoint_utils._upgrade_state_dict(state)
    cfg=convert_namespace_to_omegaconf(state['args']) if state.get('args') is not None else state['cfg']
    task=tasks.setup_task(cfg.task)
    if 'task_state' in state:task.load_state_dict(state['task_state'])
    with torch.device('meta'):
        model=task.build_model(cfg.model)
    model.upgrade_state_dict(state['model'])
    weights=checkpoint_utils.prune_state_dict(state['model'],cfg.model)
    torch.nn.Module.load_state_dict(model,weights,strict=True,assign=True)
    if any(t.device.type!='cpu' for t in (*model.parameters(),*model.buffers())):
        raise RuntimeError('The assigned encoder contains non-CPU or unmaterialized tensors.')
    return [model.float().eval().requires_grad_(False)],cfg,task
