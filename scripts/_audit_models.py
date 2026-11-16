import os
import torch

paths = [
    'models/alphabet_classifier.pt',
    'models/external/avhubert/base_vox_iter5.pt',
    'runs/finetune/lrw_500words/alphabet_classifier.pt',
    'runs/finetune/lrw_500words/best_model.pt',
    'runs/finetune/lrw_500words/checkpoint_epoch_1.pt'
]

for p in paths:
    if not os.path.exists(p):
        print(f"=== {p}: NOT FOUND ===")
        continue
    sz = os.path.getsize(p)
    print(f"\n=== {p} ({sz:,} bytes) ===")
    try:
        data = torch.load(p, map_location='cpu')
        if isinstance(data, dict):
            print("  Type: dict, keys:", list(data.keys())[:8])
            if 'model_state_dict' in data:
                state = data['model_state_dict']
            elif 'state_dict' in data:
                state = data['state_dict']
            else:
                state = data
        else:
            print("  Type:", type(data))
            state = data
        
        if hasattr(state, 'items'):
            keys = list(state.keys())
            print(f"  Total weight tensors: {len(keys)}")
            for k in keys[-4:]:
                t = state[k]
                if hasattr(t, 'shape'):
                    print(f"    {k}: shape {tuple(t.shape)}")
                else:
                    print(f"    {k}: {type(t)}")
    except Exception as e:
        print(f"  Load error: {e}")
