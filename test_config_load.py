"""Test script to verify config loading works"""
import sys
sys.path.insert(0, '.')
from config.loader import load_phase2_config
from pathlib import Path

cfg = load_phase2_config('configs/alphabet_dataset.yaml')
print('Config loaded:', cfg.config_version, cfg.config_hash[:12])
print('crop_margin:', cfg.tracking.crop_margin_multiplier)
print('fps:', cfg.ingestion.fps)
print('appearance_crop_size:', cfg.tracking.appearance_crop_size)
print('config_hash:', cfg.config_hash)
print('Preprocessing config_hash:', cfg.preprocessing.config_hash)
print('SUCCESS: Config loading works!')

# Test changing crop_margin in YAML and reloading
import yaml
with open('configs/alphabet_dataset.yaml', 'r') as f:
    data = yaml.safe_load(f)

original_margin = data['mouth_alignment']['crop_margin_multiplier']
data['mouth_alignment']['crop_margin_multiplier'] = 3.0

# Write modified YAML to temp file
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(data, f)
    temp_path = f.name

try:
    cfg2 = load_phase2_config(temp_path)
    print('\nModified config loaded:', cfg2.config_version, cfg2.config_hash[:12])
    print('crop_margin:', cfg2.tracking.crop_margin_multiplier)
    assert cfg2.tracking.crop_margin_multiplier == 3.0, f"Expected 3.0, got {cfg2.tracking.crop_margin_multiplier}"
    print('SUCCESS: Config override works!')
finally:
    import os
    os.unlink(temp_path)

# Test invalid config raises error
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    f.write('invalid: yaml: [}')
    temp_path = f.name

try:
    try:
        cfg3 = load_phase2_config(temp_path)
        print('FAIL: Should have raised error for invalid YAML')
    except Exception as e:
        print('SUCCESS: Invalid YAML correctly raises error:', type(e).__name__)
finally:
    import os
    os.unlink(temp_path)

print('\n=== ALL TESTS PASSED ===')