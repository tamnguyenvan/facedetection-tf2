import os
import yaml
import collections

# Load config
curr_dir = os.path.dirname(__file__)
cfg_file = os.path.join(curr_dir, 'config.yaml')
with open(cfg_file) as f:
    cfg = yaml.load(f, yaml.FullLoader)
