import os
import json
from types import SimpleNamespace
from library.trainer import Trainer

base_dir = "assets"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

subdirectories = ["configs", "results", "test_datasets"]
for sub_dir in subdirectories:
    dir_path = os.path.join(base_dir, sub_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

cnfs = "./assets/configs/"
for path in os.listdir(cnfs):
    with open(cnfs+path) as file:     
        config = json.load(file, object_hook=lambda d: SimpleNamespace(**d))

    
    trn = Trainer(config)
    # TODO run tests
    