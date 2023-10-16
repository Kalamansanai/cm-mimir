import os
import json
import shutil
from types import SimpleNamespace
from library.trainer import Trainer

# Create directory structure if not exists

base_dir = "assets"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

subdirectories = ["configs", "results", "datasets", "test_datasets"]
for sub_dir in subdirectories:
    dir_path = os.path.join(base_dir, sub_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# Iterate over configs, train and test models

cnfs = "./assets/configs/"
for path in os.listdir(cnfs):
    with open(cnfs+path) as file:     
        config = json.load(file, object_hook=lambda d: SimpleNamespace(**d))

    
    trn = Trainer(config)
    # TODO run tests
    