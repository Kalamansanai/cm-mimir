import os
import cv2
import glob
import json
import numpy as np
from PIL import Image
from ultralytics import YOLO
from roboflow import Roboflow
import matplotlib.pyplot as plt
from types import SimpleNamespace
from wandb.integration.yolov8 import add_callbacks as add_wandb_callbacks


class Trainer:

    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config

        self.start()

    def start(self) -> None:
        self.download_dataset()
        self.preprocess()
        self.train()

    def download_dataset(self) -> None:
        rf = Roboflow(api_key=self.config.roboflow.api_key)
        project = rf.workspace(self.config.roboflow.workspace).project(self.config.roboflow.project)
        self.dataset = project.version(self.config.roboflow.version).download(self.config.roboflow.format)

    def preprocess(self) -> None:
        pass

    def train(self) -> None:
        pass