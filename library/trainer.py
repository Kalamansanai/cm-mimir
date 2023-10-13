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


class Trainer:

    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config
        self.model = YOLO(config.model.model)

        self.start()

    def start(self) -> None:
        self.download_dataset()
        self.augment()
        self.preprocess()
        self.train()

    def download_dataset(self) -> None:
        rf = Roboflow(api_key=self.config.roboflow.api_key)
        project = rf.workspace(self.config.roboflow.workspace).project(self.config.roboflow.project)
        self.dataset = project.version(self.config.roboflow.version).download(self.config.roboflow.format)

        # TODO move folder to assets and correct data.yaml

    def augment(self) -> None:
        pass

    def preprocess(self) -> None:
        pass

    def train(self) -> None:
        results = self.model.train(
            data=f"{self.dataset.location}/data.yaml",
            project="assets/results",
            imgsz=self.config.model.size,
            epochs=self.config.model.epoch,
            batch=self.config.model.batch,
            name=self.config.model.name,
            device=self.config.model.device,
            pretrained=self.config.model.pretrained
        )