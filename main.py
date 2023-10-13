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


for file in os.listdir("./assets/configs"):
    print(file)
    # Itt egy tanítás
    # Majd egy teszt
