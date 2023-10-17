import os
import cv2
import time
import shutil
from ultralytics import YOLO
from roboflow import Roboflow
from types import SimpleNamespace
import library.preprocessors as pre
import library.augmentators as aug


class Trainer:

    def __init__(self, config: SimpleNamespace) -> None:
        '''
        Initializes a Trainer instance with the given configuration.
        
        :param config: A configuration object of type SimpleNamespace.
        '''
        self.config = config
        self.model = YOLO(config.model.model)

        self.start()

    def start(self) -> None:
        '''
        Starts the training process, including data download, dataset movement, augmentation, preprocessing, and training.
        Runs on init.
        '''
        self.download_dataset()
        self.move_dataset()
        #self.augment_images()
        self.preprocess_images()
        self.train()

    def download_dataset(self) -> None:
        '''
        Downloads a dataset using Roboflow.    
        '''
        rf = Roboflow(api_key=self.config.roboflow.api_key)
        project = rf.workspace(self.config.roboflow.workspace).project(self.config.roboflow.project)
        self.dataset = project.version(self.config.roboflow.version).download("yolov8")

    def move_dataset(self) -> None:
        '''
        Moves the downloaded dataset to the specified destination directory.
        '''
        self.source_path = self.dataset.location
        self.destination_path = f"assets/datasets/{self.dataset.location.split('/')[-1]}"

        # Check if the destination directory already exists, and if so, delete it
        if os.path.exists(self.destination_path):
            shutil.rmtree(self.destination_path)

        # Move the entire dataset directory to the destination directory
        shutil.move(self.source_path, self.destination_path)

    def augment_images(self) -> None:
        images_list = pre.find_and_read_jpg_images(self.destination_path)

        processed = []
        for augment_item in self.config.model.augment:
            name = augment_item.name  # Name of the method
            params = augment_item.params  # Parameters of the method

            # Convert SimpleNamespace to a dictionary
            params_dict = vars(params)

            # Add the image list to the parameters
            params_dict["images_list"] = images_list            

            # Call the method
            method = getattr(aug, name)      
            images_list = method(**params_dict)

            # Append result to rest
            for i in images_list :
                processed.append(i)

        # Save augmented images
        for path, image in processed:
            cv2.imwrite(os.path.join(path, f"{round(time.time() * 1000)}.jpg"), image)            


    def preprocess_images(self) -> None:
        '''
        Preprocess a list of images using methods specified in the configuration.                
        '''
        images_list = pre.find_and_read_jpg_images(self.destination_path)
        
        for preproc_item in self.config.model.preproc:
            name = preproc_item.name  # Name of the method
            params = preproc_item.params  # Parameters of the method

            # Convert SimpleNamespace to a dictionary
            params_dict = vars(params)

            # Add the image list to the parameters
            params_dict["images_list"] = images_list            

            # Call the method
            method = getattr(pre, name)        
            images_list = method(**params_dict)

        # Save preprocessed images
        for path, image in images_list:
            cv2.imwrite(path, image)
        

    def train(self):
        '''
        Trains the YOLO model with the specified parameters.
        
        :return: Result of ultralytics fit method 
        '''
        return self.model.train(
            data=f"assets/datasets/{self.dataset.location.split('/')[-1]}/data.yaml",
            project="assets/results",
            imgsz=self.config.model.size,
            epochs=self.config.model.epoch,
            batch=self.config.model.batch,
            name=self.config.model.name,
            device=self.config.model.device,
            pretrained=self.config.model.pretrained,
            patience=self.config.model.patience,
            save=self.config.model.save,
            save_period=self.config.model.save_period,
            cache=self.config.model.cache,
            workers=self.config.model.workers,
            exist_ok=self.config.model.exist_ok,
            optimizer=self.config.model.optimizer,
            verbose=self.config.model.verbose,
            seed=self.config.model.seed,
            deterministic=self.config.model.deterministic,
            single_cls=self.config.model.single_cls,
            rect=self.config.model.rect,
            cos_lr=self.config.model.cos_lr,
            close_mosaic=self.config.model.close_mosaic,
            resume=self.config.model.resume,
            amp=self.config.model.amp,
            fraction=self.config.model.fraction,
            profile=self.config.model.profile,
            lr0=self.config.model.lr0,
            lrf=self.config.model.lrf,
            momentum=self.config.model.momentum,
            weight_decay=self.config.model.weight_decay,
            warmup_epochs=self.config.model.warmup_epochs,
            warmup_momentum=self.config.model.warmup_momentum,
            warmup_bias_lr=self.config.model.warmup_bias_lr,
            box=self.config.model.box,
            cls=self.config.model.cls,
            dfl=self.config.model.dfl,
            pose=self.config.model.pose,
            kobj=self.config.model.kobj,
            label_smoothing=self.config.model.label_smoothing,
            nbs=self.config.model.nbs,
            overlap_mask=self.config.model.overlap_mask,
            mask_ratio=self.config.model.mask_ratio,
            dropout=self.config.model.dropout,
            val=self.config.model.val
        )