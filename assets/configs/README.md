
# Usage

In thi folder you can place json files containing a model training and testing parameters.
Each configuration will run and will be saved to results folder separately.

# Example json

    {
        "roboflow": {
            "api_key": "yOpj3WsdetMKERI8oDmX",
            "workspace": "development-2023",
            "project": "consumption-meter-numbers-colore",
            "version": 6,
            "format": "yolov8"
        },
        "model": {
	        "augment": [
                {
                    "name": "brightness",
                    "params": {
                        "value": 1
                    }
                }
            ],
            "preproc": [
                {
                    "name": "auto_brightness_contrast_grayscale",
                    "params": {
                        "clip_hist_percent": 1
                    }
                }
            ],
            "name": "NumbersWaterMeters",
            "model": "yolov8x.pt",
            "size": 224,
            "batch": 16,
            "epoch": 100,
            "device": 0,
            "pretrained": true
        },
        "test": {
            "dataset_path": "./assets/test_dataset/test_0"
        },
        "description": "Lorem ipsum"
    }
## Explanation

 - **Roboflow:**
 The dataset is annotated and stored on roboflow. Do not use any preprocessing or augmentation beacuse it will done locally! You can find the required parameters in the export tab on robofolow
 - **Model**
	 - Augment: These are the methods used for image augmentation. **The *"name"* is transalted to a method**, and 	paramters passed. In this list you can add any augmentation method that you can find in `/library/augmentators.py` file.
	 - Prepoc: These are the methods used for image preprocessing. **The *"name"* is transalted to a method**, and 	paramters passed. In this list you can add any preprocessing method that you can find in `/library/preprocessing.py` file.
	 - Etc: The other parameters are used for training configuration. You can find the documentation for these arguments here: [https://docs.ultralytics.com/modes/train/#arguments](https://docs.ultralytics.com/modes/train/#arguments)
	 - Name: Run will be saved with this name. If the name already exists, an index number will be added

- **Tests:**
	- Datset_path: Here you can specify a path to a folder containing test images. Test dataset are store in `/assets/test_datasets` folder
- **Description:** Some description about the model, parameters or expectations.

  
