
  

# Documentation

 This framework is used to train and test multiple YOLO v8 object detection models with different configurations in a row. All the metrics will be saved separetly. You can define as many configurations in a json file as you like, and all of them will be executed in a row.
 Place your desired configuration json files in the `assets/configs` folder.

# Example json
  

    {
    "roboflow":{
        "api_key":"yOpj3WsdetMKERI8oDmX",
        "workspace":"development-2023",
        "project":"consumption-meter-numbers-colore",
        "version":7
    },
    "model":{
        "preproc":[
            {
                "name":"auto_brightness_contrast_grayscale",
                "params":{
                    "clip_hist_percent":1
                }
            }
        ],
        "augment":[
            {
                "name":"brightness",
                "params":{
                    "value":1
                }
            }
        ],
        "name":"NumbersWaterMeters",
        "model":"yolov8x.pt",
        "size":224,
        "batch":16,
        "epoch":100,
        "device":0,
        "pretrained":true,
        "patience":50,
        "save":true,
        "save_period":-1,
        "cache":false,
        "workers":8,
        "exist_ok":false,
        "optimizer":"auto",
        "verbose":false,
        "seed":0,
        "deterministic":true,
        "single_cls":false,
        "rect":false,
        "cos_lr":false,
        "close_mosaic":10,
        "resume":false,
        "amp":true,
        "fraction":1.0,
        "profile":false,
        "lr0":0.01,
        "lrf":0.01,
        "momentum":0.937,
        "weight_decay":0.0005,
        "warmup_epochs":3.0,
        "warmup_momentum":0.8,
        "warmup_bias_lr":0.1,
        "box":7.5,
        "cls":0.5,
        "dfl":1.5,
        "pose":12.0,
        "kobj":2.0,
        "label_smoothing":0.0,
        "nbs":64,
        "overlap_mask":true,
        "mask_ratio":4,
        "dropout":0.0,
        "val":true
    },
    "test":{
        "dataset_path":"./assets/test_dataset/test_0"
    },
    "description":"Lorem ipsum"
    }

## Explanation
-  **Roboflow:**

The dataset is annotated and stored on roboflow. Do not use any preprocessing or augmentation beacuse it will done locally! You can find the required parameters in the export tab on robofolow

-  **Model**

	- Augment: These are the methods used for image augmentation. **The *"name"* is transalted to a method**, and paramters passed. In this list you can add any augmentation method that you can find in `/library/augmentators.py` file.

	- Prepoc: These are the methods used for image preprocessing. **The *"name"* is transalted to a method**, and paramters passed. In this list you can add any preprocessing method that you can find in `/library/preprocessing.py` file.

	- Etc: The other parameters are used for training configuration. You can find the documentation for these arguments here: [https://docs.ultralytics.com/modes/train/#arguments](https://docs.ultralytics.com/modes/train/#arguments)

	- Name: Run will be saved with this name. If the name already exists, an index number will be added

-  **Tests:**

	- Datset_path: Here you can specify a path to a folder containing test images. Test dataset are store in `/assets/test_datasets` folder

-  **Description:** Some description about the model, parameters or expectations.