import pytest
import cv2
import json
from types import SimpleNamespace

images_root_path = "/Users/bormilan/Documents/GitHub/cm-mimir/assets/test_datasets"


def test_cases(model_path, images_path):
    # load model
    labels_path = f"{images_root_path}/{images_path}/labels.json"
    with open(labels_path) as file:
        data = json.load(file, object_hook=lambda d: SimpleNamespace(**d))

    print(data)
    results = []
    for case in data:
        img = cv2.imread(f"{images_root_path}/{images_path}/{case.image_name}")
        # result = predict(case.image_name)
        result = ""

        if result == case.label:
            results.append(True)
        results.append(False)

    return results


print(test_cases("", "test0"))
