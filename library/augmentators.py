import cv2
import matplotlib.pyplot as plt
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from typing import List

# TODO Create new labels for augmented images

def flip_v(images_list: List):

    processed = []

    vflip= iaa.Flipud(p=1.0)
    for item in images_list:
        image_path, image = item

        image_vf= vflip.augment_image(image)

        processed.append([
            image_path[:image_path.rindex("/")]+"/images", # Folder path for future save
            image_vf
        ])

    return processed


def flip_h(images_list: List):

    processed = []

    hflip= iaa.Fliplr(p=1.0)
    for item in images_list:
        image_path, image = item

        image_hf = hflip.augment_image(image)

        processed.append([
            image_path[:image_path.rindex("/")]+"/images", # Folder path for future save
            image_hf
        ])

    return processed