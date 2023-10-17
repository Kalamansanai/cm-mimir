import cv2
import matplotlib.pyplot as plt
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from typing import List

def example(images_list: List, example_param: float) -> List: 
    '''
        This is an example method for image augmentation.
        Each type of augmentation should have their own method

        :param example_param: Just an example, defined in a config json
        :return A list of only the augmented images in the format [(image_parent_folder_path, augmented_image)]
    '''
    processed = []
    for item in images_list:
        image_path, image = item

        # Here you can implement some augmentation for 'image'
        # TODO

        processed.append([
            image_path[:image_path.rindex("/")]+"/images", # Folder path for future save
            image
        ])

    return processed