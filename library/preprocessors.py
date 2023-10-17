import os
import cv2
from typing import List

def find_and_read_jpg_images(path: str) -> List:
    '''
    Recursively finds and reads .jpg images in the specified directory and its subdirectories using OpenCV.
    
    :param path: The root directory to start searching for .jpg images.
    :return: A list of images as numpy arrays representing the found .jpg images and their paths.
    '''
    jpg_images = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_path = os.path.join(root, file)
                try:                    
                    image = cv2.imread(image_path)
                    if image is not None:
                        jpg_images.append([image_path, image])
                except Exception as e:
                    print(f"Error reading {image_path}: {str(e)}")
    
    return jpg_images


def grayscale(images_list: List) -> List:
    '''
    Converts a list of images to grayscale.
    
    :param images_list: A list of image data, where each item is a list with [image_path, image_data].
    '''
    processed = []

    for item in images_list:
        image_path, image = item
        is_gray = len(image.shape) == 2

        if not is_gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   

        processed.append([image_path, image])

        cv2.imwrite(image_path, image)

    return processed

def brightness(images_list: List, value: float) -> List:
    '''
    Adjust the brightness of images in the provided list.

    :param images_list: A list of image items where each item is a tuple (image_path, image).
    :param value: The brightness adjustment value.
    :return: A list of processed images in the format [(image_path, adjusted_image)].
    '''

    processed = []

    for item in images_list:
        image_path, image = item

        # Apply brightness adjustment
        adjusted_img = cv2.convertScaleAbs(image, alpha=value, beta=0)

        processed.append([image_path, adjusted_img])        

    return processed


def contrast(images_list: List, value: float) -> List:
    '''
    Adjust contrast of images in the provided list.

    :param images_list: A list of image items where each item is a tuple (image_path, image).
    :param value: The contrast adjustment value. 0 = lower, 1 = original, 1+ = higher
    :return: A list of processed images in the format [(image_path, enhanced_image)].
    '''

    processed = []

    for item in images_list:
        image_path, image = item

        # Check if the image is grayscale
        is_gray = len(image.shape) == 2

        if is_gray:
            # Apply contrast adjustment directly to grayscale image
            clahe = cv2.createCLAHE(clipLimit=value, tileGridSize=(8, 8))
            enhanced_img = clahe.apply(image)
        else:
            # Convert RGB image to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)

            # Apply CLAHE to L-channel
            clahe = cv2.createCLAHE(clipLimit=value, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)

            # Merge the CLAHE enhanced L-channel with the a and b channels
            limg = cv2.merge((cl, a, b))

            # Convert image from LAB Color model to BGR color space
            enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        processed.append([image_path, enhanced_img])        

    return processed


def auto_brightness_contrast_grayscale(images_list: List, clip_hist_percent: float) -> List:
    '''
    WARNING: Only for grayscale usage!
    Automatically adjust brightness and contrast for grayscale images.

    :param images_list: A list of image items where each item is a tuple (image_path, image).
    :param clip_hist_percent: The percentage of the histogram to clip on both sides.
    :return: A list of processed images in the format [(image_path, auto_result)].
    '''

    processed = []

    for item in images_list:
        image_path, image = item
        tmp = clip_hist_percent

        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate grayscale histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = [float(hist[0])]
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        tmp *= (maximum / 100.0)
        tmp /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < tmp:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - tmp):
            maximum_gray -= 1

        # Calculate alpha and beta values for contrast adjustment
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        auto_result = cv2.cvtColor(auto_result, cv2.COLOR_BGR2GRAY)
        auto_result = cv2.cvtColor(auto_result, cv2.COLOR_GRAY2RGB)

        processed.append([image_path, auto_result])

    return processed
