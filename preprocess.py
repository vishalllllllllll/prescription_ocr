import cv2
import numpy as np

def preprocess_image(image_path):
    """
    This function preprocesses the input image (jpg) to enhance text for OCR.
    It includes grayscale conversion, noise removal, binarization, and dilation.
    """

    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply median blur to reduce noise
    blurred = cv2.medianBlur(gray, 5)

    # Binarize the image using adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)

    # Apply dilation to enhance character thickness
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Save the preprocessed image with a .jpg extension
    preprocessed_image_path = image_path.replace(".jpg", "_preprocessed.jpg")
    cv2.imwrite(preprocessed_image_path, dilated)

    return preprocessed_image_path
