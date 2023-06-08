import cv2 as cv
import numpy as np

def preprocess(image):
    image = cv.resize(image, (1024, 1024))
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    X, Y = np.meshgrid(np.arange(-512, 512), np.arange(-512, 512))
    image[np.sqrt(X**2 + Y**2) > 979/2] = 255
    return image


def sharpen(image):
    image = cv.medianBlur(image, 7)
    kernel = np.array([
        [1, 0, -1],
        [3, 0, -3],
        [1, 0, -1]
    ])

    image = cv.GaussianBlur(image, (7, 7), 0)

    image1 = cv.filter2D(image, -1, kernel)
    image2 = cv.filter2D(image, -1, -kernel)
    image3 = cv.filter2D(image, -1, kernel.T)
    image4 = cv.filter2D(image, -1, -(kernel.T))

    image = image1 + image2 + image3 + image4
    
    image[image > 50] = 0
    return image



def detect(image):
    image = sharpen(image)
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1, 20, None, 100, 6, 6, 8)
    detected_points = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            detected_points.append((i[0], i[1]))
    return np.array(detected_points, dtype=np.float32)