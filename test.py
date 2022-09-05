import cv2
import numpy as np

## This class obly works with image-1.jpg

def resize(image):
    width = 800
    height = 600
    dim = (width, height)

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


## Detect Region of Interest (ROI)
def region(image):
    height, width = image.shape
    triangle = np.array([
        [(100, height), (width // 2, height // 2), (width, height)]
    ])

    mask = np.zeros_like(image)

    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask


def make_points(image, average):
    slope, y_int = average
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])


def average(image, lines):
    """
  This function averages out the lines made in the cv2.HoughLinesP function.
    """

    left = []
    right = []
    if lines is not None:
        for line in lines:
            # print(line)
            x1, y1, x2, y2 = line.reshape(4)
            # fit line to points, return slope and y-int
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            # print(parameters)
            slope = parameters[0]
            y_int = parameters[1]
            # lines on the right have positive slope, and lines on the left have neg slope
            if slope < 0:
                left.append((slope, y_int))
            else:
                right.append((slope, y_int))

        # takes average among all the columns (column0: slope, column1: y_int)
        right_avg = np.average(right, axis=0)
        left_avg = np.average(left, axis=0)
        # create lines based on averages calculates
        left_line = make_points(image, left_avg)
        right_line = make_points(image, right_avg)
        return np.array([left_line, right_line])


def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image


def predict(image_path):
    image = cv2.imread(image_path)
    print('Image shape ', image.shape)
    image = resize(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(image, 100, 200)

    ## Convert to HSV
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lo_yellow = np.array([20, 100, 100], dtype="uint8")
    hi_yellow = np.array([30, 255, 255], dtype="uint8")

    mask_yellow = cv2.inRange(img_hsv, lo_yellow, hi_yellow)
    mask_white = cv2.inRange(gray, 200, 255)
    mask_yellow_white = cv2.bitwise_or(mask_white, mask_yellow)
    mask = cv2.bitwise_and(gray, mask_yellow_white)
    cropped_image = region(edges)

    ## Hough line transform
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=100, minLineLength=40, maxLineGap=5)

    copy = np.copy(image)
    averaged_lines = average(copy, lines)
    black_lines = display_lines(copy, averaged_lines)
    lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)

    cv2.imshow('Result', lanes)
