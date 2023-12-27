import os
import tqdm
import time

import cv2
import numpy as np

threshold1 = 0
threshold2 = 110
dial_iter = 5
img_threshold_iter = 5

def new_value_threshold1(x):
    global threshold1
    threshold1 = x


def new_value_threshold2(x):
    global threshold2
    threshold2 = x


def new_value_dial_iter(x):
    global dial_iter
    dial_iter = x


def new_value_img_threshold_iter(x):
    global img_threshold_iter
    img_threshold_iter = x


cv2.namedWindow('Trackbar')
cv2.resizeWindow('Trackbar', 360, 240)
cv2.createTrackbar("Threshold1", "Trackbar", 0, 255, new_value_threshold1)
cv2.createTrackbar("Threshold2", "Trackbar", 110, 255, new_value_threshold2)
cv2.createTrackbar("Dial Iter", "Trackbar", 5, 15, new_value_dial_iter)
cv2.createTrackbar("Threshold Iter", "Trackbar", 5, 15, new_value_img_threshold_iter)

folder = os.path.join(os.getcwd(), "dataset")
filenames = os.listdir(folder)

# image = cv2.imread(f"{folder}/{filenames[0]}")
# image = cv2.resize(image, (980, 640))
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# img_blur = cv2.GaussianBlur(gray, (3, 3), 1)

img_start = 0
show_num_polygon = True
draw_fill_polygon = True
calibration = False

while True:

    key = cv2.waitKey(1) & 0xFF
    better_threshold = 0

    image = cv2.imread(f"{folder}/{filenames[img_start]}")
    image = cv2.resize(image, (980, 640))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, (3, 3), 1)

    kernel = np.ones((3, 3), np.uint8)

    # Пересобрать
    if calibration:
        calibration = False

        param = []
        contours_approx = []
        start_time = time.time()

        for dial_i in tqdm.tqdm(range(0, 11)):
            for thresh_img_i in range(0, 11):
                param.append((dial_i, thresh_img_i))
                cv2.setTrackbarPos('Dial Iter', "Trackbar", dial_i)
                cv2.setTrackbarPos('Threshold Iter', "Trackbar", dial_i + thresh_img_i)

                canny = cv2.Canny(img_blur, threshold1, threshold2)
                dial = cv2.dilate(canny, kernel, iterations=dial_i)
                img_threshold = cv2.erode(dial, kernel, iterations=thresh_img_i)
                contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                bd = 0

                for cnt in contours:
                    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

                    x_points = []
                    y_points = []

                    if (len(approx) >= 4) and (len(approx) <= 8):
                        x_points = [x[0][0] for x in approx]
                        y_points = [y[0][1] for y in approx]
                        x_min, x_max, y_min, y_max = min(x_points), max(x_points), min(y_points), max(y_points)
                        square = (x_max - x_min) * (y_max - y_min)

                        if (x_max - x_min >= 100 or y_max - y_min >= 100 or x_max - x_min <= 20 or y_max - y_min <= 20
                                or square >= 1000 or square <= 500):
                            continue

                        bd += 1

                contours_approx.append(bd)

        print("Bette param: ", param[contours_approx.index(max(contours_approx))], "max approx = ", max(contours_approx))
        print("Time calibration: ", time.time() - start_time)

        dial_iter, img_threshold_iter = param[contours_approx.index(max(contours_approx))]
        cv2.setTrackbarPos('Dial Iter', "Trackbar", dial_iter)
        cv2.setTrackbarPos('Threshold Iter', "Trackbar", img_threshold_iter)


    canny = cv2.Canny(img_blur, threshold1, threshold2)
    dial = cv2.dilate(canny, kernel, iterations=dial_iter)
    img_threshold = cv2.erode(dial, kernel, iterations=img_threshold_iter)

    img_contours = image.copy()
    contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

        x_points = []
        y_points = []

        if (len(approx) >= 4) and (len(approx) <= 8):
            x_points = [x[0][0] for x in approx]
            y_points = [y[0][1] for y in approx]
            x_min, x_max, y_min, y_max = min(x_points), max(x_points), min(y_points), max(y_points)
            square = (x_max - x_min) * (y_max - y_min)

            if (x_max - x_min >= 100 or y_max - y_min >= 100 or x_max - x_min <= 20 or y_max - y_min <= 20
                    or square >= 1000 or square <= 500):
                continue

            if square >= 800 or square <= 600:
                if draw_fill_polygon:
                    cv2.fillPoly(img_contours, [approx], (0, 0, 255))
                else:
                    cv2.polylines(img_contours, [approx], True, (0, 0, 255), 2)
            else:
                if draw_fill_polygon:
                    cv2.fillPoly(img_contours, [approx], (0, 255, 0))
                else:
                    cv2.polylines(img_contours, [approx], True, (0, 255, 0), 2)

            text_put = len(approx) if show_num_polygon else square
            cv2.putText(img_contours, f"{text_put}",
                        (x_min + 5, y_min + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (255, 0, 0), 1, 1)

            better_threshold += 1

    cv2.putText(img_contours, str(better_threshold), (10, 40),  cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 255), 1, 1)

    cv2.imshow("canny", dial)
    cv2.imshow("image", image)
    cv2.imshow("contours", img_contours)

    if key == ord('n'):

        if img_start == len(filenames):
            cv2.destroyAllWindows()
            break
        else:
            img_start += 1

    elif key == ord("c"):
        calibration = True

    elif key == ord('f'):
        draw_fill_polygon = not draw_fill_polygon
    elif key == ord('s'):
        show_num_polygon = not show_num_polygon

    elif key == ord('q'):
        cv2.destroyAllWindows()
        break
