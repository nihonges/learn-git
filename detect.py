import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time
import cv2
import tifffile
import imagecodecs
import numpy as np

def count_holes(binary_img):
    img = binary_img.copy()
    # 0 as background, 255 as foreground, circle
    img_inverted = cv2.bitwise_not(img)
    contours1, hierarchy1 = cv2.findContours(img_inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy1 is None:
        return False

    hierarchy1 = hierarchy1[0]
    hole_count = 0
    if len(contours1) < 2:
        return 0

    for i in range(len(contours1)):
        if hierarchy1[i][2] != -1:  # 说明这个是某个轮廓的内部 -> 洞
            hole_count += 1

    return hole_count

def image_processing(folder_path, output_folder):
    start = time.time()
    start1 = time.time()
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            inputPath = os.path.join(folder_path, filename)
            #image = cv2.imread(inputPath)
            image = tifffile.imread(inputPath)

            end = time.time()
            print("Load image:" + str(end - start1))
            if image is None:
                print(f"cannot read image: {filename}")
                continue

            h, w = image.shape[:2]
            print(h, w)

            xs = 200
            xe = w - 200
            ys = 400
            ye = h - 200

            cropped_image = image[ys:ye, xs: xe]

            end = time.time()
            print("crop image:" + str(end - start1))

            img_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            end = time.time()
            print("convert image to grayscale:" + str(end - start1))

            ret, thresh_img = cv2.threshold(img_gray, 200, 255, 0)

            end = time.time()
            print("threhold: " + str(end - start1))

            print("findContours")
            contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            closed_contours = []
            for idx, contour in enumerate(contours):
                rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)

                M = cv2.moments(contour)

                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx = -1
                    cy = -1

                (ex, ey), radius = cv2.minEnclosingCircle(contour)
                center = (int(ex), int(ey))
                center_full = (int(ex+xs), int(ey+ys))

                hc, wc = thresh_img.shape

                if radius > 3.5 and radius < 12.5 and abs(cx - center[0]) < 3 and abs(cy - center[1]) < 3 and abs(
                        rect_w - rect_h) <= 3:
                    # second filter
                    xc1 = int(max(cx - 10, 0))
                    xc2 = int(min(cx + 30, wc))
                    yc1 = int(max(cy - 10, 0))
                    yc2 = int(min(cy + 30, hc))

                    if xc1 >= xc2 or yc1 >= yc2:
                        continue

                    candidate_crop = thresh_img[yc1:yc2, xc1:xc2]
                    holes = count_holes(candidate_crop)

                    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(candidate_crop, connectivity=4)
                    flag = 0
                    for i in range(0, num_labels):
                        if flag > 0: continue
                        areax = stats[i, cv2.CC_STAT_AREA]
                        xx, yy, ww, hh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], \
                            stats[
                                i, cv2.CC_STAT_HEIGHT]

                        # thresholds
                        if 8 <= areax <= 40 and holes > 0:  #
                            aspect_ratio = ww / hh
                            if 0.60 <= aspect_ratio <= 1.1:
                                if len(contour) >= 5:  # use 5 points to fit an ellipse
                                    ellipse = cv2.fitEllipse(contour)
                                    (centere, axese, anglee) = ellipse
                                    major, minor = axese
                                    ratioe = major / minor
                                    if ratioe > 0.70:
                                        print("add:" + str(center_full) + "|" + str(radius) + "|" + str(rect_x+xs) + "," + str(rect_y+ys) + "|" + str(
                                            rect_h) + "," + str(rect_w) )
                                        closed_contours.append(contour)
                                        cv2.rectangle(image, (rect_x + xs, rect_y + ys), (rect_x + xs + rect_w, rect_y + ys + rect_h), (0, 0, 255), 2)
                                        cv2.rectangle(image, (rect_x + xs - 150, rect_y + ys - 150),
                                                      (rect_x + xs + rect_w + 150, rect_y + ys + rect_h + 150), (0, 0, 255), 10)
                                        flag = 1
            end = time.time()
            print("finish processing:" + str(end - start1))
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_circles.png")
            #cv2.imwrite(output_path, image)
            tifffile.imwrite(output_path, image)
            end = time.time()
            print("save image:" + str(end - start1))


if __name__ == '__main__':

    folder_path = r'C:\Arbeit\python_opencv'
    output_folder = r'C:\Arbeit\python_opencv\result'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_processing(folder_path, output_folder)

    print('Finished')