import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time
import cv2
import numpy as np

def image_processing3(folder_path, output_folder):
    start = time.time()
    start1 = time.time()
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            inputPath = os.path.join(folder_path, filename)
            image = cv2.imread(inputPath)

            end = time.time()
            print("Load image:" + str(end - start1))
            if image is None:
                print(f"cannot read image: {filename}")
                continue

            h, w = image.shape[:2]

            # xs = 100
            # xe = w - 200
            # ys = 100
            # ye = h - 100
            # xs=10542
            # ys=2336
            # xe =xs+20
            # ye= ys+20
            # # xs=4586
            # # ys=19054
            # # xe =xs+20
            # # ye= ys+20
            cropped_image = image
            # cropped_image = image[ys:ye, xs: xe]
            h, w = cropped_image.shape[:2]
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_circles.png")
            # cv2.imwrite(output_path, binary)
            cv2.imwrite(output_path, cropped_image)
            end = time.time()
            print("crop image:" + str(end - start1))

            img_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            end = time.time()
            print("convert image to grayscale:" + str(end - start1))

            ret, thresh_img = cv2.threshold(img_gray, 170, 255, 0)

            end = time.time()
            print("threhold:" + str(end - start1))

            print("findContours")
            contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            end = time.time()
            print("find contours:" + str(end - start1))

            closed_contours = []
            open_contours = []
            for idx, contour in enumerate(contours):
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)

                M = cv2.moments(contour)

                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx = -1
                    cy = -1

                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                tmp_w = abs(box[0][0] - box[1][0])
                tmp_h = abs(box[2][1] - box[1][1])
                if radius > 3.5 and radius < 12.5 and abs(cx - center[0]) < 3 and abs(cy - center[1]) < 3 and abs(
                        rect_w - rect_h) < 3 and abs(tmp_w - tmp_h) < 3 and tmp_w > 0 and tmp_h > 0:
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    #
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    circle_area = np.pi * (radius ** 2)
                    contour_area = cv2.contourArea(contour)

                    ratio = contour_area / circle_area
                    print("Area ratio:", ratio)
                    #
                    if len(contour) >= 5:  # 拟合椭圆至少需要5个点
                        ellipse = cv2.fitEllipse(contour)
                        (center, axes, angle) = ellipse
                        major, minor = axes
                        ratio = major / minor

                    print("add:" + str(center) + "|" + str(radius) + "|" + str(rect_x) + "," + str(rect_y) + "|" + str(
                        rect_h) + "," + str(rect_w) + "|" + str(tmp_h) + "," + str(tmp_w))
                    closed_contours.append(contour)
                    cv2.rectangle(cropped_image, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 255), 2)
                    cv2.rectangle(cropped_image, (rect_x - 50, rect_y - 50),
                                  (rect_x - 50 + rect_w + 100, rect_y - 50 + rect_h + 100), (0, 0, 255), 10)

            end = time.time()
            print("filter contours and mark roi:" + str(end - start1))
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_circles.png")
            # cv2.imwrite(output_path, binary)
            cv2.imwrite(output_path, cropped_image)

def image_processing(folder_path, output_folder):
    start = time.time()
    start1 = time.time()
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            inputPath = os.path.join(folder_path, filename)
            img = cv2.imread(inputPath, cv2.IMREAD_GRAYSCALE)

            end = time.time()
            print("Load image:" + str(end - start1))
            if img is None:
                print(f"cannot read image: {filename}")
                continue

            # test zone time
            h, w = img.shape[:2]

            xs = 200
            xe = w - 200
            ys = 200
            ye = h - 200

            cropped_image = img[ys:ye, xs:xe]
            h, w = cropped_image.shape[:2]
            end = time.time()
            print("crop image:" + str(end - start1))

            # binary image
            _, binary = cv2.threshold(cropped_image, 200, 255, cv2.THRESH_BINARY)

            end = time.time()
            print("threhold:" + str(end - start1))

            contours, hierarchy = cv2.findContours(binary, 1, 2)

            closed_contours = []
            print("find contours:" + str(end - start1))
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 25 or area > 120:
                    continue  # 跳过太小的轮廓

                rect = cv2.minAreaRect(contour)  # 最小外接矩形（旋转）
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)  # 得到包围盒

                M = cv2.moments(contour)  # 计算轮廓的矩
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])  # 轮廓中心点
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx, cy = -1, -1

                (x, y), radius = cv2.minEnclosingCircle(contour)  # 最小外接圆
                center = (int(x), int(y))

                # 比较宽高等几何特征，筛选近似为圆的小封闭轮廓
                tmp_w = abs(box[0][0] - box[1][0])
                tmp_h = abs(box[2][1] - box[1][1])

                if (
                        radius > 4.0 and radius < 12.5 and
                        abs(cx - center[0]) < 3 and
                        abs(cy - center[1]) < 3 and
                        abs(rect_w - rect_h) < 3 and
                        abs(tmp_w - tmp_h) < 3 and
                        tmp_w > 0 and tmp_h > 0
                ):
                    center_in_full = (center[0] + 100, center[1] + 100)
                    print("add:" + str(center_in_full) + "|" + str(radius) + "|" + str(rect_x) + "," + str(rect_y) + "|" + str(
                        rect_h) + "," + str(rect_w) + "|" + str(tmp_h) + "," + str(tmp_w))
                    closed_contours.append(contour)

                    # 在图上画红框标记
                    cv2.rectangle(img, (rect_x + xs, rect_y + ys), (rect_x + rect_w + xs, rect_y + rect_h + ys), (0, 0, 255), 2)
                    # 更大一圈的红框
                    cv2.rectangle(img, (rect_x - 50+ xs, rect_y - 50 + ys),
                                  (rect_x - 50 + rect_w + 100 + xs, rect_y - 50 + rect_h + 100 + ys), (0, 0, 255), 10)

            end = time.time()
            print("filter contours and mark roi:" + str(end - start1))
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_circles.png")
            # cv2.imwrite(output_path, binary)
            cv2.imwrite(output_path, img)

def image_processing_zone(folder_path, output_folder):
    start = time.time()
    start1 = time.time()
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            inputPath = os.path.join(folder_path, filename)
            img = cv2.imread(inputPath, cv2.IMREAD_GRAYSCALE)

            end = time.time()
            print("Load image:" + str(end - start1))
            if img is None:
                print(f"cannot read image: {filename}")
                continue

            # test zone time
            h, w = img.shape[:2]

            xs = 100
            xe = w - 200
            ys = 100
            ye = h - 100

            cropped_image = img[ys:ye, xs:xe]
            h, w = cropped_image.shape[:2]
            end = time.time()
            print("crop image:" + str(end - start1))

            # binary image
            _, binary = cv2.threshold(cropped_image, 200, 255, cv2.THRESH_BINARY)

            end = time.time()
            print("threhold:" + str(end - start1))

            # 1. connected zones
            num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)
            end = time.time()
            print("connect:" + str(end - start1))

            # 2. candidates
            binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            end = time.time()
            print("backto color:" + str(end - start1))
            for i in range(0, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], \
                stats[
                    i, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[i]

                # thresholds
                if 8 <= area <= 40:  #
                    aspect_ratio = w / h
                    if 0.60 <= aspect_ratio <= 1.4:
                        print(
                            f"Defects detected, center: ({int(cx)}, {int(cy)}), area: {area}")
                        # draw circles and rectangle
                        end = time.time()
                        print("suchen:" + str(end - start1))
                        cv2.rectangle(binary_color, (x - 300, y - 300), (x + w + 300, y + h + 300), (0, 0, 255), 10)
                        cv2.circle(binary_color, (int(cx), int(cy)), 2 + 50, (0, 0, 255), 10)
                        end = time.time()
                        print("analyze:" + str(end - start1))

            end = time.time()
            print("connectedzone:" + str(end - start1))
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_circles.png")

            # cv2.imwrite(output_path, binary)
            cv2.imwrite(output_path, binary_color)
            end = time.time()
            print("save:" + str(end - start1))

if __name__ == '__main__':

    folder_path = r'D:\Lesson\opencv-learn\python'
    output_folder = r'D:\Lesson\opencv-learn\python\result'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # image_processing_zone(folder_path, output_folder)
    image_processing3(folder_path, output_folder)


    # folder_path = Path(folder_path)
    # image_files = list(folder_path.glob("*.tif"))
    # args = [(str(image_path), output_folder) for image_path in image_files]
    #
    # with ProcessPoolExecutor() as executor:
    #     results = executor.map(image_processing, args)

    # image_processing(folder_path, output_folder)

    print('Finished')

# template = np.array([
#     [255, 255, 0, 0, 0, 0, 0, 0, 255, 255],
#     [255, 255, 0, 255, 255, 255, 255, 0, 0, 255],
#     [255, 0, 255, 255, 255, 255, 255, 255, 0, 255],
#     [255, 0, 255, 255, 255, 255, 255, 255, 0, 255],
#     [255, 0, 255, 255, 255, 255, 255, 255, 0, 255],
#     [255, 0, 255, 255, 255, 255, 255, 255, 0, 255],
#     [255, 0, 255, 255, 255, 255, 255, 255, 0, 255],
#     [255, 255, 0, 255, 255, 255, 255, 0, 0, 255],
#     [255, 255, 0, 0, 0, 0, 0, 0, 255, 255],
# ], dtype=np.uint8)
#
# binary_01 = binary // 255
# template_01 = template // 255
#
# # ）
# res = cv2.matchTemplate(binary_01.astype(np.float32), template_01.astype(np.float32), cv2.TM_CCOEFF_NORMED)
#
# #
# threshold = 0.5  #
# loc = np.where(res >= threshold)
#
# #
# for pt in zip(*loc[::-1]):  # x, y
#     print(f"coordinate: (x={pt[0]}, y={pt[1]})，匹配度={res[pt[1], pt[0]]:.4f}")


# #
# circles = cv2.HoughCircles(
#     binary,
#     cv2.HOUGH_GRADIENT,
#     dp=1.2,
#     minDist=20,
#     param1=200,
#     param2=50,
#     minRadius=3,
#     maxRadius=6
# )
#
# # draw circles and save results
# output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#
# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         # draw circle
#         cv2.circle(output_img, (i[0], i[1]), i[2] + 250, (0, 255, 0), -1)
#         # draw center point
#         cv2.circle(output_img, (i[0], i[1]), 2, (0, 0, 255), 8)
#     print(f" {filename} detected {len(circles[0])} circles")
# else:
#     print(f"{filename} circle not detected")
#
# if SHOW_IMAGES:
#     show_image(output_img, title="Detected Circles")

#
# #
# cropped_img = binary[y:y + h, x:x + w]
#
# cv2.imshow('Cropped', cropped_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# if SHOW_IMAGES:
#     show_image(binary, title="Binary Image")
