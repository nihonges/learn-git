# # #############################
# import os
# import tifffile
# import numpy as np
# import cv2
# import time
# from concurrent.futures import ProcessPoolExecutor
#
# # -----------------------------
# # 核心 tile 处理逻辑
# # -----------------------------
# def count_holes(binary_img):
#     img = binary_img.copy()
#     # 0 as background, 255 as foreground, circle
#     img_inverted = cv2.bitwise_not(img)
#     contours1, hierarchy1 = cv2.findContours(img_inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     if hierarchy1 is None:
#         return False
#
#     hierarchy1 = hierarchy1[0]
#     hole_count = 0
#     if len(contours1) < 2:
#         return 0
#
#     for i in range(len(contours1)):
#         if hierarchy1[i][2] != -1:  # 说明这个是某个轮廓的内部 -> 洞
#             hole_count += 1
#
#     return hole_count
#
# def process_tile(tile):
#     start = time.time()
#     start1 = time.time()
#     image = tile.copy()
#
#     end = time.time()
#     print("Load image:" + str(end - start1))
#
#     cropped_image = image
#     xs = 0
#     ys = 0
#     end = time.time()
#     print("crop image:" + str(end - start1))
#     if cropped_image.shape[2] == 3:
#         img_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#     else:
#         img_gray = cropped_image
#
#     end = time.time()
#     print("convert image to grayscale:" + str(end - start1))
#
#     ret, thresh_img = cv2.threshold(img_gray, 200, 255, 0)
#
#     end = time.time()
#     print("threhold: " + str(end - start1))
#
#     print("findContours")
#     contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     closed_contours = []
#     for idx, contour in enumerate(contours):
#         rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
#
#         M = cv2.moments(contour)
#
#         if M['m00'] != 0:
#             cx = int(M['m10'] / M['m00'])
#             cy = int(M['m01'] / M['m00'])
#         else:
#             cx = -1
#             cy = -1
#
#         (ex, ey), radius = cv2.minEnclosingCircle(contour)
#         center = (int(ex), int(ey))
#
#
#         hc, wc = thresh_img.shape
#
#         if radius > 3.5 and radius < 12.5 and abs(cx - center[0]) < 3 and abs(cy - center[1]) < 3 and abs(
#                 rect_w - rect_h) <= 3:
#             # second filter
#             xc1 = int(max(cx - 10, 0))
#             xc2 = int(min(cx + 30, wc))
#             yc1 = int(max(cy - 10, 0))
#             yc2 = int(min(cy + 30, hc))
#
#             if xc1 >= xc2 or yc1 >= yc2:
#                 continue
#
#             candidate_crop = thresh_img[yc1:yc2, xc1:xc2]
#             holes = count_holes(candidate_crop)
#
#             num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(candidate_crop, connectivity=4)
#             flag = 0
#             for i in range(0, num_labels):
#                 if flag > 0: continue
#                 areax = stats[i, cv2.CC_STAT_AREA]
#                 xx, yy, ww, hh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], \
#                     stats[
#                         i, cv2.CC_STAT_HEIGHT]
#
#                 # thresholds
#                 if 8 <= areax <= 40 and holes > 0:  #
#                     aspect_ratio = ww / hh
#                     if 0.60 <= aspect_ratio <= 1.1:
#                         if len(contour) >= 5:  # use 5 points to fit an ellipse
#                             ellipse = cv2.fitEllipse(contour)
#                             (centere, axese, anglee) = ellipse
#                             major, minor = axese
#                             ratioe = major / minor
#                             if ratioe > 0.70:
#                                 print("add:" + str(center) + "|" + str(radius) + "|" + str(rect_x+xs) + "," + str(rect_y+ys) + "|" + str(
#                                     rect_h) + "," + str(rect_w) )
#                                 closed_contours.append(contour)
#                                 cv2.rectangle(image, (rect_x + xs, rect_y + ys), (rect_x + xs + rect_w, rect_y + ys + rect_h), (0, 0, 255), 2)
#                                 cv2.rectangle(image, (rect_x + xs - 150, rect_y + ys - 150),
#                                               (rect_x + xs + rect_w + 150, rect_y + ys + rect_h + 150), (0, 0, 255), 10)
#                                 flag = 1
#     end = time.time()
#     print("finish processing:" + str(end - start1))
#
#     print("save image:" + str(end - start1))
#     return image
#
# # def process_tile(tile):
# #     cropped_image = tile.copy()
# #     img_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
# #     _, thresh_img = cv2.threshold(img_gray, 170, 255, 0)
# #     contours, _ = cv2.findContours(thresh_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# #
# #     for contour in contours:
# #         rect = cv2.minAreaRect(contour)
# #         box = cv2.boxPoints(rect)
# #         box = np.intp(box)
# #         rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
# #
# #         M = cv2.moments(contour)
# #         if M['m00'] != 0:
# #             cx = int(M['m10'] / M['m00'])
# #             cy = int(M['m01'] / M['m00'])
# #         else:
# #             cx = -1
# #             cy = -1
# #
# #         (x, y), radius = cv2.minEnclosingCircle(contour)
# #         center = (int(x), int(y))
# #         tmp_w = abs(box[0][0] - box[1][0])
# #         tmp_h = abs(box[2][1] - box[1][1])
# #
# #         # 筛选圆形
# #         if radius > 3.5 and radius < 12.5 and abs(cx - center[0]) < 3 and abs(cy - center[1]) < 3 \
# #            and abs(rect_w - rect_h) < 3 and abs(tmp_w - tmp_h) < 3 and tmp_w > 0 and tmp_h > 0:
# #
# #             # 拟合椭圆或画矩形
# #             if len(contour) >= 5:
# #                 ellipse = cv2.fitEllipse(contour)
# #                 cv2.ellipse(cropped_image, ellipse, (0, 0, 255), 2)
# #             else:
# #                 cv2.rectangle(cropped_image, (rect_x, rect_y),
# #                               (rect_x + rect_w, rect_y + rect_h), (0, 0, 255), 2)
# #
# #     return cropped_image
#
# # -----------------------------
# # 分块读取 TIFF + 并行处理 + 拼回
# # -----------------------------
# def process_large_tiff(tiff_path, output_path, tile_size=1024, overlap=32, workers=None):
#     import os
#     from tqdm import tqdm  # 可选：显示进度条
#
#     # 直接读取整张图像，避免每 tile 重复打开
#     image = tifffile.imread(tiff_path)
#     if image.ndim == 2:
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#
#     H, W = image.shape[:2]
#     result = np.zeros((H, W, 3), dtype=np.uint8)
#
#     tile_coords = []
#     tiles = []
#
#     # 切 tile（含 overlap）+ 保存坐标
#     for y in range(0, H, tile_size):
#         for x in range(0, W, tile_size):
#             x_start = max(x - overlap, 0)
#             y_start = max(y - overlap, 0)
#             x_end = min(x + tile_size + overlap, W)
#             y_end = min(y + tile_size + overlap, H)
#
#             tile = image[y_start:y_end, x_start:x_end].copy()
#             tiles.append(tile)
#             tile_coords.append((x_start, y_start, x, y, min(x + tile_size, W), min(y + tile_size, H)))
#
#     print(f"🧩 Total tiles to process: {len(tiles)}")
#
#     # 单线程处理每个 tile
#     for idx, (tile, coord) in enumerate(tqdm(zip(tiles, tile_coords), total=len(tiles), desc="Processing tiles")):
#         tile_result = process_tile(tile)
#
#         x_start, y_start, x_tile0, y_tile0, x_tile1, y_tile1 = coord
#         dx0 = x_tile0 - x_start
#         dy0 = y_tile0 - y_start
#         dx1 = dx0 + (x_tile1 - x_tile0)
#         dy1 = dy0 + (y_tile1 - y_tile0)
#
#         cropped_valid = tile_result[dy0:dy1, dx0:dx1]
#         result[y_tile0:y_tile1, x_tile0:x_tile1] = cropped_valid
#
#     cv2.imwrite(output_path, result)
#     print(f"✅ Done: {output_path}")
#
# # -----------------------------
# # 处理整个文件夹
# # -----------------------------
# def process_folder(input_folder, output_folder, tile_size=1024, overlap=32, workers=None):
#     os.makedirs(output_folder, exist_ok=True)
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(".tif"):
#             input_path = os.path.join(input_folder, filename)
#             output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_result.png")
#             print(f"Processing: {filename}")
#             process_large_tiff(input_path, output_path, tile_size, overlap, workers)
#
# # -----------------------------
# # 示例运行
# # -----------------------------
# if __name__ == "__main__":
#     input_folder = r"C:\Arbeit\python_opencv"
#     output_folder = r"C:\Arbeit\python_opencv\result"
#     # x=os.cpu_count()
#     process_folder(input_folder, output_folder, tile_size=1024, overlap=32, workers=8)


# #############################
import os
import tifffile
import numpy as np
import cv2
import time
from concurrent.futures import ProcessPoolExecutor

# -----------------------------
# 核心 tile 处理逻辑
# -----------------------------
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

def process_tile(tile):
    start = time.time()
    start1 = time.time()
    image = tile.copy()

    end = time.time()
    #print("Load image:" + str(end - start1))

    cropped_image = image
    xs = 0
    ys = 0
    end = time.time()
    #print("crop image:" + str(end - start1))
    if cropped_image.shape[2] == 3:
        img_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = cropped_image

    end = time.time()
    #print("convert image to grayscale:" + str(end - start1))

    ret, thresh_img = cv2.threshold(img_gray, 200, 255, 0)

    end = time.time()
    #print("threhold: " + str(end - start1))

    #print("findContours")
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
                                print("add:" + str(center) + "|" + str(radius) + "|" + str(rect_x+xs) + "," + str(rect_y+ys) + "|" + str(
                                    rect_h) + "," + str(rect_w) )
                                closed_contours.append(contour)
                                cv2.rectangle(image, (rect_x + xs, rect_y + ys), (rect_x + xs + rect_w, rect_y + ys + rect_h), (0, 0, 255), 2)
                                cv2.rectangle(image, (rect_x + xs - 150, rect_y + ys - 150),
                                              (rect_x + xs + rect_w + 150, rect_y + ys + rect_h + 150), (0, 0, 255), 10)
                                flag = 1
    end = time.time()
    #print("finish processing:" + str(end - start1))

    #print("save image:" + str(end - start1))
    return image

# def process_tile(tile):
#     cropped_image = tile.copy()
#     img_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#     _, thresh_img = cv2.threshold(img_gray, 170, 255, 0)
#     contours, _ = cv2.findContours(thresh_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#
#     for contour in contours:
#         rect = cv2.minAreaRect(contour)
#         box = cv2.boxPoints(rect)
#         box = np.intp(box)
#         rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
#
#         M = cv2.moments(contour)
#         if M['m00'] != 0:
#             cx = int(M['m10'] / M['m00'])
#             cy = int(M['m01'] / M['m00'])
#         else:
#             cx = -1
#             cy = -1
#
#         (x, y), radius = cv2.minEnclosingCircle(contour)
#         center = (int(x), int(y))
#         tmp_w = abs(box[0][0] - box[1][0])
#         tmp_h = abs(box[2][1] - box[1][1])
#
#         # 筛选圆形
#         if radius > 3.5 and radius < 12.5 and abs(cx - center[0]) < 3 and abs(cy - center[1]) < 3 \
#            and abs(rect_w - rect_h) < 3 and abs(tmp_w - tmp_h) < 3 and tmp_w > 0 and tmp_h > 0:
#
#             # 拟合椭圆或画矩形
#             if len(contour) >= 5:
#                 ellipse = cv2.fitEllipse(contour)
#                 cv2.ellipse(cropped_image, ellipse, (0, 0, 255), 2)
#             else:
#                 cv2.rectangle(cropped_image, (rect_x, rect_y),
#                               (rect_x + rect_w, rect_y + rect_h), (0, 0, 255), 2)
#
#     return cropped_image

# -----------------------------
# 分块读取 TIFF + 并行处理 + 拼回
# -----------------------------
def process_large_tiff(tiff_path, output_path, tile_size=1024, overlap=32, workers=None):
    import os
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor
    tif = time.time()
    image = tifffile.imread(tiff_path)
    readimage = time.time()
    print("read tiff:" + str(readimage-tif))
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    H, W = image.shape[:2]
    result = np.zeros((H, W, 3), dtype=np.uint8)

    tile_coords = []
    tiles = []

    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            x_start = max(x - overlap, 0)
            y_start = max(y - overlap, 0)
            x_end = min(x + tile_size + overlap, W)
            y_end = min(y + tile_size + overlap, H)

            tile = image[y_start:y_end, x_start:x_end].copy()
            tiles.append(tile)
            tile_coords.append((x_start, y_start, x, y, min(x + tile_size, W), min(y + tile_size, H)))

    print(f"🧩 Total tiles to process: {len(tiles)}")
    dealstart = time.time()
    print("dealwith tiff:" + str(dealstart-tif))

    with ProcessPoolExecutor(max_workers=workers) as executor:
    #with ProcessPoolExecutor(max_workers=workers or os.cpu_count() - 1) as executor:
        futures = [executor.submit(process_tile, tile) for tile in tiles]

        for fut, coord in tqdm(zip(futures, tile_coords), total=len(tiles), desc="Processing tiles"):
            tile_result = fut.result()

            x_start, y_start, x_tile0, y_tile0, x_tile1, y_tile1 = coord
            dx0 = x_tile0 - x_start
            dy0 = y_tile0 - y_start
            dx1 = dx0 + (x_tile1 - x_tile0)
            dy1 = dy0 + (y_tile1 - y_tile0)
            eachtile = time.time()
            print("start tiff:" + str(eachtile - tif))
            cropped_valid = tile_result[dy0:dy1, dx0:dx1]
            result[y_tile0:y_tile1, x_tile0:x_tile1] = cropped_valid
            eachtile = time.time()
            print("end tiff:" + str(eachtile - tif))

    dealwith = time.time()
    print("dealwith tiff:" + str(dealwith-tif))
    #cv2.imwrite(output_path, result)
    tifffile.imwrite(output_path, image)
    write = time.time()
    print("write tiff:" + str(write-tif))
    print(f"✅ Done: {output_path}")

# -----------------------------
# 处理整个文件夹
# -----------------------------
def process_folder(input_folder, output_folder, tile_size=1024, overlap=32, workers=None):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".tif"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_result.png")
            print(f"Processing: {filename}")
            process_large_tiff(input_path, output_path, tile_size, overlap, workers)

# -----------------------------
# 示例运行
# -----------------------------
if __name__ == "__main__":
    input_folder = r"C:\Arbeit\python_opencv"
    output_folder = r"C:\Arbeit\python_opencv\result"
    # x=os.cpu_count()
    start = time.time()

    process_folder(input_folder, output_folder, tile_size=4096, overlap=32, workers=8)
    end = time.time()
    print("Processing image:" + str(end - start))
