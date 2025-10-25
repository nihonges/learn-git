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

            thresholdend = time.time()
            print("threshold: " + str(thresholdend - start1))

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
            print("finish processing:" + str(end - thresholdend))
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_circles.png")
            #cv2.imwrite(output_path, image)
            # tifffile.imwrite(output_path, image)
            tifffile.imwrite(output_path, image, compression='jpeg')
            end = time.time()
            print("save image:" + str(end - start1))


if __name__ == '__main__':

    folder_path = r'D:\Lesson\opencv-learn\arbeit'
    output_folder = r'D:\Lesson\opencv-learn\arbeit\result'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_processing(folder_path, output_folder)

    print('Finished')
#############
# import os
# import tifffile
# import numpy as np
# import cv2
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from tqdm import tqdm
# import time
#
# # -----------------------------
# # 核心 tile 处理逻辑
# # -----------------------------
# def count_holes(binary_img):
#     img_inverted = cv2.bitwise_not(binary_img)
#     contours, hierarchy = cv2.findContours(img_inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if hierarchy is None:
#         return 0
#     hierarchy = hierarchy[0]
#     return sum(1 for h in hierarchy if h[2] != -1)
#
# def process_tile(tile, debug=False):
#     pid = os.getpid()
#     start_time = time.time()  # 记录开始
#     if debug:
#         print(f"Start processing tile in process {pid}")
#
#     image = tile.copy()
#     if image.ndim == 2:
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#
#     img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, thresh_img = cv2.threshold(img_gray, 200, 255, 0)
#     contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     for contour in contours:
#         rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
#         M = cv2.moments(contour)
#         if M['m00'] != 0:
#             cx = int(M['m10'] / M['m00'])
#             cy = int(M['m01'] / M['m00'])
#         else:
#             cx = cy = -1
#
#         (ex, ey), radius = cv2.minEnclosingCircle(contour)
#         center = (int(ex), int(ey))
#         hc, wc = thresh_img.shape
#
#         if radius > 3.5 and radius < 12.5 and abs(cx - center[0]) < 3 and abs(cy - center[1]) < 3 and abs(rect_w - rect_h) <= 3:
#             xc1 = max(cx - 10, 0)
#             xc2 = min(cx + 30, wc)
#             yc1 = max(cy - 10, 0)
#             yc2 = min(cy + 30, hc)
#             if xc1 >= xc2 or yc1 >= yc2:
#                 continue
#             candidate_crop = thresh_img[yc1:yc2, xc1:xc2]
#             holes = count_holes(candidate_crop)
#             num_labels, _, stats, _ = cv2.connectedComponentsWithStats(candidate_crop, connectivity=4)
#             for i in range(num_labels):
#                 areax = stats[i, cv2.CC_STAT_AREA]
#                 ww, hh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
#                 if 8 <= areax <= 40 and holes > 0:
#                     aspect_ratio = ww / hh
#                     if 0.6 <= aspect_ratio <= 1.1 and len(contour) >= 5:
#                         ellipse = cv2.fitEllipse(contour)
#                         major, minor = ellipse[1]
#                         if major / minor > 0.7:
#                             if debug:
#                                 print(f"Found circle at {center}")
#                             cv2.rectangle(image, (rect_x, rect_y), (rect_x+rect_w, rect_y+rect_h), (0,0,255), 2)
#
#     end_time = time.time()
#     process_time = end_time - start_time  # 计算耗时
#     if debug:
#         print(f"Done processing tile in process {pid}, time: {process_time:.3f}s")
#     return image, process_time  # 返回 tile 和耗时
#
# # -----------------------------
# # 并行处理大 TIFF
# # -----------------------------
# def process_large_tiff(tiff_path, output_path, tile_size=2048, overlap=32, workers=None, debug=False):
#     tif = time.time()
#     image = tifffile.imread(tiff_path)
#     readimage = time.time()
#     print("read tiff:" + str(readimage-tif))
#     H, W = image.shape[:2]
#     result = np.zeros((H, W, 3), dtype=np.uint8)
#
#     tiles = []
#     tile_coords = []
#     for y in range(0, H, tile_size):
#         for x in range(0, W, tile_size):
#             x_start = max(x - overlap, 0)
#             y_start = max(y - overlap, 0)
#             x_end = min(x + tile_size + overlap, W)
#             y_end = min(y + tile_size + overlap, H)
#             tiles.append(image[y_start:y_end, x_start:x_end].copy())
#             tile_coords.append((x_start, y_start, x_end, y_end))
#
#     dealstart = time.time()
#     print("dealwith tiff:" + str(dealstart-tif))
#     tile_times = []
#     with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as executor:
#         futures = {executor.submit(process_tile, tile, debug): coord for tile, coord in zip(tiles, tile_coords)}
#         for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing tiles"):
#             tile_result, t = fut.result()
#             coord = futures[fut]
#             tile_times.append((coord, t))  # 记录每个 tile 的耗时
#
#             x0, y0, x1, y1 = coord
#             dx0 = 0 if x0 == 0 else overlap
#             dy0 = 0 if y0 == 0 else overlap
#             dx1 = dx0 + min(tile_size, x1 - x0)
#             dy1 = dy0 + min(tile_size, y1 - y0)
#             eachtile = time.time()
#             print("start tiff:" + str(eachtile - tif))
#             cropped_valid = tile_result[dy0:dy1, dx0:dx1]
#             result[y0+dy0:y0+dy0+cropped_valid.shape[0], x0+dx0:x0+dx0+cropped_valid.shape[1]] = cropped_valid
#             eachtile = time.time()
#             print("end tiff:" + str(eachtile - tif))
#
#     dealwith = time.time()
#     print("dealwith tiff:" + str(dealwith-tif))
#     tifffile.imwrite(output_path, result)
#     write = time.time()
#     print("write tiff:" + str(write-tif))
#     print(f"✅ Done: {output_path}")
#
#     # 打印每个 tile 的耗时
#     for coord, t in tile_times:
#         print(f"Tile {coord} processed in {t:.3f}s")
#
# # -----------------------------
# # 示例运行
# # -----------------------------
# if __name__ == "__main__":
#     input_folder = r"D:\Lesson\opencv-learn\arbeit"
#     output_folder = r"D:\Lesson\opencv-learn\arbeit\result"
#     os.makedirs(output_folder, exist_ok=True)
#
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(".tif"):
#             input_path = os.path.join(input_folder, filename)
#             output_path = os.path.join(output_folder, os.path.splitext(filename)[0]+"_result.tif")
#             print(f"Processing {filename}")
#             process_large_tiff(input_path, output_path, tile_size=4096, overlap=32, workers=7, debug=False)








#############
# import os
# import time
# import cv2
# import numpy as np
# import tifffile
# from tqdm import tqdm
# from concurrent.futures import ProcessPoolExecutor
# from multiprocessing import shared_memory
#
# # -----------------------------
# # 核心 tile 处理逻辑
# # -----------------------------
# def count_holes(binary_img):
#     img_inverted = cv2.bitwise_not(binary_img)
#     contours, hierarchy = cv2.findContours(img_inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if hierarchy is None:
#         return 0
#     hierarchy = hierarchy[0]
#     return sum(1 for h in hierarchy if h[2] != -1)
#
# def process_tile(coord, shm_name, shape, dtype, debug=False):
#     # 连接共享内存
#     shm = shared_memory.SharedMemory(name=shm_name)
#     image = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
#
#     x0, y0, x1, y1 = coord
#     copystart = time.time()
#     print("copystart tiff:" )
#     tile = image[y0:y1, x0:x1]  # 只复制 tile
#     copyend = time.time()
#     print("copyend tiff:" + str(copyend-copystart))
#
#
#     if tile.ndim == 2:
#         tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
#
#     img_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
#     _, thresh_img = cv2.threshold(img_gray, 200, 255, 0)
#     contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     for contour in contours:
#         rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
#         M = cv2.moments(contour)
#         if M['m00'] != 0:
#             cx = int(M['m10']/M['m00'])
#             cy = int(M['m01']/M['m00'])
#         else:
#             cx = cy = -1
#
#         (ex, ey), radius = cv2.minEnclosingCircle(contour)
#         center = (int(ex), int(ey))
#
#         hc, wc = thresh_img.shape
#         if radius > 3.5 and radius < 12.5 and abs(cx-center[0])<3 and abs(cy-center[1])<3 and abs(rect_w-rect_h)<=3:
#             xc1 = max(cx-10,0); xc2 = min(cx+30,wc)
#             yc1 = max(cy-10,0); yc2 = min(cy+30,hc)
#             if xc1 >= xc2 or yc1 >= yc2: continue
#             candidate_crop = thresh_img[yc1:yc2, xc1:xc2]
#             holes = count_holes(candidate_crop)
#             num_labels, _, stats, _ = cv2.connectedComponentsWithStats(candidate_crop, connectivity=4)
#             for i in range(num_labels):
#                 areax = stats[i, cv2.CC_STAT_AREA]
#                 ww, hh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
#                 if 8 <= areax <= 40 and holes > 0:
#                     aspect_ratio = ww/hh
#                     if 0.6 <= aspect_ratio <= 1.1 and len(contour)>=5:
#                         ellipse = cv2.fitEllipse(contour)
#                         major, minor = ellipse[1]
#                         if major/minor > 0.7:
#                             if debug:
#                                 print(f"Found circle at {center} in tile {coord}")
#                             cv2.rectangle(tile, (rect_x, rect_y), (rect_x+rect_w, rect_y+rect_h), (0,0,255), 2)
#     shm.close()
#     return coord, tile
#
# # -----------------------------
# # 并行处理大 TIFF
# # -----------------------------
# def process_large_tiff(tiff_path, output_path, tile_size=2048, overlap=32, workers=None, debug=False):
#     start_time = time.time()
#     tif = time.time()
#     image = tifffile.imread(tiff_path)
#     readimage = time.time()
#     print("read tiff:" + str(readimage-tif))
#     H, W = image.shape[:2]
#
#     # 建立共享内存
#     shm = shared_memory.SharedMemory(create=True, size=image.nbytes)
#     shm_image = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)
#     copystart = time.time()
#     print("copystart tiff:" + str(copystart-tif))
#     np.copyto(shm_image, image)
#     copyend = time.time()
#     print("copyend tiff:" + str(copyend-tif))
#
#     result = np.zeros_like(image)
#
#     # 构建 tile 坐标
#     tile_coords = []
#     for y in range(0,H,tile_size):
#         for x in range(0,W,tile_size):
#             x_start = max(x-overlap,0)
#             y_start = max(y-overlap,0)
#             x_end = min(x+tile_size+overlap, W)
#             y_end = min(y+tile_size+overlap, H)
#             tile_coords.append((x_start, y_start, x_end, y_end))
#
#     dealstart = time.time()
#     print("dealwith tiff:" + str(dealstart-tif))
#     # 并行处理 tile
#     with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as executor:
#         futures = [executor.submit(process_tile, coord, shm.name, image.shape, image.dtype, debug) for coord in tile_coords]
#         for fut in tqdm(futures, desc="Processing tiles"):
#             coord, tile_result = fut.result()
#             x0, y0, x1, y1 = coord
#             dx0 = x0 if x0==0 else overlap
#             dy0 = y0 if y0==0 else overlap
#             dx1 = dx0 + min(tile_size, x1-x0)
#             dy1 = dy0 + min(tile_size, y1-y0)
#             eachtile = time.time()
#             print("start tiff:" + str(eachtile - tif))
#             cropped_valid = tile_result[dy0:dy1, dx0:dx1]
#             result[y0 + dy0:y0 + dy0 + cropped_valid.shape[0],
#             x0 + dx0:x0 + dx0 + cropped_valid.shape[1]] = cropped_valid
#             eachtile = time.time()
#             print("end tiff:" + str(eachtile - tif))
#
#
#     dealwith = time.time()
#     print("dealwith tiff:" + str(dealwith-tif))
#     tifffile.imwrite(output_path, result)
#     write = time.time()
#     print("write tiff:" + str(write-tif))
#     shm.close()
#     shm.unlink()
#     print(f"✅ Done: {output_path}, total time: {time.time()-start_time:.2f}s")
#
# # -----------------------------
# # 处理文件夹
# # -----------------------------
# def process_folder(input_folder, output_folder, tile_size=2048, overlap=32, workers=None, debug=False):
#     os.makedirs(output_folder, exist_ok=True)
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(".tif"):
#             input_path = os.path.join(input_folder, filename)
#             output_path = os.path.join(output_folder, os.path.splitext(filename)[0]+"_result.tif")
#             print(f"Processing {filename}")
#             process_large_tiff(input_path, output_path, tile_size, overlap, workers, debug)
#
# # -----------------------------
# # 示例运行
# # -----------------------------
# if __name__ == "__main__":
#     input_folder = r"D:\Lesson\opencv-learn\arbeit"
#     output_folder = r"D:\Lesson\opencv-learn\arbeit\result"
#     process_folder(input_folder, output_folder, tile_size=2048, overlap=32, workers=7, debug=False)
################
# import os
# import time
# import cv2
# import tifffile
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm
#
# # -----------------------------
# # 统计洞数量
# # -----------------------------
# def count_holes(binary_img):
#     img_inverted = cv2.bitwise_not(binary_img)
#     contours, hierarchy = cv2.findContours(img_inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     if hierarchy is None:
#         return 0
#
#     hierarchy = hierarchy[0]
#     hole_count = 0
#     if len(contours) < 2:
#         return 0
#
#     for i in range(len(contours)):
#         if hierarchy[i][2] != -1:
#             hole_count += 1
#
#     return hole_count
#
# # -----------------------------
# # 单 tile 处理
# # -----------------------------
# def process_tile(coord, bin_img, debug=False):
#     x0, y0, x1, y1 = coord
#     tile = bin_img[y0:y1, x0:x1]  # 视图，不拷贝
#
#     contours, hierarchy = cv2.findContours(tile, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     tile_color = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
#
#     for contour in contours:
#         rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
#         M = cv2.moments(contour)
#         cx = int(M['m10']/M['m00']) if M['m00'] != 0 else -1
#         cy = int(M['m01']/M['m00']) if M['m00'] != 0 else -1
#         (ex, ey), radius = cv2.minEnclosingCircle(contour)
#         center = (int(ex), int(ey))
#
#         hc, wc = tile.shape
#         if radius > 3.5 and radius < 12.5 and abs(cx-center[0])<3 and abs(cy-center[1])<3 and abs(rect_w-rect_h)<=3:
#             xc1 = max(cx-10,0); xc2 = min(cx+30,wc)
#             yc1 = max(cy-10,0); yc2 = min(cy+30,hc)
#             if xc1 >= xc2 or yc1 >= yc2:
#                 continue
#             candidate_crop = tile[yc1:yc2, xc1:xc2]
#             holes = count_holes(candidate_crop)
#             num_labels, _, stats, _ = cv2.connectedComponentsWithStats(candidate_crop, connectivity=4)
#             for i in range(num_labels):
#                 areax = stats[i, cv2.CC_STAT_AREA]
#                 ww, hh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
#                 if 8 <= areax <= 40 and holes > 0:
#                     aspect_ratio = ww/hh
#                     if 0.6 <= aspect_ratio <= 1.1 and len(contour)>=5:
#                         ellipse = cv2.fitEllipse(contour)
#                         major, minor = ellipse[1]
#                         if major/minor > 0.7:
#                             if debug:
#                                 print(f"Found circle at {rect_x},{rect_y} in tile {coord}")
#                             cv2.rectangle(tile_color, (rect_x, rect_y), (rect_x+rect_w, rect_y+rect_h), (0,0,255), 2)
#     return coord, tile_color
#
# # -----------------------------
# # 并行处理大 TIFF
# # -----------------------------
# def process_large_tiff(tiff_path, output_path, tile_size=2048, overlap=32, workers=None, debug=False):
#     start_time = time.time()
#     t0 = time.time()
#     image = tifffile.imread(tiff_path)
#     t1 = time.time()
#     print("read tiff:" + str(t1-t0))
#
#     H, W = image.shape[:2]
#
#     # 全图灰度 + 阈值
#     t2 = time.time()
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, bin_img = cv2.threshold(image_gray, 200, 255, 0)
#     t3 = time.time()
#     print("convert to grayscale + threshold:" + str(t3-t0))
#
#     # 构建 tile 坐标
#     tile_coords = []
#     for y in range(0,H,tile_size):
#         for x in range(0,W,tile_size):
#             x_start = max(x-overlap,0)
#             y_start = max(y-overlap,0)
#             x_end = min(x+tile_size+overlap, W)
#             y_end = min(y+tile_size+overlap, H)
#             tile_coords.append((x_start, y_start, x_end, y_end))
#
#     # 并行处理 tile
#     result = np.zeros_like(image)
#     t4 = time.time()
#     print("start processing tiles:" + str(t4-t0))
#     with ThreadPoolExecutor(max_workers=workers or os.cpu_count()) as executor:
#         futures = [executor.submit(process_tile, coord, bin_img, debug) for coord in tile_coords]
#         for fut in tqdm(futures, desc="Processing tiles"):
#             coord, tile_result = fut.result()
#             x0, y0, x1, y1 = coord
#             dx0 = x0 if x0==0 else overlap
#             dy0 = y0 if y0==0 else overlap
#             dx1 = dx0 + min(tile_size, x1-x0)
#             dy1 = dy0 + min(tile_size, y1-y0)
#             cropped_valid = tile_result[dy0:dy1, dx0:dx1]
#             result[y0 + dy0:y0 + dy0 + cropped_valid.shape[0],
#                    x0 + dx0:x0 + dx0 + cropped_valid.shape[1]] = cropped_valid
#     t5 = time.time()
#     print("dealwith tiff:" + str(t5-t0))
#
#     tifffile.imwrite(output_path, result)
#     t6 = time.time()
#     print("write tiff:" + str(t6-t0))
#     print(f"✅ Done: {output_path}, total time: {time.time()-start_time:.2f}s")
#
# # -----------------------------
# # 批量处理文件夹
# # -----------------------------
# def process_folder(folder_path, output_folder, tile_size=2048, overlap=32, workers=None, debug=False):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     for filename in os.listdir(folder_path):
#         if filename.lower().endswith(".tif"):
#             input_path = os.path.join(folder_path, filename)
#             output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_result.tif")
#             process_large_tiff(input_path, output_path, tile_size, overlap, workers, debug)
#
# # -----------------------------
# # main
# # -----------------------------
# if __name__ == "__main__":
#     folder_path = r'D:\Lesson\opencv-learn\arbeit'
#     output_folder = r'D:\Lesson\opencv-learn\arbeit\result'
#     process_folder(folder_path, output_folder, tile_size=4096, overlap=16, workers=8, debug=False)
#########
# import os
# import time
# import cv2
# import tifffile
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm
#
# # -----------------------------
# # 统计洞数量
# # -----------------------------
# def count_holes(binary_img):
#     img_inverted = cv2.bitwise_not(binary_img)
#     contours, hierarchy = cv2.findContours(img_inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     if hierarchy is None:
#         return 0
#
#     hierarchy = hierarchy[0]
#     hole_count = 0
#     if len(contours) < 2:
#         return 0
#
#     for i in range(len(contours)):
#         if hierarchy[i][2] != -1:
#             hole_count += 1
#
#     return hole_count
#
# # -----------------------------
# # 单 tile 处理，返回矩形列表
# # -----------------------------
# def process_tile(coord, bin_img, debug=False):
#     x0, y0, x1, y1 = coord
#     tile = bin_img[y0:y1, x0:x1]  # 视图，不拷贝
#     rects = []
#
#     contours, hierarchy = cv2.findContours(tile, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     for contour in contours:
#         rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
#         M = cv2.moments(contour)
#         cx = int(M['m10']/M['m00']) if M['m00'] != 0 else -1
#         cy = int(M['m01']/M['m00']) if M['m00'] != 0 else -1
#         (ex, ey), radius = cv2.minEnclosingCircle(contour)
#         center = (int(ex), int(ey))
#
#         hc, wc = tile.shape
#         if radius > 3.5 and radius < 12.5 and abs(cx-center[0])<3 and abs(cy-center[1])<3 and abs(rect_w-rect_h)<=3:
#             xc1 = max(cx-10,0); xc2 = min(cx+30,wc)
#             yc1 = max(cy-10,0); yc2 = min(cy+30,hc)
#             if xc1 >= xc2 or yc1 >= yc2:
#                 continue
#             candidate_crop = tile[yc1:yc2, xc1:xc2]
#             holes = count_holes(candidate_crop)
#             num_labels, _, stats, _ = cv2.connectedComponentsWithStats(candidate_crop, connectivity=4)
#             for i in range(num_labels):
#                 areax = stats[i, cv2.CC_STAT_AREA]
#                 ww, hh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
#                 if 8 <= areax <= 40 and holes > 0:
#                     aspect_ratio = ww/hh
#                     if 0.6 <= aspect_ratio <= 1.1 and len(contour)>=5:
#                         ellipse = cv2.fitEllipse(contour)
#                         major, minor = ellipse[1]
#                         if major/minor > 0.7:
#                             # 返回相对于全图的坐标
#                             rects.append((rect_x + x0, rect_y + y0, rect_w, rect_h))
#                             if debug:
#                                 print(f"Found circle at {rect_x + x0},{rect_y + y0} in tile {coord}")
#     return rects
#
# # -----------------------------
# # 并行处理大 TIFF
# # -----------------------------
# def process_large_tiff(tiff_path, output_path, tile_size=2048, overlap=32, workers=None, debug=False):
#     start_time = time.time()
#     t0 = time.time()
#     image = tifffile.imread(tiff_path)
#     t1 = time.time()
#     print("read tiff:" + str(t1-t0))
#
#     H, W = image.shape[:2]
#
#     # 全图灰度 + 阈值
#     t2 = time.time()
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, bin_img = cv2.threshold(image_gray, 200, 255, 0)
#     t3 = time.time()
#     print("convert to grayscale + threshold:" + str(t3-t0))
#
#     # 构建 tile 坐标
#     tile_coords = []
#     for y in range(0,H,tile_size):
#         for x in range(0,W,tile_size):
#             x_start = max(x-overlap,0)
#             y_start = max(y-overlap,0)
#             x_end = min(x+tile_size+overlap, W)
#             y_end = min(y+tile_size+overlap, H)
#             tile_coords.append((x_start, y_start, x_end, y_end))
#
#     # 并行处理 tile
#     all_rects = []
#     t4 = time.time()
#     print("start processing tiles:" + str(t4-t0))
#     with ThreadPoolExecutor(max_workers=workers or os.cpu_count()) as executor:
#         futures = [executor.submit(process_tile, coord, bin_img, debug) for coord in tile_coords]
#         for fut in tqdm(futures, desc="Processing tiles"):
#             rects = fut.result()
#             all_rects.extend(rects)
#     t5 = time.time()
#     print("dealwith tiff:" + str(t5-t0))
#
#     # 在原图上画矩形
#     for rect in all_rects:
#         x, y, w, h = rect
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,255), 2)
#         cv2.rectangle(image, (x-250, y-250), (x + w+250, y + h+250), (0, 0, 255), 10)
#
#     tp = time.time()
#     print("process time:" + str(tp-t4))
#     tifffile.imwrite(output_path, image)
#     t6 = time.time()
#     print("write tiff:" + str(t6-t0))
#
#     print(f"✅ Done: {output_path}, total time: {time.time()-start_time:.2f}s")
#
# # -----------------------------
# # 批量处理文件夹
# # -----------------------------
# def process_folder(folder_path, output_folder, tile_size=2048, overlap=32, workers=None, debug=False):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     for filename in os.listdir(folder_path):
#         if filename.lower().endswith(".tif"):
#             input_path = os.path.join(folder_path, filename)
#             output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_result.tif")
#             process_large_tiff(input_path, output_path, tile_size, overlap, workers, debug)
#
# # -----------------------------
# # main
# # -----------------------------
# if __name__ == "__main__":
#     folder_path = r'D:\Lesson\opencv-learn\arbeit'
#     output_folder = r'D:\Lesson\opencv-learn\arbeit\result'
#     process_folder(folder_path, output_folder, tile_size=4096, overlap=16, workers=8, debug=False)
