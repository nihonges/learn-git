import os
import time
import cv2
import tifffile
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image
# -----------------------------
# 统计洞数量
# -----------------------------
def count_holes(binary_img):
    img_inverted = cv2.bitwise_not(binary_img)
    contours, hierarchy = cv2.findContours(img_inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return 0

    hierarchy = hierarchy[0]
    hole_count = 0
    if len(contours) < 2:
        return 0

    for i in range(len(contours)):
        if hierarchy[i][2] != -1:
            hole_count += 1

    return hole_count

# -----------------------------
# 单 tile 处理，返回矩形列表
# -----------------------------
def process_tile(coord, bin_img, debug=False):
    x0, y0, x1, y1 = coord
    tile = bin_img[y0:y1, x0:x1]  # 视图，不拷贝
    rects = []

    contours, hierarchy = cv2.findContours(tile, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00']) if M['m00'] != 0 else -1
        cy = int(M['m01']/M['m00']) if M['m00'] != 0 else -1
        (ex, ey), radius = cv2.minEnclosingCircle(contour)
        center = (int(ex), int(ey))

        hc, wc = tile.shape
        if radius > 3.5 and radius < 12.5 and abs(cx-center[0])<3 and abs(cy-center[1])<3 and abs(rect_w-rect_h)<=3:
            xc1 = max(cx-10,0); xc2 = min(cx+30,wc)
            yc1 = max(cy-10,0); yc2 = min(cy+30,hc)
            if xc1 >= xc2 or yc1 >= yc2:
                continue
            candidate_crop = tile[yc1:yc2, xc1:xc2]
            holes = count_holes(candidate_crop)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(candidate_crop, connectivity=4)
            for i in range(num_labels):
                areax = stats[i, cv2.CC_STAT_AREA]
                ww, hh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                if 8 <= areax <= 40 and holes > 0:
                    aspect_ratio = ww/hh
                    if 0.6 <= aspect_ratio <= 1.1 and len(contour)>=5:
                        ellipse = cv2.fitEllipse(contour)
                        major, minor = ellipse[1]
                        if major/minor > 0.7:
                            # 返回相对于全图的坐标
                            rects.append((rect_x + x0, rect_y + y0, rect_w, rect_h))
                            if debug:
                                print(f"Found circle at {rect_x + x0},{rect_y + y0} in tile {coord}")
    return rects

# -----------------------------
# 并行处理大 TIFF
# -----------------------------
def process_large_tiff(tiff_path, output_path, tile_size=2048, overlap=32, workers=None, debug=False):
    start_time = time.time()
    t0 = time.time()
    image = tifffile.imread(tiff_path)
    t1 = time.time()
    print("read tiff:" + str(t1-t0))

    H, W = image.shape[:2]

    # 全图灰度 + 阈值
    t2 = time.time()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(image_gray, 200, 255, 0)
    t3 = time.time()
    print("convert to grayscale + threshold:" + str(t3-t0))

    # 构建 tile 坐标
    tile_coords = []
    for y in range(0,H,tile_size):
        for x in range(0,W,tile_size):
            x_start = max(x-overlap,0)
            y_start = max(y-overlap,0)
            x_end = min(x+tile_size+overlap, W)
            y_end = min(y+tile_size+overlap, H)
            tile_coords.append((x_start, y_start, x_end, y_end))

    # 并行处理 tile
    all_rects = []
    t4 = time.time()
    print("start processing tiles:" + str(t4-t0))
    with ThreadPoolExecutor(max_workers=workers or os.cpu_count()) as executor:
        futures = [executor.submit(process_tile, coord, bin_img, debug) for coord in tile_coords]
        for fut in tqdm(futures, desc="Processing tiles"):
            rects = fut.result()
            all_rects.extend(rects)
    t5 = time.time()
    print("dealwith tiff:" + str(t5-t0))
    #
    # 在原图上画矩形
    for rect in all_rects:
        x, y, w, h = rect
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,255), 2)
        cv2.rectangle(image, (x-250, y-250), (x + w+250, y + h+250), (0, 0, 255), 10)

    t6 = time.time()
    print("process time:" + str(t6-t4))
    # cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    # image = Image.fromarray(np.uint8(image))
    # image.save(output_path)
    tifffile.imwrite(output_path, image, compression='jpeg')
    # tifffile.imwrite(output_path, image, compression=None, tile=(1024,1024))
    t7 = time.time()
    print("write tiff:" + str(t7-t0))

    print(f"✅ Done: {output_path}, total time: {time.time()-start_time:.2f}s")

# -----------------------------
# 批量处理文件夹
# -----------------------------
def process_folder(folder_path, output_folder, tile_size=2048, overlap=32, workers=None, debug=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".tif"):
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_result.tif")
            process_large_tiff(input_path, output_path, tile_size, overlap, workers, debug)

# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    folder_path = r'D:\Lesson\opencv-learn\arbeit'
    output_folder = r'D:\Lesson\opencv-learn\arbeit\result'
    process_folder(folder_path, output_folder, tile_size=4096, overlap=16, workers=8, debug=False)
