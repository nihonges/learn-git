# import cv2
# import numpy as np
# from concurrent.futures import ProcessPoolExecutor, as_completed
#
#
# overlap = 16
# tile = img[max(0, y-overlap):y+tile_size+overlap,
#             max(0, x-overlap):x+tile_size+overlap]
# # ----------------------------
# # 每个 tile 的处理函数
# # ----------------------------
# def process_tile(args):
#     (tile, x, y) = args  # tile图像 + 左上角坐标
#     # 举例：做一个简单的高斯模糊 + 阈值操作
#     blur = cv2.GaussianBlur(tile, (5,5), 0)
#     _, binary = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY)
#     return (binary, x, y)
#
# # ----------------------------
# # 主函数：分块 + 并行 + 拼接
# # ----------------------------
# def process_large_image_in_parallel(image_path, tile_size=1024, workers=4):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     h, w = img.shape
#     result = np.zeros_like(img)
#
#     # 生成所有tile任务
#     tasks = []
#     for y in range(0, h, tile_size):
#         for x in range(0, w, tile_size):
#             tile = img[y:y+tile_size, x:x+tile_size]
#             tasks.append((tile, x, y))
#
#     # 并行执行任务
#     with ProcessPoolExecutor(max_workers=workers) as executor:
#         futures = [executor.submit(process_tile, t) for t in tasks]
#         for fut in as_completed(futures):
#             tile_result, x, y = fut.result()
#             result[y:y+tile_result.shape[0], x:x+tile_result.shape[1]] = tile_result
#
#     return result
#
# # ----------------------------
# # 测试
# # ----------------------------
# if __name__ == '__main__':
#     output = process_large_image_in_parallel('large_image.png', tile_size=1024, workers=8)
#     cv2.imwrite('result.png', output)
#     print("✅ Done: 并行分块处理完成")

####################################
# import cv2
# import numpy as np
# import os
# from concurrent.futures import ProcessPoolExecutor, as_completed
#
# def process_tile(tile, x0, y0):
#     """
#     tile: np.array, tile 图像
#     x0, y0: tile 在原图中的左上角坐标
#     返回：处理后的 tile
#     """
#     cropped_image = tile.copy()
#     img_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#     ret, thresh_img = cv2.threshold(img_gray, 170, 255, 0)
#     contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#
#     for contour in contours:
#         rect = cv2.minAreaRect(contour)
#         box = cv2.boxPoints(rect)
#         box = np.intp(box)
#         rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
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
#         if radius > 3.5 and radius < 12.5 and abs(cx - center[0]) < 3 and abs(cy - center[1]) < 3 \
#            and abs(rect_w - rect_h) < 3 and abs(tmp_w - tmp_h) < 3 and tmp_w > 0 and tmp_h > 0:
#
#             cv2.rectangle(cropped_image, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 255), 2)
#             cv2.rectangle(cropped_image, (rect_x - 50, rect_y - 50),
#                           (rect_x - 50 + rect_w + 100, rect_y - 50 + rect_h + 100), (0, 0, 255), 10)
#
#     return cropped_image
#
# # -----------------------------
# # 主函数：分块 + 重叠 + 并行 + 拼回
# # -----------------------------
# def image_processing_tiles_overlap(image_path, output_path, tile_size=1024, overlap=32, workers=8):
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Cannot read image: {image_path}")
#         return
#
#     H, W = image.shape[:2]
#     result = np.zeros_like(image)
#
#     # 生成 tile 列表
#     tiles = []
#     tile_coords = []
#
#     for y in range(0, H, tile_size):
#         for x in range(0, W, tile_size):
#             # 计算 tile 范围 + overlap
#             x_start = max(x - overlap, 0)
#             y_start = max(y - overlap, 0)
#             x_end = min(x + tile_size + overlap, W)
#             y_end = min(y + tile_size + overlap, H)
#
#             tile = image[y_start:y_end, x_start:x_end].copy()
#             tiles.append(tile)
#             tile_coords.append((x_start, y_start, x, y, min(x + tile_size, W), min(y + tile_size, H)))
#
#     # 并行处理
#     with ProcessPoolExecutor(max_workers=workers) as executor:
#         futures = [executor.submit(process_tile, t) for t in tiles]
#         for fut, coords in zip(futures, tile_coords):
#             tile_result = fut.result()
#             x_start, y_start, x_tile0, y_tile0, x_tile1, y_tile1 = coords
#             # 从 tile_result 中取有效区域（去掉重叠部分）
#             dx0 = x_tile0 - x_start
#             dy0 = y_tile0 - y_start
#             dx1 = dx0 + (x_tile1 - x_tile0)
#             dy1 = dy0 + (y_tile1 - y_tile0)
#             cropped_valid = tile_result[dy0:dy1, dx0:dx1]
#             result[y_tile0:y_tile1, x_tile0:x_tile1] = cropped_valid
#
#     cv2.imwrite(output_path, result)
#     print(f"✅ Done. Saved to {output_path}")
#
#
# # -----------------------------
# # 示例运行
# # -----------------------------
# if __name__ == "__main__":
#     workers = os.cpu_count() - 1
#     input_path = "input_images/large_image.png"
#     output_path = "output_images/large_image_result.png"
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#     image_processing_tiles_overlap(input_path, output_path, tile_size=1024, overlap=32, workers=8)
# #############################
import os
import tifffile
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor

# -----------------------------
# 核心 tile 处理逻辑
# -----------------------------
def process_tile(tile):
    cropped_image = tile.copy()
    img_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(img_gray, 170, 255, 0)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
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

        # 筛选圆形
        if radius > 3.5 and radius < 12.5 and abs(cx - center[0]) < 3 and abs(cy - center[1]) < 3 \
           and abs(rect_w - rect_h) < 3 and abs(tmp_w - tmp_h) < 3 and tmp_w > 0 and tmp_h > 0:

            # 拟合椭圆或画矩形
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(cropped_image, ellipse, (0, 0, 255), 2)
            else:
                cv2.rectangle(cropped_image, (rect_x, rect_y),
                              (rect_x + rect_w, rect_y + rect_h), (0, 0, 255), 2)

    return cropped_image

# -----------------------------
# 分块读取 TIFF + 并行处理 + 拼回
# -----------------------------
def process_large_tiff(tiff_path, output_path, tile_size=1024, overlap=32, workers=None):
    if workers is None:
        import os
        workers = max(os.cpu_count() - 1, 1)  # 默认用 CPU 核心数-1

    with tifffile.TiffFile(tiff_path) as tif:
        full_shape = tif.series[0].shape
        H, W = full_shape[:2]
        C = full_shape[2] if len(full_shape) == 3 else 3
        result = np.zeros((H, W, C), dtype=np.uint8)

        tile_coords = []
        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                x_start = max(x - overlap, 0)
                y_start = max(y - overlap, 0)
                x_end = min(x + tile_size + overlap, W)
                y_end = min(y + tile_size + overlap, H)
                tile_coords.append((x_start, y_start, x, y, min(x + tile_size, W), min(y + tile_size, H), x_end, y_end))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for coord in tile_coords:
                x_start, y_start, x_tile0, y_tile0, x_tile1, y_tile1, x_end, y_end = coord
                # 局部读取 tile
                with tifffile.TiffFile(tiff_path) as tif_local:
                    tile = tif_local.asarray(series=0)[y_start:y_end, x_start:x_end]
                    if tile.ndim == 2:
                        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
                futures.append(executor.submit(process_tile, tile))

            for fut, coord in zip(futures, tile_coords):
                tile_result = fut.result()
                x_start, y_start, x_tile0, y_tile0, x_tile1, y_tile1, x_end, y_end = coord
                dx0 = x_tile0 - x_start
                dy0 = y_tile0 - y_start
                dx1 = dx0 + (x_tile1 - x_tile0)
                dy1 = dy0 + (y_tile1 - y_tile0)
                cropped_valid = tile_result[dy0:dy1, dx0:dx1]
                result[y_tile0:y_tile1, x_tile0:x_tile1] = cropped_valid

    cv2.imwrite(output_path, result)
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
    input_folder = "input_tiffs"
    output_folder = "output_tiffs"
    process_folder(input_folder, output_folder, tile_size=1024, overlap=32, workers=None)
