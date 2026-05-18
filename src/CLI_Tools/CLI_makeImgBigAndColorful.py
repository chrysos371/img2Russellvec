import csv
import os
import argparse
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
'''
用途
将所有 CSV 中的 48x48 灰度图像 resize 为 224x224 RGB，图片集中存入 img/ 文件夹，
CSV 第二列替换为图片路径。超过 1000 行自动分块。
依赖
pip install pillow numpy torchvision
用法
python CLI_makeImgBigAndColorful.py input_dir output_dir
'''


def process_csv(input_path, output_dir, resize, global_counter):
    """
    处理单个 CSV 文件：
    - 图片统一存入 output_dir/img/ 下，编号从 global_counter 开始自增
    - 返回更新后的 global_counter
    """
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    with open(input_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        data_rows = list(reader)

    total_rows = len(data_rows)
    base_name = os.path.splitext(os.path.basename(input_path))[0] + "_processed"

    # 修改表头第二列为 'image_path'
    new_header = header.copy()
    new_header[1] = "image_path"

    def write_chunk(rows, part_label=""):
        nonlocal global_counter   # 使用外部计数器保证全局编号唯一
        csv_name = f"{base_name}{part_label}.csv"
        csv_path = os.path.join(output_dir, csv_name)

        with open(csv_path, "w", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(new_header)

            for row in rows:
                # 解析原灰度像素串并 resize
                gray_vals = np.array(row[1].strip().split(), dtype=np.uint8).reshape(48, 48)
                img = Image.fromarray(gray_vals, mode="L").convert("RGB")
                img_resized = resize(img)

                # 保存图片（全局编号，固定6位宽度）
                img_filename = f"{global_counter:06d}.png"
                img_path = os.path.join(img_dir, img_filename)
                img_resized.save(img_path)

                # CSV 中写入相对于 output_dir 的路径
                row[1] = os.path.join("img", img_filename)
                writer.writerow(row)

                global_counter += 1

        print(f"已保存: {csv_path} (共 {len(rows)} 行)")

    # 按需分块
    if total_rows <= 1000:
        write_chunk(data_rows)
    else:
        for part_num, i in enumerate(range(0, total_rows, 1000), start=1):
            chunk = data_rows[i:i+1000]
            write_chunk(chunk, f"_part{part_num}")

    return global_counter

def main():
    parser = argparse.ArgumentParser(
        description="批量处理CSV中48x48灰度图 -> 224x224 RGB图，图片集中存放，CSV记录路径"
    )
    parser.add_argument("input_dir", help="包含CSV文件的输入文件夹")
    parser.add_argument("output_dir", help="输出文件夹")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    resize = transforms.Resize((224, 224))
    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))

    if not csv_files:
        print("未找到CSV文件。")
        return

    global_counter = 0  # 全局图片编号计数器

    for fpath in csv_files:
        print(f"正在处理: {fpath}")
        global_counter = process_csv(fpath, args.output_dir, resize, global_counter)

if __name__ == "__main__":
    main()