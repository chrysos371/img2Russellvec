# numbers2img.py
import csv
import os
import numpy as np
from PIL import Image
from typing import Union, Iterator, List


def csv_to_images(
    input_csv: Union[str, Iterator[List[str]]],
    output_dir: str
) -> int:
    """
    将 CSV 中的像素数组转为 48x48 灰度图，保存到 output_dir。

    参数:
        input_csv: CSV 文件路径，或一个可迭代的 CSV 行（list 的迭代器）。
        output_dir: 输出目录。
    返回:
        成功生成的图片数量。
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    if isinstance(input_csv, str):
        f = open(input_csv, "r", encoding="utf-8")
        reader = csv.reader(f)
    else:
        # 假定传入的是迭代器，每条记录是字符串列表
        reader = input_csv

    try:
        # 跳过表头逻辑（与原来相同）
        first_row = next(reader, None)
        rows = []
        if first_row and not first_row[0].strip().lstrip('-').isdigit():
            pass  # 已跳过表头
        elif first_row:
            rows = [first_row]

        # 正式处理
        for idx, row in enumerate(reader, start=1):
            if len(row) < 3:
                continue
            number_str = row[0].strip()
            if not number_str.lstrip('-').isdigit():
                continue

            pixels = list(map(int, row[1].strip().split()))
            string = row[2].strip()
            if len(pixels) != 2304:
                continue

            img = Image.fromarray(
                np.array(pixels, dtype=np.uint8).reshape(48, 48), 'L'
            )
            filename = f"{idx:05d}_{number_str}{string}.jpg"
            img.save(os.path.join(output_dir, filename), 'JPEG')
            count += 1
    finally:
        if isinstance(input_csv, str):
            f.close()

    return count


