import os
import pydicom
from PIL import Image
from tqdm import tqdm
from multiprocessing.dummy import Pool
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# 遍历文件夹中的所有文件，并添加进度条
def process_file(file):
    # 如果文件是DCM格式，则进行转换
    if file.endswith(".dcm"):
        # 读取DCM文件
        dcm = pydicom.dcmread(file)

        # dcm -> 16bit PNG
        pixel_array = dcm.pixel_array
        pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))
        pixel_array = (pixel_array * 65535).astype(np.uint16)
        
        # Save
        image = Image.fromarray(pixel_array)
        png_path = file.replace(".dcm", ".png")
        image.save(png_path) 




if __name__ == "__main__":
    # 指定要遍历的文件夹路径
    folder_path = r"I:\dataset\Med\2023_Med_CQK"

    with ThreadPoolExecutor(max_workers=20) as executor:
        for root, dirs, files in tqdm(os.walk(folder_path)):
            for file in files:
                if file.endswith(".dcm"):
                    executor.submit(process_file, os.path.join(root, file))

    print("Conversion complete.")