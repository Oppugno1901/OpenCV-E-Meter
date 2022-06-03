import os
import shutil
import random

src = "D:\\Python\\OpenCv\\OpenCV-e-meter\\ElectricityMeter\\training\\images_all"  # 原文件夹路径
des = "D:\\Python\\OpenCv\\OpenCV-e-meter\\ElectricityMeter\\training\\images"  # 目标文件夹路径

isCopied = []
count = 500

while count:
    num = random.randint(1, 1500)
    if num not in isCopied:
        count -= 1
        isCopied.append(num)
        full_file_name = os.path.join(src, str(num) + '.jpg')  # 把文件的完整路径得到
        shutil.copy(full_file_name, des)  # shutil.copy函数放入原文件的路径文件全名  然后放入目标文件夹
