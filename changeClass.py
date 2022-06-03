import os

filePath = "D:\\Python\\OpenCv\\OpenCV-e-meter\\ElectricityMeter\\training\\images_all"  # 文件夹路径
fileList = os.listdir(filePath)

changeStr = '15'
remainClass = 'meter'

os.chdir(filePath)

for file in fileList:
    print(file)
    # if file.endswith("classes.txt"):
    #     with open(os.path.join(filePath, file), "w") as f:
    #         f.write(remainClass)
    #     print('-------------------------')
    # elif file.endswith(".txt"):
    #     f = open(os.path.join(filePath, file))
    #     data = f.read()
    #     f.close()
    #     if data[0:2] == changeStr:
    #         with open(os.path.join(filePath, file), "w") as f:
    #             f.write(data.replace(changeStr, '0', 1))
    #             print('-------------------------')
    # elif file.endswith(".JPG"):
    #     os.rename(file, file.replace(".JPG", '.jpg'))  # 重命名后缀
    #     print('-------------------------')
    if file.endswith(".JPG"):
        os.rename(file, file.replace(".JPG", '.jpg'))  # 重命名后缀
        print('-------------------------')
