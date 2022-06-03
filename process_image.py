# import cv2
# import numpy as np
# import os
#
# img_path = "D:\\Python\\OpenCv\\OpenCV-e-meter\\ElectricityMeter\\evaluation\\temp\\t.jpg"
#
# erode = 2
# threshold = 37
# adjustment = 11
# iterations = 3
# blur = 3
#
#
# #  自定义函数：用于删除列表指定序号的轮廓
# #  输入 1：contours：原始轮廓
# #  输入 2：delete_list：待删除轮廓序号列表
# #  返回值：contours：筛选后轮廓
# def delete_contours(contours, delete_list):
#     delta = 0
#     for i in range(len(delete_list)):
#         # print("i= ", i)
#         del contours[delete_list[i] - delta]
#         delta = delta + 1
#     return contours
#
#
# # 4. 绘制轮廓函数
# # 自定义绘制轮廓的函数（为简化操作）
# # 输入1：winName：窗口名
# # 输入2：image：原图
# # 输入3：contours：轮廓
# # 输入4：draw_on_blank：绘制方式，True在白底上绘制，False:在原图image上绘制
# def drawMyContours(winName, image, contours, draw_on_blank):
#     # cv2.drawContours(image, contours, index, color, line_width)
#     # 输入参数：
#     # image:与原始图像大小相同的画布图像（也可以为原始图像）
#     # contours：轮廓（python列表）
#     # index：轮廓的索引（当设置为-1时，绘制所有轮廓）
#     # color：线条颜色，
#     # line_width：线条粗细
#     # 返回绘制了轮廓的图像image
#     if (draw_on_blank):  # 在白底上绘制轮廓
#         temp = np.ones(image.shape, dtype=np.uint8) * 255
#         cv2.drawContours(temp, contours, -1, (0, 0, 0), 2)
#     else:
#         temp = image.copy()
#         cv2.drawContours(temp, contours, -1, (0, 0, 255), 2)
#     cv2.imshow(winName, temp)
#
#
# def process_image():
#     img = cv2.imread(img_path)
#     alpha = float(2.5)
#
#     # Adjust the exposure
#     exposure_img = cv2.multiply(img, np.array([alpha]))
#
#     # Convert to grayscale
#     img2gray = cv2.cvtColor(exposure_img, cv2.COLOR_BGR2GRAY)
#
#     # Blur to reduce noise
#     img = cv2.GaussianBlur(img2gray, (blur, blur), 0)
#
#     cv2.imshow("Image", img)
#     cv2.waitKey(0)
#
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_gray = cv2.GaussianBlur(img_gray, (3, 3), 1)
#     # 求二值图像
#     # retv, thresh = cv2.threshold(img_gray, 125, 255, cv2.THRESH_TOZERO)
#     retv, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_TOZERO)
#     # 寻找轮廓
#     _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     print("find", len(contours), "contours")
#     # 绘制轮廓
#     cv2.drawContours(img_gray, contours, -1, (0, 0, 255), 3, lineType=cv2.LINE_AA)
#     # 显示图像
#     cv2.imshow('Contours', img_gray)
#     key = cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     font = cv2.FONT_HERSHEY_PLAIN
#
#     # 5.2使用轮廓长度滤波
#     min_size = 100
#     max_size = 280
#     delete_list = []
#     for i in range(len(contours)):
#         if (cv2.arcLength(contours[i], True) < min_size) or (cv2.arcLength(contours[i], True) > max_size):
#             delete_list.append(i)
#
#     # 根据列表序号删除不符合要求的轮廓
#     contours = delete_contours(contours, delete_list)
#     print(contours)
#     cv2.drawContours(img_gray, contours, -1, (0, 0, 255), 3, lineType=cv2.LINE_AA)
#     print(len(contours), "contours left after length filter")
#
#     for i in range(0, len(contours)):
#         (x, y, w, h) = cv2.boundingRect(contours[i])
#         cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2, cv2.LINE_AA)
#         cv2.putText(img_gray, "No.%d" % (i + 1), (x, y - 5), font, 0.8, (255, 0, 0), 2)
#
#     drawMyContours("contours after length filtering", img_gray, contours, True)
#     # drawMyContours("contours after length filtering", img_gray, contours, False)
#     # cv2.imshow('Contours', img_gray)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# def main():
#     process_image()
#
#
# if __name__ == "__main__":
#     main()
