import cv2
import numpy as np
import glob
import random

RESIZED_IMAGE_WIDTH = 40
RESIZED_IMAGE_HEIGHT = 60
count = 0

# Load Yolo
net = cv2.dnn.readNet("D:\\Python\\OpenCv\\OpenCV-e-meter\\yolo_custom_detection\\yolov3_training_last_1.weights",
                      "D:\\Python\\OpenCv\\OpenCV-e-meter\\yolo_custom_detection\\yolov3_testing.cfg")
# Name custom object
classes = [""]


def detectReading(img_path):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    # Loading image
    img = cv2.imread(img_path)
    # img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=True)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            # cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
            img = img[y:y + h, x:x + w]

    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    return img


def processImg(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # 高斯平滑
    blur = cv2.GaussianBlur(img_gray, (3, 3), 6)

    # 求二值图像
    retv, thresh = cv2.threshold(blur, 140, 255, cv2.THRESH_TOZERO)
    # 膨胀
    thresh = cv2.dilate(thresh, kernel)
    # 寻找轮廓
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("find", len(contours), "contours")
    # 绘制轮廓
    cv2.drawContours(img_gray, contours, -1, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    # 显示图像
    drawMyContours("contours after length filtering", img_gray, contours, True)

    # 5.2使用轮廓长度滤波
    min_size = 60
    max_size = 400
    delete_list = []
    for i in range(len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])
        aspect = float(w) / h
        size = w * h
        if not (hierarchy[0, i, 2] > 0 or hierarchy[0, i, 3] > 0):  # 有父轮廓或子轮廓:
            if (cv2.arcLength(contours[i], True) < min_size) or (cv2.arcLength(contours[i], True) > max_size):
                delete_list.append(i)
            elif w > h * 1.1:
                delete_list.append(i)
            # elif aspect < 0.4:
            #     delete_list.append(i)
        else:
            if cv2.arcLength(contours[i], True) > max_size:
                delete_list.append(i)
            print(aspect)
    # 根据列表序号删除不符合要求的轮廓
    contours = delete_contours(contours, delete_list)
    cv2.drawContours(img_gray, contours, -1, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    print(len(contours), "contours left after length filter")

    for i in range(0, len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img_gray, "No.%d" % (i + 1), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 2)

    new = drawMyContours("contours after length filtering", img_gray, contours, True)
    cropImg(new, contours)
    return new

    # drawMyContours("contours after length filtering", img_gray, contours, False)


def delete_contours(contours, delete_list):
    '''
     自定义函数：用于删除列表指定序号的轮廓
     输入 1：contours：原始轮廓
     输入 2：delete_list：待删除轮廓序号列表
     返回值：contours：筛选后轮廓
    '''
    delta = 0
    for i in range(len(delete_list)):
        # print("i= ", i)
        del contours[delete_list[i] - delta]
        delta = delta + 1
    return contours


def drawMyContours(winName, image, contours, draw_on_blank):
    '''
    4. 绘制轮廓函数
    自定义绘制轮廓的函数（为简化操作）
    输入1：winName：窗口名
    输入2：image：原图
    输入3：contours：轮廓
    输入4：draw_on_blank：绘制方式，True在白底上绘制，False:在原图image上绘制
    cv2.drawContours(image, contours, index, color, line_width)
    输入参数：
    image:与原始图像大小相同的画布图像（也可以为原始图像）
    contours：轮廓（python列表）
    index：轮廓的索引（当设置为-1时，绘制所有轮廓）
    color：线条颜色，
    line_width：线条粗细
    返回绘制了轮廓的图像image
    '''
    if (draw_on_blank):  # 在白底上绘制轮廓
        temp = np.ones(image.shape, dtype=np.uint8) * 255
        cv2.drawContours(temp, contours, -1, (0, 0, 0), 2)
    else:
        temp = image.copy()
        cv2.drawContours(temp, contours, -1, (0, 0, 255), 2)
    cv2.imshow(winName, temp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return temp


def cropImg(img, contours):
    global count
    for i in range(0, len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])
        cropped = img[y:y + h, x: x + w]
        Resized = cv2.resize(cropped, (40, 60))
        # cv2.imshow("c", Resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite("D:\\Python\\OpenCv\\OpenCV-e-meter\\training\\%s.jpg" % str(count), Resized)
        count += 1


def main():
    # Images path
    images_path = glob.glob(r"D:\Python\OpenCv\OpenCV-e-meter\ElectricityMeter\evaluation\*.jpg")
    # random.shuffle(images_path)
    # loop through all the images
    for img_path in images_path:
        img = detectReading(img_path)
        reading = processImg(img)


if __name__ == "__main__":
    main()
