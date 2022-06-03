import cv2
import time
import sys
import glob

from ImageProcessing import FrameProcessor, ProcessingVariables
from DisplayUtils.TileDisplay import show_img, reset_tiles

window_name = 'Playground'
# file_name = 'evaluation/2.jpg'
version = '_3_0'

erode = ProcessingVariables.erode
threshold = ProcessingVariables.threshold
adjustment = ProcessingVariables.adjustment
iterations = ProcessingVariables.iterations
blur = ProcessingVariables.blur
desired_aspect = ProcessingVariables.desired_aspect
digit_one_aspect = ProcessingVariables.digit_one_aspect
aspect_buffer = ProcessingVariables.aspect_buffer
alpha = ProcessingVariables.alpha

std_height = 90

frameProcessor = FrameProcessor(std_height, version, True, False)


def main():
    images_path = glob.glob(r"D:\Python\OpenCv\OpenCV-e-meter\MyOCR\evaluation\normal\*.jpg")
    for file_name in images_path:
        img_file = file_name
        if len(sys.argv) == 2:
            img_file = sys.argv[1]
        setup_ui()
        frameProcessor.set_image(img_file)
        process_image()
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_image():
    reset_tiles()
    start_time = time.time()
    debug_images, output = frameProcessor.process_image(blur=blur, threshold=threshold, adjustment=adjustment,
                                                        erode=erode, iterations=iterations,
                                                        desired_aspect=desired_aspect,
                                                        digit_one_aspect=digit_one_aspect, aspect_buffer=aspect_buffer,
                                                        alpha=alpha)

    for image in debug_images:
        show_img(image[0], image[1])

    print("Processed image in %s seconds" % (time.time() - start_time))

    cv2.imshow(window_name, frameProcessor.img)
    cv2.moveWindow(window_name, 600, 600)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def setup_ui():
    cv2.namedWindow(window_name)
    cv2.createTrackbar('Threshold', window_name, int(threshold), 500, change_threshold)
    cv2.createTrackbar('Iterations', window_name, int(iterations), 5, change_iterations)
    cv2.createTrackbar('Adjust', window_name, int(adjustment), 200, change_adj)
    cv2.createTrackbar('Erode', window_name, int(erode), 5, change_erode)
    cv2.createTrackbar('Blur', window_name, int(blur), 25, change_blur)
    cv2.createTrackbar('desired_aspect', window_name, int(desired_aspect * 100), 100, change_desired_aspect)
    cv2.createTrackbar('digit_one_aspect', window_name, int(digit_one_aspect * 100), 100, change_digit_one_aspect)
    cv2.createTrackbar('aspect_buffer', window_name, int(aspect_buffer * 100), 100, change_aspect_buffer)
    cv2.createTrackbar('alpha', window_name, int(alpha), 250, change_alpha)


def change_alpha(x):
    global alpha
    print('Adjust: ' + str(x))
    alpha = x
    process_image()


def change_desired_aspect(x):
    global desired_aspect
    print('Adjust: ' + str(x))
    desired_aspect = x / 100
    process_image()


def change_digit_one_aspect(x):
    global digit_one_aspect
    print('Adjust: ' + str(x))
    digit_one_aspect = x / 100
    process_image()


def change_aspect_buffer(x):
    global aspect_buffer
    print('Adjust: ' + str(x))
    aspect_buffer = x / 100
    process_image()


def change_blur(x):
    global blur
    print('Adjust: ' + str(x))
    if x % 2 == 0:
        x += 1
    blur = x
    process_image()


def change_adj(x):
    global adjustment
    print('Adjust: ' + str(x))
    adjustment = x
    process_image()


def change_erode(x):
    global erode
    print('Erode: ' + str(x))
    erode = x
    process_image()


def change_iterations(x):
    print('Iterations: ' + str(x))
    global iterations
    iterations = x
    process_image()


def change_threshold(x):
    print('Threshold: ' + str(x))
    global threshold

    if x % 2 == 0:
        x += 1
    threshold = x
    process_image()


if __name__ == "__main__":
    main()
