import cv2
import numpy as np

image_hsv = None
pixel = (20, 60, 80)


def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y, x]

        upper = np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower = np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        print(pixel, lower, upper)

        image_mask = cv2.inRange(image_hsv, lower, upper)
        cv2.imshow("mask", image_mask)


def main():
    global image_hsv, pixel

    image_src = cv2.imread("teste13.jpg")
    image_src = cv2.resize(image_src, (500, 500))
    if image_src is None:
        print("the image read is None............")
        return
    cv2.imshow("bgr", image_src)

    cv2.namedWindow('hsv')
    cv2.setMouseCallback('hsv', pick_color)

    image_hsv = cv2.cvtColor(image_src, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", image_hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=='__main__':
    main()
