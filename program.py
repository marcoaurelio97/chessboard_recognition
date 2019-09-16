import cv2
import numpy as np
import os
import glob
import requests
from functions import send_positions
from matplotlib import pyplot as plt


def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        find_pieces(frame)

        cv2.imshow("Chessboard Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main2():
    img = cv2.imread("board1.jpg")
    # 889 x 877
    img_resized = cv2.resize(img, (639, 627))

    board = find_pieces(img_resized)

    print(board)
    # send_positions(board)

    # cv2.imshow('Chessboard Recognition', img_resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main3():
    url = "http://192.168.43.1:8080/shot.jpg"

    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)

        cv2.imshow("Chessboard Recognition", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def find_pieces(img):
    img_copy = img.copy()
    # board = np.zeros((8, 8))
    board = {
        0: [0, 0, 0, 0, 0, 0, 0, 0],
        1: [0, 0, 0, 0, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 0, 0, 0],
        3: [0, 0, 0, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 0, 0, 0, 0],
        5: [0, 0, 0, 0, 0, 0, 0, 0],
        6: [0, 0, 0, 0, 0, 0, 0, 0],
        7: [0, 0, 0, 0, 0, 0, 0, 0]
    }
    square_size = 78
    offset_x = 2
    offset_y = 4

    count = 0

    for y in range(0, 8):
        for x in range(0, 8):
            start_y = offset_y + square_size * y
            start_x = offset_x + square_size * x
            end_y = start_y + square_size
            end_x = start_x + square_size

            cv2.rectangle(img_copy, (start_y, start_x), (end_y, end_x), (255, 0, 0), 3, 8, 0)
            roi = img[start_y: end_y, start_x: end_x]
            # cv2.imshow(f"{count}", roi)
            cv2.imwrite("histogram.png", roi)
            count += 1

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])

            print(histogram)
            print(len(histogram))
            plt.plot(histogram)
            plt.show()

            # ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            # cv2.imshow(f"{count}", thresh)
            # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # print(len(contours))
            # if len(contours) > 5:
            #     board[y][x] = 1

            # hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # lower_white = (0, 0, 170)
            # upper_white = (131, 255, 255)
            # mask = cv2.inRange(hsv, lower_white, upper_white)
            # # res = cv2.bitwise_and(roi, roi, mask=mask)
            # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # # if contours:
            # cv2.drawContours(roi, contours, -1, (0, 255, 0), 1)
            # if len(contours) > 5:
            #     board[y][x] = 1

    cv2.imshow("rectangles", img_copy)
    return board


def teste(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 60)
    corners = np.int0(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    cv2.imshow("agora vai", img)


if __name__ == "__main__":
    # main()  #webcam
    main2()  #photo
    # main3()  #cellphone
