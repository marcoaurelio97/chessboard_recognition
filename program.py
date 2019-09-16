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
    img = cv2.imread("board4.jpeg")
    # 889 x 877
    # img_resized = cv2.resize(img, (639, 627))

    board = find_pieces(img)

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
    square_size = 93
    offset_x = 13
    offset_y = 17

    count = 0

    for y in range(0, 8):
        for x in range(0, 8):
            start_y = offset_y + square_size * y
            start_x = offset_x + square_size * x
            end_y = start_y + square_size
            end_x = start_x + square_size

            cv2.rectangle(img_copy, (start_y, start_x), (end_y, end_x), (0, 0, 255), 1, 8, 0)
            roi = img[start_y: end_y, start_x: end_x]
            # cv2.imshow(f"{count}", roi)
            # cv2.imwrite("roi.png", roi)
            count += 1

            # roi = cv2.imread("roi.png")
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            lower_color = np.array([155, 5, -7])
            upper_color = np.array([175, 25, 73])
            mask1 = cv2.inRange(hsv, lower_color, upper_color)

            lower_color = np.array([-10, -10, 214])
            upper_color = np.array([10, 10, 294])
            mask2 = cv2.inRange(hsv, lower_color, upper_color)
            # res = cv2.bitwise_and(roi, roi, mask=mask)

            contours1, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(roi, contours1, -1, (0, 0, 255), 2)

            contours2, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(roi, contours2, -1, (0, 255, 0), 2)

            if len(contours1) or len(contours2):
                board[y][x] = 1

            # if (y == 1 and x == 2) or (y == 1 and x == 4):
                cv2.imshow(f"image{count}", roi)
                # cv2.imshow(f"mask1{count}", mask1)
                # cv2.imshow(f"mask2{count}", mask2)
                # cv2.imshow(F"result{count}", res)

    # cv2.imshow("rectangles", img_copy)
    return board


def histogram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])

    print(histogram)
    print(len(histogram))
    plt.plot(histogram)
    plt.show()


if __name__ == "__main__":
    # main()  #webcam
    main2()  #photo
    # main3()  #cellphone
