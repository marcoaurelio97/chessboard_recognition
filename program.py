import cv2
import numpy as np
import os
import glob
import requests
from functions import send_positions
from matplotlib import pyplot as plt
import time


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
    img = cv2.imread("teste10.jpg")
    img = cv2.resize(img, (475, 485))

    board = find_pieces(img)

    print(board)
    # board = {
    #     0: [1, 1, 1, 1, 1, 1, 1, 1],
    #     1: [1, 1, 1, 0, 1, 1, 1, 1],
    #     2: [0, 0, 0, 0, 0, 0, 0, 0],
    #     3: [0, 0, 0, 1, 0, 0, 0, 0],
    #     4: [0, 0, 0, 0, 0, 0, 0, 0],
    #     5: [0, 0, 0, 0, 0, 0, 0, 0],
    #     6: [1, 1, 1, 1, 1, 1, 1, 1],
    #     7: [1, 1, 1, 1, 1, 1, 1, 1]
    # }

    # send_positions(board)

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
    square_size = 61
    offset_x = 7
    offset_y = 6

    for y in range(0, 8):
        for x in range(0, 8):
            start_y = offset_y + square_size * y
            start_x = offset_x + square_size * x
            end_y = start_y + square_size - 20
            end_x = start_x + square_size - 20

            cv2.rectangle(img, (start_y, start_x), (end_y, end_x), (0, 0, 255), 1, 8, 0)
            roi = img[start_y: end_y, start_x: end_x]

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            contours1, contours2 = get_contours(hsv)

            cv2.imwrite(f"testes/{y}{x}roi.jpg", roi)
            cv2.imwrite(f"testes/{y}{x}hsv.jpg", hsv)

            # if y == 6:
            #     cv2.imshow(f"{y}{x}roi", roi)
            #     cv2.imshow(f"{y}{x}hsv", hsv)
            #     print(f"{len(contours1)} {len(contours2)}")

            if len(contours1) > 0 or len(contours2) > 0:
                board[8 - y - 1][x] = 1

    cv2.imshow("rectangles", img)
    return board


def get_contours(hsv):
    lower_color = np.array([14, 46, 55])  # 14 108 55
    upper_color = np.array([34, 200, 258])  # 34 128 135
    white_mask = cv2.inRange(hsv, lower_color, upper_color)

    contours1, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(hsv, contours1, -1, (0, 0, 255), 2)

    lower_color = np.array([90, 67, -30])
    upper_color = np.array([110, 87, 50])
    black_mask = cv2.inRange(hsv, lower_color, upper_color)

    contours2, _ = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(hsv, contours2, -1, (0, 255, 0), 2)

    return contours1, contours2


def histogram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])

    print(histogram)
    print(len(histogram))
    plt.plot(histogram)
    plt.show()


def clear_files():
    files = glob.glob("testes/*")
    for f in files:
        os.remove(f)


if __name__ == "__main__":
    clear_files()
    # main()  #webcam
    main2()  #photo
    # main3()  #cellphone
