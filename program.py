import cv2
import numpy as np
import os
import glob
import requests


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
    img = cv2.imread("initial_board.png")
    img_resized = cv2.resize(img, (550, 550))

    board = find_pieces(img_resized)

    print(board)

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
    board = np.zeros((8, 8))
    square_size = 68
    offset_x = 3
    offset_y = 3

    for y in range(0, 8):
        for x in range(0, 8):
            start_y = offset_y + square_size * y
            start_x = offset_x + square_size * x
            end_y = start_y + square_size
            end_x = start_x + square_size

            cv2.rectangle(img_copy, (start_y, start_x), (end_y, end_x), (255, 0, 0), 3, 8, 0)
            roi = img[start_y: end_y, start_x: end_x]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            if len(contours) > 1:
                board[y][x] = 1

    cv2.imshow("rectangles", img_copy)
    return board


def clear_files():
    files = glob.glob("squares/*")
    for f in files:
        os.remove(f)

    files = glob.glob("founds/*")
    for f in files:
        os.remove(f)


if __name__ == "__main__":
    clear_files()
    # main()  #webcam
    main2()  #photo
    # main3()  #cellphone
