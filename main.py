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

        # find_board(frame)
        find_lines(frame)

        cv2.imshow("Chessboard Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main2():
    img = cv2.imread("initial_board.png")
    img_resized = cv2.resize(img, (550, 550))

    # find_board(img_resized)
    # find_lines(img_resized)

    draw_rectangles(img_resized)
    find_pieces()

    cv2.imshow('Chessboard Recognition', img_resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main3():
    url = "http://192.168.43.1:8080/shot.jpg"

    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)

        # find_board(img)
        find_lines(img)

        cv2.imshow("Chessboard Recognition", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def find_board(frame):
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #
    # lower_blue = np.array([70, 160, 0])
    # upper_blue = np.array([200, 255, 255])
    #
    # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #
    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #
    # for contour in contours:
    #     cv2.drawContours(frame, contour, -1, (0, 255, 0), 2)
    #
    # cv2.imshow("mask", mask)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 200, 230, cv2.THRESH_BINARY_INV)
    ret, thresh = cv2.threshold(gray, 100, 140, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imshow("thresh", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
    print(len(contours))
    for i, ctr in enumerate(contours):
        x, y, w, h = cv2.boundingRect(ctr)
        roi = frame[y:y+h, x:x+w]
        # cv2.imshow("roi{}".format(i), roi)

        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite("squares/{}.jpg".format(i), roi)

    # count = 1
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area < 5000:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         roi = frame[y:y + h, x:x + w]
    #         cv2.imwrite("squares/" + str(count) + ".jpg", roi)
    #         count += 1


def find_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 75, 500)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, maxLineGap=90)

    count = 1

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            roi = frame[y1:y1 + y2, x1:x1 + x2]
            cv2.imwrite("squares/" + str(count) + ".jpg", roi)
            count += 1


def draw_rectangles(img):
    square_size = 68
    offset_x = 3
    offset_y = 3
    count = 1

    for i in range(0, 8):
        for j in range(0, 8):
            start_x = offset_x + square_size * i
            start_y = offset_y + square_size * j
            end_x = start_x + square_size
            end_y = start_y + square_size

            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (255, 0, 0), 3, 8, 0)

            roi = img[start_y: end_y, start_x: end_x]
            cv2.imwrite("squares/{}.jpg".format(count), roi)
            count += 1


def find_pieces():
    files = glob.glob("squares/*")
    a = 1
    for f in files:
        square = cv2.imread(f)
        gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # detector = cv2.SimpleBlobDetector_create()
        # points = detector.detect(thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 1:
            cv2.putText(square, "X", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imwrite("founds/{}.jpg".format(a), square)
        a += 1


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
    # main2()  #photo
    # main3()  #cellphone
