import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # find_board(frame)
        find_lines(frame)

        cv2.imshow('Chessboard Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main2():
    img = cv2.imread("chessboard.jpg")

    # find_board(img)
    # find_lines(img)
    # find_corners(img)
    find_corners2(img)

    cv2.imshow('Chessboard Recognition', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_board(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    thresh_filter = cv2.medianBlur(thresh, 13)

    contours, _ = cv2.findContours(thresh_filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)


def find_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 75, 500)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=90)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


def find_corners(img):
    size = (7, 7)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, size)

    if found:
        cv2.drawChessboardCorners(img, size, corners, found)


def find_corners2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = (8, 8)
    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    found, corners = cv2.findChessboardCorners(gray, size, None)

    if found:
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), stop_criteria)
        cv2.drawChessboardCorners(img, size, corners, found)


if __name__ == "__main__":
    # main()
    main2()
