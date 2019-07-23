import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

    if not cap.isOpened():
        print("Cannot open the camera!")
        return -1

    while True:
        ret, frame = cap.read()

        find_board(frame)

        # cv2.imshow('Chessboard Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def find_board(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, ksize=(7, 7), sigmaX=0, sigmaY=0)  #aplica filtro de blur

    canny = cv2.Canny(blur, 50, 200, 3)  # tracejado do tabuleiro

    lines = cv2.HoughLines(canny, 1, 3.14/180, 120, 0, 0)

    cv2.imshow("sf", canny)

    # for i in range(0, len(lines)):
    #     print(lines[i])
    #     print("teste")
    #     print(lines)
    #     print("teste2")
    #     break
    #     rho = lines[i][0]
    #     theta = lines[i][1]

    contours = cv2.findContours(canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    mu = []
    print(contours)

    for i in range(0, len(contours)):
        # mu[i] = cv2.moments(contours[i], False)
        mu[i] = contours[i]

    mc = []

    for i in range(0, len(contours)):
        mc[i] = (mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00)


if __name__ == "__main__":
    main()
