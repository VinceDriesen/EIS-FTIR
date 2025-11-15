import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    cv2.namedWindow("Threshold")
    cv2.createTrackbar("T", "Threshold", 235, 255, lambda x: None)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame = frame[int(0.2*h):int(0.9*h), int(0*w):int(1*w)]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t = cv2.getTrackbarPos("T", "Threshold")
        _, bw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

        opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 50:
                continue

            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

            if len(c) >= 5:
                ellipse = cv2.fitEllipse(c)
                center = (int(ellipse[0][0]), int(ellipse[0][1]))
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                text = f"{center[0]}, {center[1]}"
                cv2.putText(frame, text, (center[0] + 10, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow("Gray", gray)
        cv2.imshow("Threshold", bw)
        cv2.imshow("Opened", opened)
        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
