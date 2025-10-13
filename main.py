import os.path
import cv2
import numpy as np

width, height = 400, 400

img_path ="Resources/2.JPG"
if not os.path.exists(img_path):
    print("File not exits")
    exit()

def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(imgContour, [cnt], -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0,255,0), 5)
            cv2.putText(imgContour, "Points: "+str(len(approx)),(x+w+20, y+20), cv2.FONT_HERSHEY_COMPLEX,0.7, (0,255,0),2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)
def back(x):
    pass

cv2.namedWindow("parameter")
cv2.resizeWindow("parameter", width,height)
cv2.createTrackbar("threshold1", "parameter",100,255,back)
cv2.createTrackbar("threshold2", "parameter",100,255,back)
while True:
    img = cv2.imread(img_path)
    imgBlur = cv2.GaussianBlur(img, (7,7), 1)
    imgGrey = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("threshold1", "parameter")
    threshold2 = cv2.getTrackbarPos("threshold2", "parameter")
    imgCanny = cv2.Canny(imgGrey, threshold1, threshold2)

    kernel = np.ones((5,5), np.uint8)
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    edgesContour = imgDil.copy()
    getContours(imgDil,edgesContour)

    imgDilC = cv2.cvtColor(imgDil, cv2.COLOR_GRAY2BGR)
    edgesContourC = cv2.cvtColor(edgesContour, cv2.COLOR_GRAY2BGR)
    vImg = np.hstack((img, imgDilC, edgesContourC))
    cv2.imshow("vImg", vImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()