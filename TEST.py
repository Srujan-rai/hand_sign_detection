import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("MODEL/keras_model.h5", "MODEL/labels.txt")
offset = 20
imgSize = 300

folder = "DATA/C"
counter = 0
labels = ["A", "B", "C"]
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k+w)
            imgResize = cv2.resize(imgCrop,(wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((300-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction, index)

        else:

            k = imgSize / w
            hCal = math.ceil(k + h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((300 - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction, index)

    cv2.imshow("imageCrop", imgCrop)
    cv2.imshow("imageWhite", imgWhite)
    cv2.imshow("image", img)
    cv2.waitKey(1)
