import cv2
import frameTrackingModule as track
from matplotlib import pyplot as plt
import numpy as np

# Getting the video
capture = cv2.VideoCapture('Videos/Naatu.mp4')

detector = track.poseDetector()

frames = 0
listOfList = np.asarray([])
print(listOfList)
# Look/Draw loop
while True:
    # Read in the video and store its data
    success, img = capture.read()
    if (success):
        img = detector.findPose(img)
        lmList = detector.findPosition(img)

        cv2.imshow("Image", img)

        cv2.waitKey(1)
        print(np.asarray(lmList).shape)
        for i in range(len(lmList)):
            listOfList = np.append(listOfList,
                                    np.asarray([lmList[i][1],
                                                lmList[i][2]]))
    else:
        break

# Let's test out drawing it.
# X = 764, Y = 708
time = 1 / 24

fig = plt.figure()

listOfList = np.asarray(listOfList)
print(listOfList.shape)
print(len(listOfList))
print(listOfList)
pixelCount = 100
for i in range(len(listOfList) // 66):

    # Draw loop for the animation.
    plt.clf()
    plt.xlim([0, 764])
    plt.ylim([0, 708])

    xLine0to1 = np.linspace(listOfList[i], listOfList[i+2], pixelCount)
    yLine0to1 = np.linspace(listOfList[i+1], listOfList[i+3], pixelCount)
    plt.plot(xLine0to1, yLine0to1, color='red')

    # Body
    xLine11to12 = np.linspace(listOfList[i+22], listOfList[i+24], pixelCount)
    yLine11to12 = np.linspace(listOfList[i+23], listOfList[i+25], pixelCount)
    plt.plot(xLine11to12, yLine11to12, color='red')
    xLine11to23 = np.linspace(listOfList[i+22], listOfList[i+46], pixelCount)
    yLine11to23 = np.linspace(listOfList[i+23], listOfList[i+47], pixelCount)
    plt.plot(xLine11to23, yLine11to23, color='red')
    xLine12to24 = np.linspace(listOfList[i+24], listOfList[i+48], pixelCount)
    yLine12to24 = np.linspace(listOfList[i+25], listOfList[i+49], pixelCount)
    plt.plot(xLine12to24, yLine12to24, color='red')
    xLine23to24 = np.linspace(listOfList[i+46], listOfList[i+48], pixelCount)
    yLine23to24 = np.linspace(listOfList[i+47], listOfList[i+49], pixelCount)
    plt.plot(xLine23to24, yLine23to24, color='red')

    plt.pause(.1)
plt.show()
