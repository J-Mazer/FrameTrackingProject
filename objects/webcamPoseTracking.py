import cv2
# cvzone is a helpful Computer Vision library from Murtaza Hassan
# Link to the Github documentation for it is in the README
from cvzone.PoseModule import PoseDetector

# Getting the video
# capture = cv2.VideoCapture('Videos/Naatu.mp4')
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Cannot open camera")
    exit()
# Getting the pose
detector = PoseDetector()
posList3D = []
posList2D = []

# Look/Draw loop
while True:
    # Read in the video and store its data
    success, img = capture.read()
    img = detector.findPose(img)

    lmList, bboxInfo = detector.findPosition(img)
    print("Height:", img.shape[0], "Width:", img.shape[0])
    if bboxInfo:
        lmString3D = ''
        lmString2D = ''
        for lm in lmList:
            lmString3D += f'{lm[0]},{img.shape[0]-lm[1]},{lm[2]},'
            lmString2D += f'{lm[0]},{img.shape[0]-lm[1]},'
        posList3D.append(lmString3D)
        posList2D.append(lmString2D)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        with open("Animation3D.txt", 'w') as f:
            f.writelines(["%s\n" % item for item in posList3D])
        with open("Animation2D.txt", 'w') as g:
            g.writelines(["%s\n" % item for item in posList2D])

    cv2.waitKey(1)
