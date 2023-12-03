import cv2
import mediapipe as mp

class poseDetector():

    def __init__(self, mode = False, complexity = 1, smooth = True,
                 enableSeg = False, smoothSeg = True,
                 detectConf = 0.5, trackConf = 0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.detectConf = detectConf
        self.trackConf = trackConf

        # Drawing the landmarks and lines between them for the skeleton
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose

        # Getting the pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth,
                                     self.enableSeg, self.smoothSeg,
                                     self.detectConf, self.trackConf)

    def findPose(self, img, draw = True):
        # Convert video to color and process the results
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        # Draw the landmarks and connections
        if (self.results.pose_landmarks):
            if (draw):
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                            self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw = True):
        lmList = []
        if (self.results.pose_landmarks):
            # Landmark number, Landmark position/visibility vector
            # num, [x, y, z, visibility]
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # Image Height, Image Width, Image Channel
                h, w, c = img.shape
                # X, Y, Z pixel of the image (possibly change Z)
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                # ID, X, Y, Z, Visibility
                lmList.append([id, cx, cy, cz])

                if (draw):
                    # Marking the (X, Y) points for confirmation they're right
                    cv2.circle(img, (cx, cy), 5,
                               (255, 0, 8), cv2.FILLED)
        return lmList

# Typically used for a test script of some sort.
def main():
    # Getting the video
    capture = cv2.VideoCapture('Videos/breakdance.mp4')

    detector = poseDetector()

    # Look/Draw loop
    while True:
        # Read in the video and store its data
        success, img = capture.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        cv2.imshow("Image", img)

        cv2.waitKey(1)

# If you run this in general it'll run the main() above.
# Otherwise, if you run a specific function, it won't run that main().
if __name__ == "__main__":
    main()