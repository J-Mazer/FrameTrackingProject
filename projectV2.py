# Import necessary libraries
import pygame as pg
import pyrr.matrix44
from OpenGL.GL import *
import numpy as np
from OpenGL.GLUT import *
import shaderLoader
import guiV1

import cv2
# cvzone is a helpful Computer Vision library from Murtaza Hassan
# Link to the Github documentation for it is in the README
from cvzone.PoseModule import PoseDetector




'====================='
'SET UP THE WINDOW'
'====================='
# Initialize pygame
pg.init()

# Set up OpenGL context version
pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)


# Create a window for graphics using OpenGL
width = 640
height = 480
screen = pg.display.set_mode((width, height), pg.OPENGL | pg.DOUBLEBUF)
pg.display.set_caption('Tracking')
# Background color
glClearColor(0.3, 0.4, 0.5, 1.0)

# Enable depth testing and point size
glEnable(GL_DEPTH_TEST)
glEnable(GL_PROGRAM_POINT_SIZE)

# Load in the shader
shader = shaderLoader.compile_shader(
    "shaders/vert.glsl", "shaders/frag.glsl")
glUseProgram(shader)

# Create Slider GUI for Camera FOV
cameraFOV = 130.0
gui = guiV1.SimpleGUI("Camera GUI")

fovSlider = gui.add_slider(
    "Camera FOV Slider", 30, 360, cameraFOV)

'====================='
'SET UP RENDERING PIPELINE'
'====================='
# Generate the VAO
vao = glGenVertexArrays(1)
glBindVertexArray(vao)

# Create the VBO
vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo)

# Configure position
position_loc = glGetAttribLocation(shader, "position")
glVertexAttribPointer(index=position_loc, size=2, type=GL_FLOAT,
                      normalized=GL_FALSE, stride=8,
                      pointer=ctypes.c_void_p(0))
glEnableVertexAttribArray(position_loc)

# Get the matrix locations.
view_loc = glGetUniformLocation(shader, "view_matrix")
proj_loc = glGetUniformLocation(shader, "proj_matrix")

'====================='
'CAPTURE VIDEO'
'====================='
# Getting the video
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Cannot open camera")
    exit()
# Getting the pose
detector = PoseDetector()

'====================='
'GET THE POINT POSITIONS AND DRAW THE MODEL'
'====================='


'====================='
'VIDEO CAPTURE AND DRAW LOOP'
'====================='
lmString3D = ''
lmString2D = ''
lockedCamera = False
# Run a loop to keep the program running
draw = True
while draw:
    # Quit out if need be
    for event in pg.event.get():
        if event.type == pg.QUIT:
            draw = False


    '---------------------'
    'VIDEO CAPTURE'
    '---------------------'
    # Read in the video and store its data
    success, readImg = capture.read()
    img = detector.findPose(readImg)

    # matrix = np.array(readImg, dtype=np.uint8)
    # matrix_bgr = cv2.cvtColor(matrix, cv2.COLOR_RGBA2RGB)
    # cv2.imwrite('bgImage.png', matrix_bgr)
    # cv2.waitKey(1)

    lmList, bboxInfo = detector.findPosition(img)
    pointList2D = []
    pointList3D = []
    # print("Height:", img.shape[0], "Width:", img.shape[0])
    if bboxInfo:
        lmString3D = ''
        lmString2D = ''
        for lm in lmList:
            lmString3D += f'{lm[0]},{img.shape[0] - lm[1]},{lm[2]},'
            lmString2D += f'{lm[0]},{img.shape[0] - lm[1]},'

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    # Once here, we have the position data of a full frame in posList2D.
    startI = 0
    endI = startI
    if lmString2D != "":
        while endI < len(lmString2D) and lmString2D[endI] != ',':
            endI += 1
            if endI < len(lmString2D) and lmString2D[endI] == ',':
                pointList2D.append(float(lmString2D[startI:endI]))
                startI = endI + 1
                endI = startI
    # Once here, we have the position data of a full frame in posList3D.
    startI = 0
    endI = startI
    if lmString3D != "":
        while endI < len(lmString3D) and lmString3D[endI] != ',':
            endI += 1
            if endI < len(lmString3D) and lmString3D[endI] == ',':
                pointList3D.append(float(lmString3D[startI:endI]))
                startI = endI + 1
                endI = startI

    # Once here, pointList is a list of all points in the given frame.
    fPointList = np.array(pointList2D, dtype="float32")

    fPointList3D = np.array(pointList3D, dtype="float32")

    '---------------------'
    'Drawing'
    '---------------------'
    # Draw Points
    glUseProgram(shader)
    glBindVertexArray(vao)

    # Using fPointList, aka the list of 2D points, to draw the points.
    glBufferData(GL_ARRAY_BUFFER, size=fPointList.nbytes,
                 data=fPointList, usage=GL_STATIC_DRAW)

    # Clear color buffer and depth buffer before drawing each frame
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


    glDrawArrays(GL_POINTS, 0, len(fPointList) // 2)

    glColor3f(255, 255, 255)
    glRasterPos2f(width/2, height/2)
    text = "Calibrating..."
    len = len(text)
    for i in range(len):
        glutBitmapCharacter(text[i])
    cameraFOV = fovSlider.get_value()

    # Camera center should be the center of the video capture,
    oHead = (fPointList3D[24] - fPointList3D[21],
             fPointList3D[26] - fPointList3D[23])
    p1Head = (fPointList3D[21], fPointList3D[23])
    p2Head = (fPointList3D[24], fPointList3D[26])
    hyHead = np.sqrt((p2Head[1] - oHead[1]) ** 2 +
                     (p1Head[0] - oHead[0]) ** 2)
    thetaHead = (np.arccos((p2Head[1] - oHead[1]) / hyHead) *
                 180 / np.pi - 120) * 3

    # Compute camera matrices
    eye = [width/2, height/2, width/5.5] # 4.5

    # Change target from center to [0, 0, 0].
    view_matrix = pyrr.matrix44.create_look_at(
        eye, [width/2, height/2, 0], np.asarray([0, 1, 0]))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix)

    proj_matrix = pyrr.matrix44.create_perspective_projection_matrix(
        cameraFOV, width/height, 0.1, 1000)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, proj_matrix)

    # Refresh the display to show what's been drawn
    pg.display.flip()


# Cleanup
glDeleteVertexArrays(1, [vao])
glDeleteBuffers(1, [vbo])
glDeleteProgram(shader)

pg.quit()   # Close the graphics window
quit()      # Exit the program