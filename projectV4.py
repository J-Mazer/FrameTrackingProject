# Import necessary libraries
import pygame as pg
import pyrr.matrix44
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import os
from PIL import Image
import shaderLoader
import guiV1
import time
import datetime
import cv2
# cvzone is a helpful Computer Vision library from Murtaza Hassan
# Link to the Github documentation for it is in the README
from cvzone.PoseModule import PoseDetector

'====================='
'SET UP CUBEMAP'
'====================='

def load_image(filename, format="RGB", flip=False):
    img = pg.image.load(filename)
    img_data = pg.image.tobytes(img, format, flip)
    w, h = img.get_size()
    return img_data, w, h

def create_cubemap_texture(cubemap_paths):
    # Generate a texture ID
    texture_id = glGenTextures(1)

    # Bind the texture as a cubemap
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id)

    # Define texture parameters
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Define the faces of the cubemap
    faces = [GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
             GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
             GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z]

    # Load and bind images to the corresponding faces
    for i in range(6):
        img_data, img_w, img_h = load_image(cubemap_paths[i], format="RGB", flip=False)
        glTexImage2D(faces[i], 0, GL_RGB, img_w, img_h, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    # Generate mipmaps
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP)

    # Unbind the texture
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0)

    return texture_id

'====================='
'SET UP THE WINDOW'
'====================='
# Initialize pygame
pg.init()

# Set up OpenGL context version
pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)

# Create a window for graphics using OpenGL
width = 480
height = 480
screen = pg.display.set_mode((width, height), pg.OPENGL | pg.DOUBLEBUF)
pg.display.set_caption('Tracking')
# Background color
glClearColor(0.1098, 1.0, 0.0117, 1.0)

# Enable depth testing and point size
glEnable(GL_DEPTH_TEST)
glEnable(GL_PROGRAM_POINT_SIZE)

# Load in the shader
shader = shaderLoader.compile_shader(
    "shaders/vert.glsl", "shaders/frag.glsl")
glUseProgram(shader)

shaderSkybox = shaderLoader.compile_shader("shaders/vert_skybox.glsl",
                           "shaders/frag_skybox.glsl")

# Create Slider GUI for Camera FOV
cameraFOV = 130.0
gui = guiV1.SimpleGUI("Camera GUI")

fovSlider = gui.add_slider(
    "Camera FOV Slider", 30, 360, cameraFOV)

'====================='
'SET UP SKYBOX STUFFS'
'====================='
cubemap_paths = [
    "skybox/left.png",
    "skybox/left.png",
    "skybox/left.png",
    "skybox/left.png",
    "skybox/left.png",
    "skybox/left.png"
]
cubemap_texture = create_cubemap_texture(cubemap_paths)
size_position = 3

# Define the vertices of the quad.
quad_vertices = (
            # Position
            -1, -1,
             1, -1,
             1,  1,
             1,  1,
            -1,  1,
            -1, -1
)
skybox_vertices = np.array(quad_vertices, dtype=np.float32)

skybox_size_position = 2       # x, y, z
skybox_stride = skybox_size_position * 4
skybox_offset_position = 0
quad_n_vertices = len(skybox_vertices) // skybox_size_position  # number of vertices

vao_quad = glGenVertexArrays(1)
glBindVertexArray(vao_quad)
vbo_quad = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo_quad)
quad_n_vertices = len(skybox_vertices) // skybox_size_position
glBufferData(GL_ARRAY_BUFFER,
             skybox_vertices.nbytes,
             skybox_vertices,
             GL_STATIC_DRAW)

glVertexAttribPointer(
    index=0,                                                # index of normal
    size=skybox_size_position,       # x, y, z,             # number of position components
    type=GL_FLOAT,                                          # type of the components
    normalized=GL_FALSE,                                    # data is/is not normalized
    stride=skybox_stride,                                          # number of bytes between verticies
    pointer=ctypes.c_void_p(skybox_offset_position)                  # pointer of the beginning of the position data
)
glEnableVertexAttribArray(0)
glUseProgram(shaderSkybox)

view_loc = glGetUniformLocation(shader, "view_matrix")
proj_loc = glGetUniformLocation(shader, "proj_matrix")

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
cube_map_loc = glGetUniformLocation(shaderSkybox, "cubeMapTex")
inv_proj_loc = glGetUniformLocation(shaderSkybox, "invViewProjectionMatrix")

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
# For storing the calibration data.
calibratedPose = False
takenPicture = False
startedTimer = False
startedTimerPicture = False
timer = time.time_ns()
second = 1
inFrame = False
notInFrame = True
calibList2D = np.array([0]*66, dtype="float32")
calibList3D = np.array([0]*99, dtype="float32")

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
    # 66 elements
    fPointList = np.array(pointList2D, dtype="float32")
    # 99 elements
    fPointList3D = np.array(pointList3D, dtype="float32")

    '---------------------'
    'CALIBRATION'
    '---------------------'
    # Check if frame is empty (no human detected)
    frameIsEmpty = len(lmList) == 0

    # Taking the initial background picture:
    if not takenPicture:
        backgroundImg = readImg
        # Timer to take picture when frame is empty
        if not startedTimerPicture and not takenPicture:
            oldTimer = timer
            second = 1
            print("Taking picture of the environment in...")
            startedTimerPicture = True

        if startedTimerPicture and not takenPicture:
            timer = time.time_ns()
            deltaTime = (timer - oldTimer) / 1e9
            if deltaTime >= second and second <= 5:
                print("0:00:0" + str(6 - second) + "...")
                second += 1
            if second > 5:
                startedTimerPicture = False

        if not startedTimerPicture:
            if not frameIsEmpty and not startedTimer:
                print("Please make sure nobody is visible in frame! Restarting "
                      "timer...")
                second = 1
            elif frameIsEmpty and not takenPicture:
                print("Picture taken!")
                takenPicture = True
                # Once this flag flips, the backgroundImg will stay constant.
                os.chdir('skybox')
                backgroundImg = cv2.flip(backgroundImg, 1)
                backgroundImg = backgroundImg[:, 80:560]
                cv2.imwrite("left.png", backgroundImg)

        if not takenPicture:
            continue

    # Taking the initial calibration picture:
    if not calibratedPose:
        # Calibration timer
        if not startedTimer and not calibratedPose:
            oldTimer = timer
            second = 1
            print("Taking picture of the actor in...")
            startedTimer = True

        if startedTimer and not calibratedPose:
            timer = time.time_ns()
            deltaTime = (timer - oldTimer) / 1e9
            if deltaTime >= second and second <= 5:
                print("0:00:0"+str(6-second)+"...")
                second += 1
            if second > 5:
                startedTimer = False

        if not startedTimer:
            if len(fPointList) < 66:
                print("No data points found!  Make sure you are in frame!")
                continue
            elif fPointList[63] < 0 and fPointList[65] < 0:
                print("Please make sure your entire body is in frame!")
                continue
            elif fPointList[63] >= 0 and fPointList[65] >= 0:
                inFrame = True

        #print(fPointList[63], fPointList[65])

        if not calibratedPose and inFrame:
            calibList2D = fPointList
            calibList3D = fPointList3D
            print("Calibration complete!")
            calibratedPose = True

    #prevents from crashing if no data points are found
    if len(fPointList) < 66:
        continue


    # Shift all x coordinates over by 80.
    for i in range(len(fPointList)):
        if i % 2 == 0:
            fPointList[i] -= 80

    '---------------------'
    'Drawing'
    '---------------------'
    # Draw Points
    glUseProgram(shader)
    glBindVertexArray(vao)

    # Using fPointList, aka the list of 2D points, to draw the points.
    glBufferData(GL_ARRAY_BUFFER, size=fPointList.nbytes,
                 data=fPointList, usage=GL_STATIC_DRAW) # calibList2D

    # Clear color buffer and depth buffer before drawing each frame
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


    glDrawArrays(GL_POINTS, 0, len(fPointList) // 2)

    # Camera center should be the center of the video capture,
    oHead = (fPointList3D[24] - fPointList3D[21],
             fPointList3D[26] - fPointList3D[23])
    p1Head = (fPointList3D[21], fPointList3D[23])
    p2Head = (fPointList3D[24], fPointList3D[26])
    hyHead = np.sqrt((p2Head[1] - oHead[1]) ** 2 +
                     (p1Head[0] - oHead[0]) ** 2)
    thetaHead = (np.arccos((p2Head[1] - oHead[1]) / hyHead) *
                 180 / np.pi - 120) * 3

    #pLeftShoulder = (fPointList[])

    # Compute camera matrices
    eye = [(width/2), height/2, width/4.5] # 5.5

    # Change target from center to [0, 0, 0].
    view_matrix = pyrr.matrix44.create_look_at(
        eye, [width/2, height/2, 0], np.asarray([0, 1, 0]))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix)

    proj_matrix = pyrr.matrix44.create_perspective_projection_matrix(
        cameraFOV, width/height, 0.1, 1000)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, proj_matrix)

    '---------------------'
    'Drawing Skybox'
    '---------------------'
    glDepthFunc(GL_LEQUAL)    # Change depth function so depth test passes when values are equal to depth buffer's content
    glUseProgram(shaderSkybox)

    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture)

    view_mat_without_translation = view_matrix.copy()
    view_mat_without_translation[3][:3] = [0,0,0]

    zoomedFOV = 90.0
    proj_matrix_zoomed_in = pyrr.matrix44.create_perspective_projection_matrix(
        zoomedFOV, width/height, 0.1, 1000)

    inverseViewProjection_mat = pyrr.matrix44.inverse(pyrr.matrix44.multiply(view_mat_without_translation, proj_matrix_zoomed_in))

    # shaderProgram_skybox["cubeMapTex"] = 1
    glUniform1i(cube_map_loc, 1)
    # shaderProgram_skybox["invViewProjectionMatrix"] = inverseViewProjection_mat
    glUniformMatrix4fv(inv_proj_loc, 1, GL_FALSE, inverseViewProjection_mat)

    glBindVertexArray(vao_quad)
    glDrawArrays(GL_TRIANGLES, 0, quad_n_vertices * 2)  # Draw the triangle
    glDepthFunc(GL_LESS)      # Set depth function back to default
    # Refresh the display to show what's been drawn
    pg.display.flip()


# Cleanup
glDeleteVertexArrays(1, [vao])
glDeleteBuffers(1, [vbo])
glDeleteProgram(shader)

pg.quit()   # Close the graphics window
quit()      # Exit the program