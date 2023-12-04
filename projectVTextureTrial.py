# Import necessary libraries
import pygame as pg
import pyrr.matrix44
from OpenGL.GL import *
import numpy as np
import os
import shaderLoader
import guiV3
import time
from objLoaderV2 import ObjLoader
import cv2
# cvzone is a helpful Computer Vision library from Murtaza Hassan
# Link to the Github documentation for it is in the README
from cvzone.PoseModule import PoseDetector

'====================='
'SET UP CUBEMAP'
'====================='

def load_texture(image_path):
    texture_surface = pg.image.load(image_path)
    texture_data = pg.image.tostring(texture_surface, "RGBA", 1)
    width, height = texture_surface.get_size()

    # Step 2: Generate a texture ID
    texture_id = glGenTextures(1)

    # Step 3: Bind the texture
    glBindTexture(GL_TEXTURE_2D, texture_id)

    # Step 4: Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Step 5: Upload texture data to the GPU
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

    # Step 6: Generate mipmaps (optional)
    glGenerateMipmap(GL_TEXTURE_2D)

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
glClearColor(0.0941, 0.7490, 0.0235, 1.0)

# Enable depth testing and point size
glEnable(GL_DEPTH_TEST)
glEnable(GL_PROGRAM_POINT_SIZE)

# Load in the shader
shader = shaderLoader.compile_shader(
    "shaders/vert.glsl", "shaders/frag.glsl")
glUseProgram(shader)

shaderSkybox = shaderLoader.compile_shader("shaders/vert_skybox.glsl",
                           "shaders/frag_skybox.glsl")

rayman_shader = shaderLoader.compile_shader("shaders/raymanVS.glsl",
                                     "shaders/raymanFS.glsl")

glUseProgram(rayman_shader)

""
"Object 1"
""
obj1 = ObjLoader("objects/raymanHead.obj")
texture1 = load_texture("objects/raymanModel.png")

vertices1 = np.array(obj1.vertices, dtype="float32")
center1 = obj1.center
dia1 = obj1.dia

# Sizes
size_position = 3
size_texture = 2
size_normal = 3

float_byte_size = vertices1[0].nbytes

# Byte values
stride = float_byte_size * (size_position + size_texture + size_normal)
offset_position = 0
offset_texture = float_byte_size * size_position
offset_normal = float_byte_size * (size_position + size_texture)

n_vertices1 = len(vertices1) // (size_position + size_texture + size_normal)

# Create the VAO
vao1 = glGenVertexArrays(1)
glBindVertexArray(vao1)

# Create the VBO
vbo1 = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo1)
glBufferData(GL_ARRAY_BUFFER, size=vertices1.nbytes, data=vertices1,
             usage=GL_STATIC_DRAW)

# Configure position
glVertexAttribPointer(index=0, size=size_position, type=GL_FLOAT,
                      normalized=GL_FALSE, stride=stride,
                      pointer=ctypes.c_void_p(offset_position))
glEnableVertexAttribArray(0)

# Configure normal
glVertexAttribPointer(1, size=size_normal, type=GL_FLOAT,
                      normalized=GL_FALSE, stride=stride,
                      pointer=ctypes.c_void_p(offset_normal))
glEnableVertexAttribArray(1)

glVertexAttribPointer(2, size=size_texture, type=GL_FLOAT,
                      normalized=GL_FALSE, stride=stride,
                      pointer=ctypes.c_void_p(offset_texture))
glEnableVertexAttribArray(2)

# Get variables
scale_loc1 = glGetUniformLocation(rayman_shader, "scale")
# Setting scale to 2 / dia
scale1 = .75 / dia1
glUniform1f(scale_loc1, scale1)

center_loc1 = glGetUniformLocation(rayman_shader, "center")
# Setting center to the model's center.
glUniform3f(center_loc1, center1[0], center1[1], center1[2])

aspect_loc1 = glGetUniformLocation(rayman_shader, "aspect")
# Setting aspect to width / height
aspect = width / height
glUniform1f(aspect_loc1, aspect)

# Model matrix
modelM1 = pyrr.matrix44.create_identity()

# Assignment 4:
transM1 = pyrr.matrix44.create_from_translation([
    -center1[0], -center1[1], -center1[2]])

# Get the matrix locations.
model_loc1 = glGetUniformLocation(rayman_shader, "model_matrix")
view_loc1 = glGetUniformLocation(rayman_shader, "view_matrix")
proj_loc1 = glGetUniformLocation(rayman_shader, "proj_matrix")

'====================='
'SET UP SKYBOX STUFFS'
'====================='
# Ask the user if they want a green screen or the environment.
drawingSkybox = input("Do you want to capture the environment? Yes or "
                      "No?\n")
if drawingSkybox == "Yes":
    drawingSkybox = True
else:
    drawingSkybox = False
print("Capturing images shortly...")
time.sleep(2)
#drawingSkybox = 0

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

glUniform1i(glGetUniformLocation(rayman_shader, "tex2D"), 0)

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
cameraFOV = 130.0
lmString3D = ''
lmString2D = ''
# For storing the calibration data.
calibratedPose = False
takenPicture = False
startedTimer = False
startedTimerPicture = False
inFrame = False
notInFrame = True

timer = time.time_ns()
second = 1

calibList2D = np.array([0]*66, dtype="float32")
calibList3D = np.array([0]*99, dtype="float32")

calibratedHeadDistance = 0
calibratedBodyDistance = 0

scaleConv = 0

headTurnDir = [width/2, width/2]
headTurning = 0
bodyTurnDir = [width/2, width/2]
bodyTurning = 0

rotationX = 0
rotationY = 180
rotationZ = 0

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
    if drawingSkybox:
        # Check if frame is empty (no human detected)
        frameIsEmpty = len(lmList) == 0

        # Taking the initial background picture:
        if not takenPicture:
            backgroundImg = readImg
            # Timer to take picture when frame is empty
            if not startedTimerPicture and not takenPicture:
                oldTimer = timer
                second = 1
                print("Initializing actor's environment in...")
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
                    cubemap_paths = [
                        "left.png",
                        "left.png",
                        "left.png",
                        "left.png",
                        "left.png",
                        "left.png"
                    ]
                    # print("Initializing image...")
                    # time.sleep(3)
                    # print("Image intialized!")
                    cubemap_texture = create_cubemap_texture(cubemap_paths)

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
            cutoffTime = 10
            timer = time.time_ns()
            deltaTime = (timer - oldTimer) / 1e9
            if deltaTime >= second and second <= cutoffTime:
                print("0:00:0"+str(cutoffTime+1-second)+"...")
                second += 1
            if second > cutoffTime:
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
            calibratedBodyDistance = abs(calibList2D[24] - calibList2D[22])
            #scaleConv = scaleR - (calibratedBodyDistance / float(width * 1.5))
            print("Calibration complete!")
            calibratedPose = True

    #prevents from crashing if no data points are found
    if len(fPointList) < 66:
        continue


    # Shift all x coordinates over by 80.
    for i in range(len(fPointList)):
        if i % 2 == 0:
            fPointList[i] -= 80
    for i in range(len(fPointList3D)):
        if i % 3 == 0:
            fPointList3D[i] -= 80


    '---------------'
    'Head Angle'
    '---------------'
    #print(fPointList[18], fPointList[20])
    #print(fPointList[22], fPointList[24])
    # print(fPointList3D[33], fPointList3D[34], fPointList3D[35], "|",
    #       calibList3D[35])
    # 18 19, 20 21 for x, y of right mouth, left mouth respectively
    oldHeadAngle = headTurnDir
    headTurnDir = (fPointList[18], fPointList[20])
    oldHeadTurning = headTurning
    headTurning = np.sqrt((fPointList[20] - fPointList[18]) ** 2 +
                          (fPointList[21] - fPointList[19]) ** 2)
    #print(oldHeadAngle, headTurnDir)
    #print(abs(headTurning - oldHeadTurning))
    if abs(headTurning - oldHeadTurning) > 1.5:
        if (headTurnDir[0] < oldHeadAngle[0] - 3 and headTurnDir[1] >
            oldHeadAngle[1] + 3):
            print("Turning Head Right!")
        elif (headTurnDir[0] > oldHeadAngle[0] + 3 and headTurnDir[1] <
              oldHeadAngle[1] - 3):
            print("Turning Head Left!")

    '---------------'
    'Body Angle'
    '---------------'
    # 22 23, 24 25 for x, y of right shoulder, left shoulder respectively
    calibratedBodyDistance = abs(calibList2D[24] - calibList2D[22])

    oldBodyAngle = bodyTurnDir
    bodyTurnDir = (fPointList3D[35], fPointList3D[38])
    oldBodyTurning = bodyTurning
    bodyTurning = np.sqrt((fPointList[24] - fPointList[22]) ** 2 +
                          (fPointList[25] - fPointList[23]) ** 2)
    #print(oldHeadAngle, headTurnDir)
    #print(abs(headTurning - oldHeadTurning))
    if abs(bodyTurning - oldBodyTurning) > 1.5:
        if (bodyTurnDir[0] - oldBodyAngle[0] > 0 and bodyTurnDir[1] -
            oldBodyAngle[1] < 0):
            #print("Turning Body Left!")
            #rotationY += 3
            x = 1
        if (bodyTurnDir[0] - oldBodyAngle[0] < 0 and bodyTurnDir[1] -
            oldBodyAngle[1] > 0):
            #print("Turning Body Right!")
            #rotationY -= 3
            x = 1

    bodyDistance = abs(fPointList[24] - fPointList[22])

    '==============='
    'Drawing'
    '==============='
    # Clear color buffer and depth buffer before drawing each frame
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader)

    '---------------'
    'Camera'
    '---------------'
    if bodyDistance == 0 or calibratedBodyDistance == 0:
        calibConv = 1
    else:
        calibConv = bodyDistance / calibratedBodyDistance
    # Compute camera matrices
    eye = [(width/2), height/2, 10 * width/(4.45)] # 4.5

    # Change target from center to [0, 0, 0].
    view_matrix = pyrr.matrix44.create_look_at(
        eye, [width/2, height/2, 0], np.asarray([0, 1, 0]))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix)

    proj_matrix = pyrr.matrix44.create_perspective_projection_matrix(
        cameraFOV, width/height, 0.1, 1000)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, proj_matrix)

    '---------------'
    'Drawing Raymans Head'
    '---------------'
    # Draw Rayman
    glUseProgram(rayman_shader)

    glActiveTexture(GL_TEXTURE0)
    # if texture1 is not None:
    #     # Use the texture
    #     glBindTexture(GL_TEXTURE_2D, texture1)
    # else:
    #     print("Error loading texture.")

    scale1 = (.75 / dia1) * calibConv
    # Compute matrices
    rPosC = -center1
    rPosC[0] = -(rPosC[0] + (22 * (fPointList[0] / float(width))) - 11)
    rPosC[1] = (rPosC[1] + (22 * (fPointList[1] / float(height))) - 11) #20, 10
    #rPosC[2] = -(rPosC[2] + 5 * (1 - calibConv))

    scaleM1 = pyrr.matrix44.create_from_scale(np.array([scale1,
                                                       scale1,
                                                       scale1]))
    rotateZM1 = pyrr.matrix44.create_from_z_rotation(np.deg2rad(rotationZ))
    rotateYM1 = pyrr.matrix44.create_from_y_rotation(np.deg2rad(rotationY))
    rotateXM1 = pyrr.matrix44.create_from_x_rotation(np.deg2rad(rotationX))


    transM1 = pyrr.matrix44.create_from_translation(rPosC)

    model_matrix1 = pyrr.matrix44.multiply(transM1,
                pyrr.matrix44.multiply(rotateXM1,
                pyrr.matrix44.multiply(rotateYM1,
                pyrr.matrix44.multiply(rotateZM1,
                                        scaleM1))))
    glUniformMatrix4fv(model_loc1, 1, GL_FALSE, model_matrix1)
    glBindVertexArray(vao1)

    glDrawArrays(GL_TRIANGLES, 0, n_vertices1)

    '---------------------'
    'Drawing Skybox'
    '---------------------'
    if not drawingSkybox:
        pg.display.flip()
        continue

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