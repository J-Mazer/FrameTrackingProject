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
texture1 = load_texture("objects/rayman.png")

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
position_loc1 = glGetAttribLocation(rayman_shader, "position")
glVertexAttribPointer(index=position_loc1, size=size_position, type=GL_FLOAT,
                      normalized=GL_FALSE, stride=stride,
                      pointer=ctypes.c_void_p(offset_position))
glEnableVertexAttribArray(position_loc1)

# Configure normal
normal_loc1 = glGetAttribLocation(rayman_shader, "normal")
glVertexAttribPointer(normal_loc1, size=size_normal, type=GL_FLOAT,
                      normalized=GL_FALSE, stride=stride,
                      pointer=ctypes.c_void_p(offset_normal))
glEnableVertexAttribArray(normal_loc1)

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

""
"Object 2"
""
obj2 = ObjLoader("objects/raymanTorso.obj")

vertices2 = np.array(obj2.vertices, dtype="float32")
center2 = obj2.center
dia2 = obj2.dia

float_byte_size = vertices2[0].nbytes

n_vertices2 = len(vertices2) // (size_position + size_texture + size_normal)

# Create the VAO
vao2 = glGenVertexArrays(1)
glBindVertexArray(vao2)

# Create the VBO
vbo2 = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo2)
glBufferData(GL_ARRAY_BUFFER, size=vertices2.nbytes, data=vertices2,
             usage=GL_STATIC_DRAW)

# Configure position
position_loc2 = glGetAttribLocation(rayman_shader, "position")
glVertexAttribPointer(index=position_loc2, size=size_position, type=GL_FLOAT,
                      normalized=GL_FALSE, stride=stride,
                      pointer=ctypes.c_void_p(offset_position))
glEnableVertexAttribArray(position_loc2)

# Configure normal
normal_loc2 = glGetAttribLocation(rayman_shader, "normal")
glVertexAttribPointer(normal_loc2, size=size_normal, type=GL_FLOAT,
                      normalized=GL_FALSE, stride=stride,
                      pointer=ctypes.c_void_p(offset_normal))
glEnableVertexAttribArray(normal_loc2)

# Get variables
scale_loc2 = glGetUniformLocation(rayman_shader, "scale")
# Setting scale to 2 / dia
scale2 = .75 / dia2
glUniform1f(scale_loc2, scale2)

center_loc2 = glGetUniformLocation(rayman_shader, "center")
# Setting center to the model's center.
glUniform3f(center_loc2, center2[0], center2[1], center2[2])

aspect_loc2 = glGetUniformLocation(rayman_shader, "aspect")
# Setting aspect to width / height
aspect = width / height
glUniform1f(aspect_loc2, aspect)

# Model matrix
modelM2 = pyrr.matrix44.create_identity()

# Assignment 4:
transM2 = pyrr.matrix44.create_from_translation([
    -center2[0], -center2[1], -center2[2]])

# Get the matrix locations.
model_loc2 = glGetUniformLocation(rayman_shader, "model_matrix")
view_loc2 = glGetUniformLocation(rayman_shader, "view_matrix")
proj_loc2 = glGetUniformLocation(rayman_shader, "proj_matrix")

""
"Object 3"
""
obj3 = ObjLoader("objects/raymanLeftHand.obj")

vertices3 = np.array(obj3.vertices, dtype="float32")
center3 = obj3.center
dia3 = obj3.dia

float_byte_size3 = vertices3[0].nbytes

n_vertices3 = len(vertices3) // (size_position + size_texture + size_normal)

# Create the VAO
vao3 = glGenVertexArrays(1)
glBindVertexArray(vao3)

# Create the VBO
vbo3 = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo3)
glBufferData(GL_ARRAY_BUFFER, size=vertices3.nbytes, data=vertices3,
             usage=GL_STATIC_DRAW)

# Configure position
position_loc3 = glGetAttribLocation(rayman_shader, "position")
glVertexAttribPointer(index=position_loc3, size=size_position, type=GL_FLOAT,
                      normalized=GL_FALSE, stride=stride,
                      pointer=ctypes.c_void_p(offset_position))
glEnableVertexAttribArray(position_loc3)

# Configure normal
normal_loc3 = glGetAttribLocation(rayman_shader, "normal")
glVertexAttribPointer(normal_loc3, size=size_normal, type=GL_FLOAT,
                      normalized=GL_FALSE, stride=stride,
                      pointer=ctypes.c_void_p(offset_normal))
glEnableVertexAttribArray(normal_loc3)

# Get variables
scale_loc3 = glGetUniformLocation(rayman_shader, "scale")
# Setting scale to 2 / dia
scale3 = .75 / dia3
glUniform1f(scale_loc3, scale3)

center_loc3 = glGetUniformLocation(rayman_shader, "center")
# Setting center to the model's center.
glUniform3f(center_loc3, center3[0], center3[1], center3[2])

aspect_loc3 = glGetUniformLocation(rayman_shader, "aspect")
# Setting aspect to width / height
aspect = width / height
glUniform1f(aspect_loc3, aspect)

# Model matrix
modelM3 = pyrr.matrix44.create_identity()

# Assignment 4:
transM3 = pyrr.matrix44.create_from_translation([
    -center3[0], -center3[1], -center3[2]])

# Get the matrix locations.
model_loc3 = glGetUniformLocation(rayman_shader, "model_matrix")
view_loc3 = glGetUniformLocation(rayman_shader, "view_matrix")
proj_loc3 = glGetUniformLocation(rayman_shader, "proj_matrix")


""
"Object 4"
""
obj4 = ObjLoader("objects/raymanRightHand.obj")

vertices4 = np.array(obj4.vertices, dtype="float32")
center4 = obj4.center
dia4 = obj4.dia

float_byte_size4 = vertices4[0].nbytes

n_vertices4 = len(vertices4) // (size_position + size_texture + size_normal)

# Create the VAO
vao4 = glGenVertexArrays(1)
glBindVertexArray(vao4)

# Create the VBO
vbo4 = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo4)
glBufferData(GL_ARRAY_BUFFER, size=vertices4.nbytes, data=vertices4,
             usage=GL_STATIC_DRAW)

# Configure position
position_loc4 = glGetAttribLocation(rayman_shader, "position")
glVertexAttribPointer(index=position_loc4, size=size_position, type=GL_FLOAT,
                      normalized=GL_FALSE, stride=stride,
                      pointer=ctypes.c_void_p(offset_position))
glEnableVertexAttribArray(position_loc4)

# Configure normal
normal_loc4 = glGetAttribLocation(rayman_shader, "normal")
glVertexAttribPointer(normal_loc4, size=size_normal, type=GL_FLOAT,
                      normalized=GL_FALSE, stride=stride,
                      pointer=ctypes.c_void_p(offset_normal))
glEnableVertexAttribArray(normal_loc4)

# Get variables
scale_loc4 = glGetUniformLocation(rayman_shader, "scale")
# Setting scale to 2 / dia
scale4 = .75 / dia4
glUniform1f(scale_loc4, scale4)

center_loc4 = glGetUniformLocation(rayman_shader, "center")
# Setting center to the model's center.
glUniform3f(center_loc4, center4[0], center4[1], center4[2])

aspect_loc4 = glGetUniformLocation(rayman_shader, "aspect")
# Setting aspect to width / height
aspect = width / height
glUniform1f(aspect_loc4, aspect)

# Model matrix
modelM4 = pyrr.matrix44.create_identity()

# Assignment 4:
transM4 = pyrr.matrix44.create_from_translation([
    -center4[0], -center4[1], -center4[2]])

# Get the matrix locations.
model_loc4 = glGetUniformLocation(rayman_shader, "model_matrix")
view_loc4 = glGetUniformLocation(rayman_shader, "view_matrix")
proj_loc4 = glGetUniformLocation(rayman_shader, "proj_matrix")

""
"Object 5"
""
obj5 = ObjLoader("objects/raymanLeftFoot.obj")

vertices5 = np.array(obj5.vertices, dtype="float32")
center5 = obj5.center
dia5 = obj5.dia

float_byte_size5 = vertices5[0].nbytes

n_vertices5 = len(vertices5) // (size_position + size_texture + size_normal)

# Create the VAO
vao5 = glGenVertexArrays(1)
glBindVertexArray(vao5)

# Create the VBO
vbo5 = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo5)
glBufferData(GL_ARRAY_BUFFER, size=vertices5.nbytes, data=vertices5,
             usage=GL_STATIC_DRAW)

# Configure position
position_loc5 = glGetAttribLocation(rayman_shader, "position")
glVertexAttribPointer(index=position_loc5, size=size_position, type=GL_FLOAT,
                      normalized=GL_FALSE, stride=stride,
                      pointer=ctypes.c_void_p(offset_position))
glEnableVertexAttribArray(position_loc5)

# Configure normal
normal_loc5 = glGetAttribLocation(rayman_shader, "normal")
glVertexAttribPointer(normal_loc5, size=size_normal, type=GL_FLOAT,
                      normalized=GL_FALSE, stride=stride,
                      pointer=ctypes.c_void_p(offset_normal))
glEnableVertexAttribArray(normal_loc5)

# Get variables
scale_loc5 = glGetUniformLocation(rayman_shader, "scale")
# Setting scale to 2 / dia
scale5 = .75 / dia5
glUniform1f(scale_loc5, scale5)

center_loc5 = glGetUniformLocation(rayman_shader, "center")
# Setting center to the model's center.
glUniform3f(center_loc5, center5[0], center5[1], center5[2])

aspect_loc5 = glGetUniformLocation(rayman_shader, "aspect")
# Setting aspect to width / height
aspect = width / height
glUniform1f(aspect_loc5, aspect)

# Model matrix
modelM5 = pyrr.matrix44.create_identity()

# Assignment 4:
transM5 = pyrr.matrix44.create_from_translation([
    -center5[0], -center5[1], -center5[2]])

# Get the matrix locations.
model_loc5 = glGetUniformLocation(rayman_shader, "model_matrix")
view_loc5 = glGetUniformLocation(rayman_shader, "view_matrix")
proj_loc5 = glGetUniformLocation(rayman_shader, "proj_matrix")


""
"Object 6"
""
obj6 = ObjLoader("objects/raymanRightFoot.obj")

vertices6 = np.array(obj6.vertices, dtype="float32")
center6 = obj6.center
dia6 = obj6.dia

float_byte_size6 = vertices6[0].nbytes

n_vertices6 = len(vertices6) // (size_position + size_texture + size_normal)

# Create the VAO
vao6 = glGenVertexArrays(1)
glBindVertexArray(vao6)

# Create the VBO
vbo6 = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo6)
glBufferData(GL_ARRAY_BUFFER, size=vertices6.nbytes, data=vertices6,
             usage=GL_STATIC_DRAW)

# Configure position
position_loc6 = glGetAttribLocation(rayman_shader, "position")
glVertexAttribPointer(index=position_loc6, size=size_position, type=GL_FLOAT,
                      normalized=GL_FALSE, stride=stride,
                      pointer=ctypes.c_void_p(offset_position))
glEnableVertexAttribArray(position_loc6)

# Configure normal
normal_loc6 = glGetAttribLocation(rayman_shader, "normal")
glVertexAttribPointer(normal_loc6, size=size_normal, type=GL_FLOAT,
                      normalized=GL_FALSE, stride=stride,
                      pointer=ctypes.c_void_p(offset_normal))
glEnableVertexAttribArray(normal_loc6)

# Get variables
scale_loc6 = glGetUniformLocation(rayman_shader, "scale")
# Setting scale to 2 / dia
scale6 = .75 / dia6
glUniform1f(scale_loc6, scale6)

center_loc6 = glGetUniformLocation(rayman_shader, "center")
# Setting center to the model's center.
glUniform3f(center_loc6, center6[0], center6[1], center6[2])

aspect_loc6 = glGetUniformLocation(rayman_shader, "aspect")
# Setting aspect to width / height
aspect = width / height
glUniform1f(aspect_loc6, aspect)

# Model matrix
modelM6 = pyrr.matrix44.create_identity()

# Assignment 4:
transM6 = pyrr.matrix44.create_from_translation([
    -center6[0], -center6[1], -center6[2]])

# Get the matrix locations.
model_loc6 = glGetUniformLocation(rayman_shader, "model_matrix")
view_loc6 = glGetUniformLocation(rayman_shader, "view_matrix")
proj_loc6 = glGetUniformLocation(rayman_shader, "proj_matrix")

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

    '---------------------'
    'Drawing Points'
    '---------------------'
    # # Draw Points
    # glUseProgram(shader)
    # glBindVertexArray(vao)
    #
    # # Using fPointList, aka the list of 2D points, to draw the points.
    # glBufferData(GL_ARRAY_BUFFER, size=calibList2D.nbytes,
    #              data=calibList2D, usage=GL_STATIC_DRAW) # calibList2D
    #
    #
    # glDrawArrays(GL_POINTS, 0, len(fPointList) // 2)

    '==============='
    'Drawing Rayman'
    '==============='

    # obj1 is rayman's head
    # obj2 is rayman's torso
    # obj3 is rayman's left hand

    '---------------'
    'Drawing Raymans Head'
    '---------------'
    # Draw Rayman
    glUseProgram(rayman_shader)

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

    '---------------'
    'Drawing Raymans Body'
    '---------------'

    scale2 = (.65 / dia2) * calibConv

    rPosC2 = -center2
    rPosC2[0] = -(rPosC2[0] + (15 * (((fPointList[46] + fPointList[48]) / 2) /
                                    float(width))) - 8)  # 10, 5
    rPosC2[1] = (rPosC2[1] + (20 * (((fPointList[47] + fPointList[49]) / 2) /
                    float(height))) - 9) # 10
    rPosC2[2] = rPosC2[2] - 5
    # Compute matrices
    scaleM2 = pyrr.matrix44.create_from_scale(np.array([scale2,
                                                       scale2,
                                                       scale2]))
    rotateZM2 = pyrr.matrix44.create_from_z_rotation(np.deg2rad(rotationZ))
    rotateYM2 = pyrr.matrix44.create_from_y_rotation(np.deg2rad(rotationY))
    rotateXM2 = pyrr.matrix44.create_from_x_rotation(np.deg2rad(rotationX))


    transM2 = pyrr.matrix44.create_from_translation(rPosC2)

    model_matrix2 = pyrr.matrix44.multiply(transM2,
                pyrr.matrix44.multiply(rotateXM2,
                pyrr.matrix44.multiply(rotateYM2,
                pyrr.matrix44.multiply(rotateZM2,
                                        scaleM2))))
    glUniformMatrix4fv(model_loc2, 1, GL_FALSE, model_matrix2)
    glBindVertexArray(vao2)
    glDrawArrays(GL_TRIANGLES, 0, n_vertices2)

    '---------------'
    'Drawing Raymans Left Hand'
    '---------------'

    scale3 = (.5 / dia3) * calibConv

    rPosC3 = -center3
    rPosC3[0] = -(rPosC3[0] + (20 * (fPointList[32] / float(width))) - 10)
    rPosC3[1] = (rPosC3[1] + (22 * (fPointList[33] / float(height))) - 11)

    # Compute matrices
    scaleM3 = pyrr.matrix44.create_from_scale(np.array([scale3,
                                                        scale3,
                                                        scale3]))
    rotateZM3 = pyrr.matrix44.create_from_z_rotation(np.deg2rad(rotationZ))
    rotateYM3 = pyrr.matrix44.create_from_y_rotation(np.deg2rad(rotationY))
    rotateXM3 = pyrr.matrix44.create_from_x_rotation(np.deg2rad(rotationX))

    transM3 = pyrr.matrix44.create_from_translation(rPosC3)

    model_matrix3 = pyrr.matrix44.multiply(transM3,
                    pyrr.matrix44.multiply(rotateXM3,
                    pyrr.matrix44.multiply(rotateYM3,
                    pyrr.matrix44.multiply(rotateZM3,
                                           scaleM3))))
    glUniformMatrix4fv(model_loc3, 1, GL_FALSE, model_matrix3)
    glBindVertexArray(vao3)
    glDrawArrays(GL_TRIANGLES, 0, n_vertices3)

    '---------------'
    'Drawing Raymans Right Hand'
    '---------------'

    scale4 = (.5 / dia4) * calibConv

    rPosC4 = -center4
    rPosC4[0] = -(rPosC4[0] + (20 * (fPointList[30] / float(width))) - 10)
    rPosC4[1] = (rPosC4[1] + (22 * (fPointList[31] / float(height))) - 11)

    # Compute matrices
    scaleM4 = pyrr.matrix44.create_from_scale(np.array([scale4,
                                                        scale4,
                                                        scale4]))
    rotateZM4 = pyrr.matrix44.create_from_z_rotation(np.deg2rad(rotationZ))
    rotateYM4 = pyrr.matrix44.create_from_y_rotation(np.deg2rad(rotationY))
    rotateXM4 = pyrr.matrix44.create_from_x_rotation(np.deg2rad(rotationX))

    transM4 = pyrr.matrix44.create_from_translation(rPosC4)

    model_matrix4 = pyrr.matrix44.multiply(transM4,
                    pyrr.matrix44.multiply(rotateXM4,
                    pyrr.matrix44.multiply(rotateYM4,
                    pyrr.matrix44.multiply(rotateZM4,
                                           scaleM4))))
    glUniformMatrix4fv(model_loc4, 1, GL_FALSE, model_matrix4)
    glBindVertexArray(vao4)
    glDrawArrays(GL_TRIANGLES, 0, n_vertices4)

    '---------------'
    'Drawing Raymans Left Foot'
    '---------------'

    scale5 = (.5 / dia5) * calibConv

    rPosC5 = -center5
    rPosC5[0] = -(rPosC5[0] + (20 * (fPointList[56] / float(width))) - 10)
    rPosC5[1] = (rPosC5[1] + (22 * (fPointList[57] / float(height))) - 11)

    # Compute matrices
    scaleM5 = pyrr.matrix44.create_from_scale(np.array([scale5,
                                                        scale5,
                                                        scale5]))
    rotateZM5 = pyrr.matrix44.create_from_z_rotation(np.deg2rad(rotationZ))
    rotateYM5 = pyrr.matrix44.create_from_y_rotation(np.deg2rad(rotationY))
    rotateXM5 = pyrr.matrix44.create_from_x_rotation(np.deg2rad(rotationX))

    transM5 = pyrr.matrix44.create_from_translation(rPosC5)

    model_matrix5 = pyrr.matrix44.multiply(transM5,
                    pyrr.matrix44.multiply(rotateXM5,
                    pyrr.matrix44.multiply(rotateYM5,
                    pyrr.matrix44.multiply(rotateZM5,
                                           scaleM5))))
    glUniformMatrix4fv(model_loc5, 1, GL_FALSE, model_matrix5)
    glBindVertexArray(vao5)
    glDrawArrays(GL_TRIANGLES, 0, n_vertices5)

    '---------------'
    'Drawing Raymans Right Foot'
    '---------------'

    scale6 = (.5 / dia6) * calibConv

    rPosC6 = -center6
    rPosC6[0] = -(rPosC6[0] + (20 * (fPointList[54] / float(width))) - 10)
    rPosC6[1] = (rPosC6[1] + (22 * (fPointList[55] / float(height))) - 11)

    # Compute matrices
    scaleM6 = pyrr.matrix44.create_from_scale(np.array([scale6,
                                                        scale6,
                                                        scale6]))
    rotateZM6 = pyrr.matrix44.create_from_z_rotation(np.deg2rad(rotationZ))
    rotateYM6 = pyrr.matrix44.create_from_y_rotation(np.deg2rad(rotationY))
    rotateXM6 = pyrr.matrix44.create_from_x_rotation(np.deg2rad(rotationX))

    transM6 = pyrr.matrix44.create_from_translation(rPosC6)

    model_matrix6 = pyrr.matrix44.multiply(transM6,
                    pyrr.matrix44.multiply(rotateXM6,
                    pyrr.matrix44.multiply(rotateYM6,
                    pyrr.matrix44.multiply(rotateZM6,
                                           scaleM6))))
    glUniformMatrix4fv(model_loc6, 1, GL_FALSE, model_matrix6)
    glBindVertexArray(vao6)
    glDrawArrays(GL_TRIANGLES, 0, n_vertices6)

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