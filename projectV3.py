# Import necessary libraries
import pygame as pg
import pyrr.matrix44
from OpenGL.GL import *
import numpy as np
from PIL import Image
import shaderLoader
import guiV1
import time
import datetime
import cv2
# cvzone is a helpful Computer Vision library from Murtaza Hassan
# Link to the Github documentation for it is in the README
from cvzone.PoseModule import PoseDetector

#https://github.com/amengede/getIntoGameDev/blob/main/pyopengl%202022/12%20-%20Text/finished/main.py
# class Font:
#
#     def __init__(self):
#         # some parameters for fine tuning.
#         w = 55.55 / 1000.0
#         h = 63.88 / 1150.0
#         heightOffset = 8.5 / 1150.0
#         margin = 0.014
#
#         """
#             Letter: (left, top, width, height)
#         """
#         self.letterTexCoords = {
#             'A': (w, 1.0 - h, w - margin, h - margin),
#             'B': (3.0 * w, 1.0 - h, w - margin, h - margin),
#             'C': (5.0 * w, 1.0 - h, w - margin, h - margin),
#             'D': (7.0 * w, 1.0 - h, w - margin, h - margin),
#             'E': (9.0 * w, 1.0 - h, w - margin, h - margin),
#             'F': (11.0 * w, 1.0 - h, w - margin, h - margin),
#             'G': (13.0 * w, 1.0 - h, w - margin, h - margin),
#             'H': (15.0 * w, 1.0 - h, w - margin, h - margin),
#             'I': (17.0 * w, 1.0 - h, w - margin, h - margin),
#             'J': (w, 1.0 - 3.0 * h + heightOffset, w - margin, h - margin),
#             'K': (
#             3.0 * w, 1.0 - 3.0 * h + heightOffset, w - margin, h - margin),
#             'L': (
#             5.0 * w, 1.0 - 3.0 * h + heightOffset, w - margin, h - margin),
#             'M': (
#             7.0 * w, 1.0 - 3.0 * h + heightOffset, w - margin, h - margin),
#             'N': (
#             9.0 * w, 1.0 - 3.0 * h + heightOffset, w - margin, h - margin),
#             'O': (
#             11.0 * w, 1.0 - 3.0 * h + heightOffset, w - margin, h - margin),
#             'P': (
#             13.0 * w, 1.0 - 3.0 * h + heightOffset, w - margin, h - margin),
#             'Q': (
#             15.0 * w, 1.0 - 3.0 * h + heightOffset, w - margin, h - margin),
#             'R': (
#             17.0 * w, 1.0 - 3.0 * h + heightOffset, w - margin, h - margin),
#             'S': (w, 1.0 - 5.0 * h + 2 * heightOffset, w - margin, h - margin),
#             'T': (
#             3.0 * w, 1.0 - 5.0 * h + 2 * heightOffset, w - margin, h - margin),
#             'U': (
#             5.0 * w, 1.0 - 5.0 * h + 2 * heightOffset, w - margin, h - margin),
#             'V': (
#             7.0 * w, 1.0 - 5.0 * h + 2 * heightOffset, w - margin, h - margin),
#             'W': (
#             9.0 * w, 1.0 - 5.0 * h + 2 * heightOffset, w - margin, h - margin),
#             'X': (
#             11.0 * w, 1.0 - 5.0 * h + 2 * heightOffset, w - margin, h - margin),
#             'Y': (
#             13.0 * w, 1.0 - 5.0 * h + 2 * heightOffset, w - margin, h - margin),
#             'Z': (
#             15.0 * w, 1.0 - 5.0 * h + 2 * heightOffset, w - margin, h - margin),
#
#             'a': (w, 1.0 - 7.0 * h, w - margin, h - margin),
#             'b': (3.0 * w, 1.0 - 7.0 * h, w - margin, h - margin),
#             'c': (5.0 * w, 1.0 - 7.0 * h, w - margin, h - margin),
#             'd': (7.0 * w, 1.0 - 7.0 * h, w - margin, h - margin),
#             'e': (9.0 * w, 1.0 - 7.0 * h, w - margin, h - margin),
#             'f': (11.0 * w, 1.0 - 7.0 * h, w - margin, h - margin),
#             'g': (13.0 * w, 1.0 - 7.0 * h, w - margin, h - margin),
#             'h': (15.0 * w, 1.0 - 7.0 * h, w - margin, h - margin),
#             'i': (17.0 * w, 1.0 - 7.0 * h, w - margin, h - margin),
#             'j': (w, 1.0 - 9.0 * h + heightOffset, w - margin, h - margin),
#             'k': (
#             3.0 * w, 1.0 - 9.0 * h + heightOffset, w - margin, h - margin),
#             'l': (
#             5.0 * w, 1.0 - 9.0 * h + heightOffset, w - margin, h - margin),
#             'm': (
#             7.0 * w, 1.0 - 9.0 * h + heightOffset, w - margin, h - margin),
#             'n': (
#             9.0 * w, 1.0 - 9.0 * h + heightOffset, w - margin, h - margin),
#             'o': (
#             11.0 * w, 1.0 - 9.0 * h + heightOffset, w - margin, h - margin),
#             'p': (
#             13.0 * w, 1.0 - 9.0 * h + heightOffset, w - margin, h - margin),
#             'q': (
#             15.0 * w, 1.0 - 9.0 * h + heightOffset, w - margin, h - margin),
#             'r': (
#             17.0 * w, 1.0 - 9.0 * h + heightOffset, w - margin, h - margin),
#             's': (w, 1.0 - 11.0 * h + 2 * heightOffset, w - margin, h - margin),
#             't': (
#             3.0 * w, 1.0 - 11.0 * h + 2 * heightOffset, w - margin, h - margin),
#             'u': (
#             5.0 * w, 1.0 - 11.0 * h + 2 * heightOffset, w - margin, h - margin),
#             'v': (
#             7.0 * w, 1.0 - 11.0 * h + 2 * heightOffset, w - margin, h - margin),
#             'w': (
#             9.0 * w, 1.0 - 11.0 * h + 2 * heightOffset, w - margin, h - margin),
#             'x': (11.0 * w, 1.0 - 11.0 * h + 2 * heightOffset, w - margin,
#                   h - margin),
#             'y': (13.0 * w, 1.0 - 11.0 * h + 2 * heightOffset, w - margin,
#                   h - margin), 'z': (
#             15.0 * w, 1.0 - 11.0 * h + 2 * heightOffset, w - margin,
#             h - margin),
#
#             '0': (w, 1.0 - 13.0 * h, w - margin, h - margin),
#             '1': (3.0 * w, 1.0 - 13.0 * h, w - margin, h - margin),
#             '2': (5.0 * w, 1.0 - 13.0 * h, w - margin, h - margin),
#             '3': (7.0 * w, 1.0 - 13.0 * h, w - margin, h - margin),
#             '4': (9.0 * w, 1.0 - 13.0 * h, w - margin, h - margin),
#             '5': (11.0 * w, 1.0 - 13.0 * h, w - margin, h - margin),
#             '6': (13.0 * w, 1.0 - 13.0 * h, w - margin, h - margin),
#             '7': (15.0 * w, 1.0 - 13.0 * h, w - margin, h - margin),
#             '8': (17.0 * w, 1.0 - 13.0 * h, w - margin, h - margin),
#             '9': (w, 1.0 - 15.0 * h + heightOffset, w - margin, h - margin),
#
#             '.': (
#             3.0 * w, 1.0 - 15.0 * h + heightOffset, w - margin, h - margin),
#             ',': (
#             5.0 * w, 1.0 - 15.0 * h + heightOffset, w - margin, h - margin),
#             ';': (
#             7.0 * w, 1.0 - 15.0 * h + heightOffset, w - margin, h - margin),
#             ':': (
#             9.0 * w, 1.0 - 15.0 * h + heightOffset, w - margin, h - margin),
#             '$': (
#             11.0 * w, 1.0 - 15.0 * h + heightOffset, w - margin, h - margin),
#             '#': (
#             13.0 * w, 1.0 - 15.0 * h + heightOffset, w - margin, h - margin),
#             '\'': (
#             15.0 * w, 1.0 - 15.0 * h + heightOffset, w - margin, h - margin),
#             '!': (
#             17.0 * w, 1.0 - 15.0 * h + heightOffset, w - margin, h - margin),
#             '"': (w, 1.0 - 17.0 * h + 2 * heightOffset, w - margin, h - margin),
#             '/': (
#             3.0 * w, 1.0 - 17.0 * h + 2 * heightOffset, w - margin, h - margin),
#             '?': (
#             5.0 * w, 1.0 - 17.0 * h + 2 * heightOffset, w - margin, h - margin),
#             '%': (
#             7.0 * w, 1.0 - 17.0 * h + 2 * heightOffset, w - margin, h - margin),
#             '&': (
#             9.0 * w, 1.0 - 17.0 * h + 2 * heightOffset, w - margin, h - margin),
#             '(': (11.0 * w, 1.0 - 17.0 * h + 2 * heightOffset, w - margin,
#                   h - margin),
#             ')': (13.0 * w, 1.0 - 17.0 * h + 2 * heightOffset, w - margin,
#                   h - margin), '@': (
#             15.0 * w, 1.0 - 17.0 * h + 2 * heightOffset, w - margin, h - margin)
#         }
#
#         self.texture = glGenTextures(1)
#         glBindTexture(GL_TEXTURE_2D, self.texture)
#         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
#         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
#         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
#                         GL_NEAREST_MIPMAP_LINEAR)
#         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
#         with Image.open("gfx/Inconsolata.png", mode="r") as img:
#             image_width, image_height = img.size
#             img = img.convert("RGBA")
#             img_data = bytes(img.tobytes())
#             glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height,
#                          0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
#         glGenerateMipmap(GL_TEXTURE_2D)
#
#     def get_bounding_box(self, letter):
#         if letter in self.letterTexCoords:
#             return self.letterTexCoords[letter]
#         return None
#
#     def use(self):
#         glActiveTexture(GL_TEXTURE0)
#         glBindTexture(GL_TEXTURE_2D, self.texture)
#
#     def destroy(self):
#         glDeleteTextures(1, (self.texture,))
#
#
# class TextLine:
#
#     def __init__(self, initial_text, font, start_position, letter_size):
#
#         self.vao = glGenVertexArrays(1)
#         self.vbo = glGenBuffers(1)
#         self.start_position = start_position
#         self.letter_size = letter_size
#         self.build_text(initial_text, font)
#
#     def build_text(self, new_text, font):
#
#         self.vertices = []
#         self.vertex_count = 0
#
#         margin_adjustment = -0.5625
#
#         for i, letter in enumerate(new_text):
#
#             bounding_box = font.get_bounding_box(letter)
#             if bounding_box is None:
#                 continue
#
#             # top left
#             self.vertices.append(
#                 self.start_position[0] - self.letter_size[0] + (
#                         (2 + margin_adjustment) * i * self.letter_size[0])
#             )
#             self.vertices.append(self.start_position[1] + self.letter_size[1])
#             self.vertices.append(bounding_box[0] - bounding_box[2])
#             self.vertices.append(bounding_box[1] + bounding_box[3])
#             # top right
#             self.vertices.append(
#                 self.start_position[0] + self.letter_size[0] + (
#                         (2 + margin_adjustment) * i * self.letter_size[0])
#             )
#             self.vertices.append(self.start_position[1] + self.letter_size[1])
#             self.vertices.append(bounding_box[0] + bounding_box[2])
#             self.vertices.append(bounding_box[1] + bounding_box[3])
#             # bottom right
#             self.vertices.append(
#                 self.start_position[0] + self.letter_size[0] + (
#                         (2 + margin_adjustment) * i * self.letter_size[0])
#             )
#             self.vertices.append(self.start_position[1] - self.letter_size[1])
#             self.vertices.append(bounding_box[0] + bounding_box[2])
#             self.vertices.append(bounding_box[1] - bounding_box[3])
#
#             # bottom right
#             self.vertices.append(
#                 self.start_position[0] + self.letter_size[0] + (
#                         (2 + margin_adjustment) * i * self.letter_size[0])
#             )
#             self.vertices.append(self.start_position[1] - self.letter_size[1])
#             self.vertices.append(bounding_box[0] + bounding_box[2])
#             self.vertices.append(bounding_box[1] - bounding_box[3])
#             # bottom left
#             self.vertices.append(
#                 self.start_position[0] - self.letter_size[0] + (
#                         (2 + margin_adjustment) * i * self.letter_size[0])
#             )
#             self.vertices.append(self.start_position[1] - self.letter_size[1])
#             self.vertices.append(bounding_box[0] - bounding_box[2])
#             self.vertices.append(bounding_box[1] - bounding_box[3])
#             # top left
#             self.vertices.append(
#                 self.start_position[0] - self.letter_size[0] + (
#                         (2 + margin_adjustment) * i * self.letter_size[0])
#             )
#             self.vertices.append(self.start_position[1] + self.letter_size[1])
#             self.vertices.append(bounding_box[0] - bounding_box[2])
#             self.vertices.append(bounding_box[1] + bounding_box[3])
#
#             self.vertex_count += 6
#
#         self.vertices = np.array(self.vertices, dtype=np.float32)
#
#         glBindVertexArray(self.vao)
#         glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
#         glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices,
#                      GL_STATIC_DRAW)
#         offset = 0
#         # position
#         glEnableVertexAttribArray(0)
#         glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16,
#                               ctypes.c_void_p(offset))
#         offset += 8
#         # texture
#         glEnableVertexAttribArray(1)
#         glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16,
#                               ctypes.c_void_p(offset))
#
#     def draw(self) -> None:
#         """
#             Draw the text.
#         """
#
#         glBindVertexArray(self.vao)
#         glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
#
#     def destroy(self):
#         glDeleteVertexArrays(1, (self.vao,))
#         glDeleteBuffers(1, (self.vbo,))



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
# For storing the calibration data.
calibratedPose = False
calibList2D = [0]*66
calibList3D = [0]*99

# Run a loop to keep the program running
draw = True
while draw:
    # Quit out if need be
    for event in pg.event.get():
        if event.type == pg.QUIT:
            draw = False

    # Calibration timer
    if not calibratedPose:
        print("Please make sure your entire body is in frame!")
        seconds = 5
        while seconds > 0:
            timer = datetime.timedelta(seconds=seconds)
            print(str(timer)+"...")
            time.sleep(1)
            seconds -= 1
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
    # 66 elements
    fPointList = np.array(pointList2D, dtype="float32")
    # 99 elements
    fPointList3D = np.array(pointList3D, dtype="float32")

    if not calibratedPose:
        calibList2D = fPointList
        calibList3D = fPointList3D
        print("Calibration complete!")
        calibratedPose = True
    '---------------------'
    'Drawing'
    '---------------------'
    # # Get Font
    # font = Font()
    # fps_label = TextLine("FPS: ", font, (-0.9, 0.9), (0.05, 0.05))
    #
    # # From amengede's getIntoGameDev GitHub.
    # def update_fps(new_fps: int) -> None:
    #     """
    #         Rebuild the fps label.
    #     """
    #
    #     fps_label.build_text(f"FPS: {new_fps}", font)
    # update_fps(100)

    # Draw Points
    glUseProgram(shader)
    glBindVertexArray(vao)

    # Using fPointList, aka the list of 2D points, to draw the points.
    glBufferData(GL_ARRAY_BUFFER, size=calibList2D.nbytes,
                 data=calibList2D, usage=GL_STATIC_DRAW) # fPOintnList

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