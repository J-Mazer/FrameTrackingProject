# Import necessary libraries
import pygame as pg
import pyrr.matrix44
from OpenGL.GL import *
import numpy as np

from objLoaderV3 import ObjLoader
import shaderLoader
import guiV1



# Initialize pygame
pg.init()

# Set up OpenGL context version
pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)


# Create a window for graphics using OpenGL
width = 708
height = 708
pg.display.set_mode((width, height), pg.OPENGL | pg.DOUBLEBUF)

# Background color
glClearColor(0.3, 0.4, 0.5, 1.0)

# Enable depth testing and point size
glEnable(GL_DEPTH_TEST)
glEnable(GL_PROGRAM_POINT_SIZE)

# Load in the shader
shader = shaderLoader.compile_shader(
    "shaders/vert.glsl", "shaders/frag.glsl")
glUseProgram(shader)

# Load in the points
pointList = []
# [[Line1], [Line2], ..., [LineX]]
# [x0, y0, z0, x1, y1, z1, ..., x33, y33, z32]
frameList = []
with open("Animation2D.txt", 'r') as f:
    line = "~~~"
    while line != "":
        line = f.readline()
        startI = 0
        endI = startI
        if line != "":
            while endI < len(line) and line[endI] != ',':
                endI += 1
                if endI < len(line) and line[endI] == ',':
                    pointList.append(float(line[startI:endI]))
                    startI = endI + 1
                    endI = startI
        if pointList != []:
            frameList.append(pointList)
        pointList = []
frameList = np.array(frameList, dtype="float32")

stride = frameList[0][0].nbytes * 2 # x, y, z

# Generate the VAO
vao = glGenVertexArrays(1)
glBindVertexArray(vao)

# Create the VBO
vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
# glBufferData(GL_ARRAY_BUFFER, size=frameList.nbytes,
#              data=frameList, usage=GL_STATIC_DRAW)

# Todo: Part 4: Configure vertex attributes using the variables defined in Part 1

# Configure position
position_loc = glGetAttribLocation(shader, "position")
glVertexAttribPointer(index=position_loc, size=2, type=GL_FLOAT,
                      normalized=GL_FALSE, stride=stride,
                      pointer=ctypes.c_void_p(0))
glEnableVertexAttribArray(position_loc)

cameraFOV = 130.0

# Get the matrix locations.
view_loc = glGetUniformLocation(shader, "view_matrix")
proj_loc = glGetUniformLocation(shader, "proj_matrix")

# Create Slider GUI
gui = guiV1.SimpleGUI("Camera GUI")

fovSlider = gui.add_slider(
    "Camera FOV Slider", 30, 360, cameraFOV)

# Todo: Part 6: Do the final rendering. In the rendering loop, do the following:
    # - Clear the color buffer and depth buffer before drawing each frame using glClear()
    # - Use the shader program using glUseProgram()
    # - Bind the VAO using glBindVertexArray()
    # - Draw the triangle using glDrawArrays()

# Run a loop to keep the program running
iteration = 0
draw = True
while draw:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            draw = False
    iteration = iteration % len(frameList)
    glBufferData(GL_ARRAY_BUFFER, size=frameList[int(iteration)].nbytes,
                 data=frameList[int(iteration)], usage=GL_STATIC_DRAW)
    iteration += 1 / 120
    # Clear color buffer and depth buffer before drawing each frame
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Draw Rayman
    glUseProgram(shader)
    glBindVertexArray(vao)
    glDrawArrays(GL_POINTS, 0, len(frameList[0]) // 2)
    cameraFOV = fovSlider.get_value()

    # Camera center should be the center of the video capture,
    # which for has dimensions 708:708
    height = width = 708

    # Compute camera matrices
    eye = [width/2, height/2, width/2]

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