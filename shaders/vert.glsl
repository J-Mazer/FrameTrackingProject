#version 330 core

in vec4 position;
uniform mat4 proj_matrix;
uniform mat4 view_matrix;

void main()
{
    gl_PointSize = 10.0;
    gl_Position = proj_matrix * view_matrix * position;
}
