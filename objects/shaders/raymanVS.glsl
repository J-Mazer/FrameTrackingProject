#version 330 core

in vec3 position;
in vec3 normal;
uniform float scale;
uniform vec3 center;
uniform float aspect;
out vec3 fragNormal;

void main()
{
    vec3 newPosition = position - center;
    newPosition = newPosition * scale;
    newPosition.x = newPosition.x / aspect;
    gl_Position = vec4(newPosition, 1.0);
    fragNormal = normalize(normal);

}
