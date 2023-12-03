#version 450 core

layout (binding = 1) uniform samplerCube cubeMapTex;
uniform mat4 invViewProjectionMatrix;

in vec2 clipboxPosition;
out vec4 glColor;

void main()
{
    vec4 direction_4d = invViewProjectionMatrix * vec4(clipboxPosition, 1, 1);
    vec3 direction_3d = normalize(direction_4d.xyz / direction_4d.w);
    glColor = texture(cubeMapTex, direction_3d);
}