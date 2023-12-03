#version 330 core

in vec3 fragNormal;
out vec4 fragColor;

void main()
{
    vec3 N = normalize(fragNormal);
    fragColor = vec4(abs(N), 1.0);
}