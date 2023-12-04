#version 330 core
#extension GL_ARB_explicit_uniform_location : enable

in vec3 fragNormal;
out vec4 outColor;

in vec2 fragUV;

layout (location = 0) uniform sampler2D tex2D;

void main()
{
    vec2 tex_coords = fragUV; // Set your desired texture coordinates here
    vec3 color_tex = texture(tex2D, tex_coords).rgb;
    outColor = vec4(color_tex, 1.0);
    //vec4(abs(normalize(fragNormal)), 1.0);
}