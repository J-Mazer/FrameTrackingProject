#version 330 core
#extension GL_ARB_explicit_uniform_location : enable

in vec3 fragNormal;
in vec3 frag_pos;
out vec4 outColor;

uniform vec3 eyePos;

in vec2 fragUV;

layout (location = 0) uniform sampler2D tex2D;
layout (location = 1) uniform samplerCube cubeMapTex;

void main()
{
    vec3 frag_normal = normalize(fragNormal);

    vec3 view_direction = normalize(eyePos - frag_pos);
    vec3 reflected_view = reflect(-view_direction, frag_normal);

    vec2 tex_coords = fragUV; // Set your desired texture coordinates here
    vec3 color_tex = texture(tex2D, tex_coords).rgb;
    vec3 color_env = texture(cubeMapTex, reflected_view).rgb;
    vec3 color_mix = mix(color_tex, color_env, 0.2);
    outColor = vec4(color_mix, 1.0);
    //vec4(abs(normalize(fragNormal)), 1.0);
}