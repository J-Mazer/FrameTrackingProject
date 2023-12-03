#version 450 core

in vec3 position;
in vec3 normal;

uniform mat4 model_matrix;
uniform float scale;
uniform vec3 center;
uniform float aspect;

out vec3 fragNormal;  // This data will be passed to the fragment shader

void main()
{
    // Transform the position from object space (a.k.a model space) to clip space. The range of clip space is [-1,1] in all 3 dimensions.
    vec4 pos = model_matrix * vec4(position, 1.0);
    pos.x = pos.x / aspect; // Correction for aspect ratio (optional)
    gl_Position = pos;

    // Transform the normal from object (or model) space to world space
    mat4 normal_matrix = transpose(inverse(model_matrix));
    vec3 new_normal = (normal_matrix * vec4(normal,0)).xyz;
    fragNormal = normalize(new_normal);
}