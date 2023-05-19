#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec2 inUV;

layout (binding = 0) uniform Camera {
    mat4 view;
    mat4 projection;
} camera;

layout (binding = 1) uniform Instance {
    mat4 model;
} instance;

layout(location = 0) out vec2 outUV;

void main() {
    outUV = inUV;
    gl_Position = camera.projection * camera.view * instance.model * vec4(inPos, 1.0);
}
