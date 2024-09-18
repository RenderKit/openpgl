#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 lower;
layout (location = 2) in vec3 upper;

uniform vec3 color;

out VS_OUT {
    vec3 color;
    vec3 lower;
    vec3 upper;
} vs_out;

void main()
{
    vs_out.color = color;
    vs_out.lower = lower;
    vs_out.upper = upper;
    gl_Position = vec4(position, 1.0); 
	gl_PointSize = 5.0;
}