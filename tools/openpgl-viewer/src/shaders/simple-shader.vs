#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;

out vec3 vertexColor;

uniform float rotation;
uniform vec2 translation;

void main()
{
	vec2 rotated_pos;
	rotated_pos.x = translation.x + position.x*cos(rotation) - position.y*sin(rotation);
	rotated_pos.y = translation.y + position.x*sin(rotation) + position.y*cos(rotation);
    gl_Position = vec4(rotated_pos.x, rotated_pos.y, position.z, 1.0);
	gl_PointSize = 10.0;
	vertexColor = color;
}