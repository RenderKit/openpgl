#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;

out vec3 vertexColor;
out vec4 centerPos;
out float outPointSize;

uniform mat4 transform;
uniform mat4 projection;
uniform float pointSize;

uniform float exposure;
uniform float gamma;

void main()
{
    gl_Position = projection * transform * vec4(position, 1.0f);
	centerPos = projection * transform * vec4(position, 1.0f);
	gl_PointSize = pointSize;
	outPointSize = pointSize;

	float scale = pow(2.f, exposure);
	vertexColor = color;
	vertexColor *= scale;
	vertexColor.x = pow(vertexColor.x, 1.f/ gamma);
	vertexColor.y = pow(vertexColor.y, 1.f/ gamma);
	vertexColor.z = pow(vertexColor.z, 1.f/ gamma);
	
}