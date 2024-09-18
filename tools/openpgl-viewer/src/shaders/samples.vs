#version 330 core

layout (location = 0) in vec3 position;
//layout (location = 1) in vec3 color;

//out vec3 vertexColor;
out vec4 centerPos;
out float outPointSize;

uniform mat4 transform;
uniform mat4 projection;
uniform float pointSize;
//uniform vec2 translation;

void main()
{
	//vec2 rotated_pos;
	//rotated_pos.x = translation.x + position.x*cos(rotation) - position.y*sin(rotation);
	//rotated_pos.y = translation.y + position.x*sin(rotation) + position.y*cos(rotation);
    gl_Position = projection * transform * vec4(position, 1.0f);
	centerPos = projection * transform * vec4(position, 1.0f);
	//gl_Position = vec4(position, 1.0f);
	gl_PointSize = pointSize;
	outPointSize = pointSize;
	//vertexColor = pointColor;
}