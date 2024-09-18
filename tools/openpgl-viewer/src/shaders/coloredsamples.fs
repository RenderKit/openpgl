#version 330 core

out vec4 FragColor;

in vec4 centerPos;
in float outPointSize;
in vec3 vertexColor;
uniform vec4 viewport;

void main()
{
	vec2 tmpCP = centerPos.xy/centerPos.w;
	tmpCP /= 2.0f;
	tmpCP += 0.5f;
	tmpCP.x *= (viewport.z-viewport.x);
	tmpCP.y *= (viewport.w-viewport.y);
	
	vec2 pos = gl_FragCoord.xy-tmpCP;
	float dist_squared = dot(pos, pos);
	if (dist_squared < outPointSize){
		FragColor = vec4(vertexColor, 1.0);
	}else {
		discard;
	}	
}