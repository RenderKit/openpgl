#version 330 core
layout (points) in;
layout (points, max_vertices = 1) out;

uniform mat4 transform;
uniform mat4 projection;
uniform float pointSize;
in VS_OUT {
    vec3 color;
} gs_in[];

out vec3 fColor;

void build_house(vec4 position)
{    
    fColor = gs_in[0].color; // gs_in[0] since there's only one input vertex
    gl_Position = projection * transform * position; //+ vec4(-1.0, -1.0, 0.0, 0.0); // 1:bottom-left   
    gl_PointSize = pointSize;
    EmitVertex();   
    //gl_Position = position + vec4( 0.2, -0.2, 0.0, 0.0); // 2:bottom-right
    //EmitVertex();
    //gl_Position = position + vec4(-0.2,  0.2, 0.0, 0.0); // 3:top-left
    //EmitVertex();
    //gl_Position = position + vec4( 0.2,  0.2, 0.0, 0.0); // 4:top-right
    //EmitVertex();
    //gl_Position = position + vec4( 0.0,  0.4, 0.0, 0.0); // 5:top
    fColor = vec3(1.0, 1.0, 1.0);
    //EmitVertex();
    EndPrimitive();
}

void main() {    
    build_house(gl_in[0].gl_Position);
}