#version 330 core
layout (points) in;
layout (line_strip, max_vertices = 24) out;

uniform mat4 transform;
uniform mat4 projection;
uniform float pointSize;
in VS_OUT {
    vec3 color;
    vec3 lower;
    vec3 upper;
} gs_in[];

out vec3 fColor;

void build_house(vec4 position)
{    
    fColor = gs_in[0].color; // gs_in[0] since there's only one input vertex
/*
    gl_Position = projection * transform * vec4(gs_in[0].upper, 1.0); // 1:bottom-left   
    //gl_Position = projection * transform * position;
    gl_PointSize = pointSize;
    EmitVertex();
    gl_Position = projection * transform * vec4(gs_in[0].lower, 1.0); // 1:bottom-left   
    //gl_Position = projection * transform * position;
    gl_PointSize = pointSize;
    EmitVertex();
    EndPrimitive();
*/
   
    gl_Position = projection * transform * vec4(gs_in[0].upper.x, gs_in[0].upper.y, gs_in[0].upper.z, 1.0); 
    EmitVertex();
    gl_Position = projection * transform * vec4(gs_in[0].lower.x, gs_in[0].upper.y, gs_in[0].upper.z, 1.0); 
    EmitVertex();    
    EndPrimitive();

    gl_Position = projection * transform * vec4(gs_in[0].lower.x, gs_in[0].upper.y, gs_in[0].upper.z, 1.0); 
    EmitVertex();
    gl_Position = projection * transform * vec4(gs_in[0].lower.x, gs_in[0].lower.y, gs_in[0].upper.z, 1.0); 
    EmitVertex();    
    EndPrimitive();

    gl_Position = projection * transform * vec4(gs_in[0].lower.x, gs_in[0].lower.y, gs_in[0].upper.z, 1.0); 
    EmitVertex();
    gl_Position = projection * transform * vec4(gs_in[0].upper.x, gs_in[0].lower.y, gs_in[0].upper.z, 1.0); 
    EmitVertex();    
    EndPrimitive();

    gl_Position = projection * transform * vec4(gs_in[0].upper.x, gs_in[0].lower.y, gs_in[0].upper.z, 1.0); 
    EmitVertex();
    gl_Position = projection * transform * vec4(gs_in[0].upper.x, gs_in[0].upper.y, gs_in[0].upper.z, 1.0); 
    EmitVertex();    
    EndPrimitive();


    gl_Position = projection * transform * vec4(gs_in[0].upper.x, gs_in[0].upper.y, gs_in[0].lower.z, 1.0); 
    EmitVertex();
    gl_Position = projection * transform * vec4(gs_in[0].lower.x, gs_in[0].upper.y, gs_in[0].lower.z, 1.0); 
    EmitVertex();    
    EndPrimitive();

    gl_Position = projection * transform * vec4(gs_in[0].lower.x, gs_in[0].upper.y, gs_in[0].lower.z, 1.0); 
    EmitVertex();
    gl_Position = projection * transform * vec4(gs_in[0].lower.x, gs_in[0].lower.y, gs_in[0].lower.z, 1.0); 
    EmitVertex();    
    EndPrimitive();

    gl_Position = projection * transform * vec4(gs_in[0].lower.x, gs_in[0].lower.y, gs_in[0].lower.z, 1.0); 
    EmitVertex();
    gl_Position = projection * transform * vec4(gs_in[0].upper.x, gs_in[0].lower.y, gs_in[0].lower.z, 1.0); 
    EmitVertex();    
    EndPrimitive();

    gl_Position = projection * transform * vec4(gs_in[0].upper.x, gs_in[0].lower.y, gs_in[0].lower.z, 1.0); 
    EmitVertex();
    gl_Position = projection * transform * vec4(gs_in[0].upper.x, gs_in[0].upper.y, gs_in[0].lower.z, 1.0); 
    EmitVertex();    
    EndPrimitive();


    gl_Position = projection * transform * vec4(gs_in[0].upper.x, gs_in[0].upper.y, gs_in[0].upper.z, 1.0); 
    EmitVertex();
    gl_Position = projection * transform * vec4(gs_in[0].upper.x, gs_in[0].upper.y, gs_in[0].lower.z, 1.0); 
    EmitVertex();    
    EndPrimitive();

    gl_Position = projection * transform * vec4(gs_in[0].lower.x, gs_in[0].upper.y, gs_in[0].upper.z, 1.0); 
    EmitVertex();
    gl_Position = projection * transform * vec4(gs_in[0].lower.x, gs_in[0].upper.y, gs_in[0].lower.z, 1.0); 
    EmitVertex();    
    EndPrimitive();

    gl_Position = projection * transform * vec4(gs_in[0].lower.x, gs_in[0].lower.y, gs_in[0].upper.z, 1.0); 
    EmitVertex();
    gl_Position = projection * transform * vec4(gs_in[0].lower.x, gs_in[0].lower.y, gs_in[0].lower.z, 1.0); 
    EmitVertex();    
    EndPrimitive();

    gl_Position = projection * transform * vec4(gs_in[0].upper.x, gs_in[0].lower.y, gs_in[0].upper.z, 1.0); 
    EmitVertex();
    gl_Position = projection * transform * vec4(gs_in[0].upper.x, gs_in[0].lower.y, gs_in[0].lower.z, 1.0); 
    EmitVertex();    
    EndPrimitive();

}

void main() {    
    build_house(gl_in[0].gl_Position);
}