#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenTexture;

uniform float exposure;
uniform float gamma;
uniform bool uselog;
//uniform bool normalize;

void main()
{
    vec3 col = texture(screenTexture, TexCoords).rgb;
    float scale = pow(2.f, exposure);
    if (uselog) {
        col += 1.f; 
        col.x = log(col.x);
        col.y = log(col.y);
        col.z = log(col.z);
    } else {
        col *= scale;
        col.x = pow(col.x, 1.f/ gamma);
        col.y = pow(col.y, 1.f/ gamma);
        col.z = pow(col.z, 1.f/ gamma);
    }



    FragColor = vec4(col, 1.0);
} 