#pragma once

#include "GLShader.h"

struct PickBuffer
{
    struct PixelInfo
    {
        float ObjectID;
        float DrawID;
        float PrimID;

        PixelInfo()
        {
            ObjectID = 0.0f;
            DrawID = 0.0f;
            PrimID = 0.0f;
        }
    };

    void init(float width, float height);
    void bind();

    void unbind();

    void rescale(float width, float height);

    PixelInfo readPixel(unsigned int x, unsigned int y);

   public:
    unsigned int m_fbo;
    unsigned int m_rbo;

    unsigned int m_texture_id;
    unsigned int m_depth_texture_id;

    float m_width;
    float m_height;
};