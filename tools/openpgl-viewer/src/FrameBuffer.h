#pragma once

#include "GLShader.h"
#include "data/Data.h"

#include <imgui.h>

struct FrameBuffer {

    FrameBuffer() = default;

    void init(float width, float height);
    void bind();

    void unbind();

    void rescale(float width, float height);

    void draw();
    void clear();
    public:
    unsigned int m_fbo;
    unsigned int m_rbo;

    unsigned int m_texture_id;
    unsigned int m_depth_texture_id;

    float m_width;
    float m_height;

    Shader m_frameBufferShader;
    unsigned int quadVAO, quadVBO;
};