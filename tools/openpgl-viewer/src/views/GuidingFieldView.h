#pragma once
#include "../GLShader.h"
#include "View.h"

#include <string>
#include <vector>
#include <openpgl/cpp/OpenPGL.h>

#include "../math/AABB.h"
#include <glm/vec3.hpp>
#include "../data/Data.h"
#include "../math/Camera.h"

class GuidingFieldView: public View {

public:
    GuidingFieldView(Data* data);

    void loadField();
    void dataUpdated() override {loadField();};
    void drawViewport() override;

    void drawUI() override;

public:
    Shader m_shader;
    Shader m_shaderBBoxes;
    openpgl::cpp::Field* m_field;

    std::vector<glm::vec3> m_cachePositions;
    std::vector<unsigned int> m_cacheIndices;

    std::vector<glm::vec3> m_cacheBoundsLowers;
    std::vector<glm::vec3> m_cacheBoundsUppers;

    std::vector<glm::vec3> m_cacheSampleBoundsLowers;
    std::vector<glm::vec3> m_cacheSampleBoundsUppers;

    std::vector<glm::vec3> m_cacheVarianceBoundsLowers;
    std::vector<glm::vec3> m_cacheVarianceBoundsUppers;

    //AABB m_sampleBounds;
    bool m_drawCachePositions {false};
    bool m_drawCacheBounds {false};
    bool m_drawSampleBounds {false};
    bool m_drawVarianceBounds {true};
    
    glm::vec3 m_cacheColor;
    glm::vec3 m_cacheBoundsColor;
    glm::vec3 m_sampleBoundsColor;
    glm::vec3 m_varianceBoundsColor;

    unsigned int vao;
    unsigned int vbo;

    unsigned int vaoBB;
    unsigned int vboBBLowers;
    unsigned int vboBBUppers;

    unsigned int vaoSampleBB;
    unsigned int vboSampleBBPositions;
    unsigned int vboSampleBBLowers;
    unsigned int vboSampleBBUppers;


    unsigned int vaoVarianceBB;
    unsigned int vboVarianceBBPositions;
    unsigned int vboVarianceBBLowers;
    unsigned int vboVarianceBBUppers;
};