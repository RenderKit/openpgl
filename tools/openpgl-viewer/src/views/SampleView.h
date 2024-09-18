#pragma once
#include "../GLShader.h"
#include "View.h"

#include <string>
#include <vector>
#include <openpgl/cpp/OpenPGL.h>

#include "../math/AABB.h"
#include <glm/vec3.hpp>

#include "../math/Camera.h"
#include "../data/Data.h"

class SampleView: public View{

public:
    SampleView(Data* data);

    void loadSampleData();

    void dataUpdated() override {loadSampleData();};
    void drawViewport() override;
    void drawUI()override;
    //void prepare
private:
    std::string m_samplesFileName;

    bool m_showSamples {true};

    Shader m_shader;
    openpgl::cpp::SampleStorage* m_sampleStorage;

    std::vector<glm::vec3> m_samplePositions;
    std::vector<unsigned int> m_sampleIndices;

    float m_pointSize;
    glm::vec3 m_pointColor;

    AABB m_sampleBounds;
    
    unsigned int vao;
    unsigned int bo;
    unsigned int ebo;
};