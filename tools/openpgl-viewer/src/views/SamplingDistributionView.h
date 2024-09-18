#pragma once 
#include "View.h"

#include <openpgl/cpp/OpenPGL.h>
#include "../data/Data.h"

#include <colormap/colormap.h>

#include <vector>
#include <string>

#include "../GLShader.h"

struct SamplingDistributionView: public View {
    enum SamplingDistributionType{
        ESURFACE_DISTRIBUTION = 0,
        EVOLUME_DISTRIBUTION
    };
    SamplingDistributionView(Data* data, SamplingDistributionType type);

    void clear();
    void dataUpdated() override {};
    void drawUI() override;
    void draw() override;
    void drawViewport() override;

    void update(glm::vec3 pos);


    private:
    void init(SamplingDistributionType type);
    int m_imgWidth {640};
    int m_imgHeight {320};
    openpgl::cpp::SurfaceSamplingDistribution* m_ssd {nullptr};
    glm::vec3* m_pdfImg {nullptr};
#ifdef OPENPGL_RADIANCE_CACHES
    glm::vec3* m_radianceImg {nullptr};
    glm::vec3* m_irradianceImg {nullptr};
    glm::vec3* m_outgoingRadianceImg {nullptr};
#endif

    unsigned int m_pdf_texture;
#ifdef OPENPGL_RADIANCE_CACHES
    unsigned int m_radiance_texture;
    unsigned int m_irradiance_texture;
    unsigned int m_outgoing_radiance_texture;
#endif

    SamplingDistributionType m_distributionType {ESURFACE_DISTRIBUTION};

    float m_minValue;
    float m_maxValue;

    float m_meanCosine {0.f};

    float m_gamma {2.2f};
    float m_exp {0.f};
    bool m_useLog {false};
    bool m_normalize {true};

    colormap::Colormap* m_colorMap;

    glm::vec3 m_pos;


    std::vector<std::string> m_modes;
    int m_prevMode = -1;
    int m_selectedMode = 0;
    const char* current_item = NULL;

    unsigned int m_texture_id;

    unsigned int m_fbo;
    Shader m_frameBufferShader;
    unsigned int quadVAO, quadVBO;


    std::vector<glm::vec3> m_samplePositions;
    std::vector<glm::vec3> m_sampleColorsPDF;
#ifdef OPENPGL_RADIANCE_CACHES
    std::vector<glm::vec3> m_sampleColorsRadiance;
    std::vector<glm::vec3> m_sampleColorsIrradiance;
    std::vector<glm::vec3> m_sampleColorsOutgoingRadiance;
#endif

    bool m_showColoredSamples {false};
    float m_coloredSampleSize {5.f};

    Shader m_shader;
    unsigned int vao_points;
    unsigned int vbo_points;
    unsigned int vbo_colors;
};