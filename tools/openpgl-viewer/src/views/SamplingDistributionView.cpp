#include "SamplingDistributionView.h"

#include <cmath>
#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
// #include <glm/gtc/matrix_transform_3d.hpp>
#include <GL/glew.h>
#include <imgui.h>
#include <tbb/parallel_for.h>

#include <iostream>
#include <limits>
#include <sstream>

#include "../FileManager.h"
#include "shaders/coloredsamples_sc.h"
#include "shaders/sampledistribution_sc.h"

inline glm::vec3 sphericalDirectionZUP(const float &theta, const float &phi)
{
    const float cosTheta = std::cos(theta);
    const float sinTheta = std::sin(theta);
    const float cosPhi = std::cos(phi);
    const float sinPhi = std::sin(phi);
    return glm::vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
};

inline glm::vec3 sphericalDirectionYUP(const float &theta, const float &phi)
{
    const float cosTheta = std::cos(theta);
    const float sinTheta = std::sin(theta);
    const float cosPhi = std::cos(phi);
    const float sinPhi = std::sin(phi);
    return glm::vec3(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi);
};

SamplingDistributionView::SamplingDistributionView(Data *data, SamplingDistributionType type) : View(data)
{
    init(type);
    // m_colorMap = new colormap::transform::Saturn();
    m_colorMap = new colormap::MATLAB::Winter();
    // m_colorMap = new colormap::MATLAB::Autumn();
    // m_colorMap = new colormap::MATLAB::Jet();
    // m_colorMap = new colormap::MATLAB::Cool();
    // m_colorMap = new colormap::transform::HotMetal();
    // m_modes = { "AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG", "HHHH", "IIII", "JJJJ", "KKKK", "LLLLLLL", "MMMM", "OOOOOOO" };
    m_modes.push_back("pdf");
#ifdef OPENPGL_RADIANCE_CACHES
    m_modes.push_back("radiance");
    if (type == ESURFACE_DISTRIBUTION)
        m_modes.push_back("irradiance");
    else
        m_modes.push_back("inscattered radiance");
    m_modes.push_back("outgoing radiance");
#endif
    current_item = m_modes[0].c_str();
}

void SamplingDistributionView::init(SamplingDistributionType type)
{
    m_frameBufferShader.init(sampledistribution_vs, sampledistribution_fs);
    m_shader.init(coloredsamples_vs, coloredsamples_fs);
    m_distributionType = type;
    m_pdfImg = new glm::vec3[m_imgWidth * m_imgHeight];
#ifdef OPENPGL_RADIANCE_CACHES
    m_radianceImg = new glm::vec3[m_imgWidth * m_imgHeight];
    m_irradianceImg = new glm::vec3[m_imgWidth * m_imgHeight];
    m_outgoingRadianceImg = new glm::vec3[m_imgWidth * m_imgHeight];
#endif
    clear();

    glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

    glGenTextures(1, &m_texture_id);
    glBindTexture(GL_TEXTURE_2D, m_texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, m_imgWidth, m_imgHeight, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texture_id, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    // Create a OpenGL texture identifier
    glGenTextures(1, &m_pdf_texture);
    glBindTexture(GL_TEXTURE_2D, m_pdf_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);  // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);  // Same

    glBindTexture(GL_TEXTURE_2D, 0);
#ifdef OPENPGL_RADIANCE_CACHES
    glGenTextures(1, &m_radiance_texture);
    glBindTexture(GL_TEXTURE_2D, m_radiance_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);  // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);  // Same

    glBindTexture(GL_TEXTURE_2D, 0);

    glGenTextures(1, &m_irradiance_texture);
    glBindTexture(GL_TEXTURE_2D, m_irradiance_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);  // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);  // Same

    glBindTexture(GL_TEXTURE_2D, 0);

    glGenTextures(1, &m_outgoing_radiance_texture);
    glBindTexture(GL_TEXTURE_2D, m_outgoing_radiance_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);  // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);  // Same

    glBindTexture(GL_TEXTURE_2D, 0);
#endif

    // Restore the default framebuffer
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    float quadVertices[] = {// vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
                            // positions   // texCoords
                            -1.0f, 1.0f, 0.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f, 0.0f,

                            -1.0f, 1.0f, 0.0f, 1.0f, 1.0f,  -1.0f, 1.0f, 0.0f, 1.0f, 1.0f,  1.0f, 1.0f};

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
}

void SamplingDistributionView::clear()
{
    if (m_pdfImg)
    {
        for (int i = 0; i < m_imgWidth * m_imgHeight; i++)
        {
            m_pdfImg[i] = glm::vec3(0.f, 0.f, 0.f);
#ifdef OPENPGL_RADIANCE_CACHES
            m_radianceImg[i] = glm::vec3(0.f, 0.f, 0.f);
            m_irradianceImg[i] = glm::vec3(0.f, 0.f, 0.f);
            m_outgoingRadianceImg[i] = glm::vec3(0.f, 0.f, 0.f);
#endif
        }
    }
}

void SamplingDistributionView::drawViewport()
{
    if (m_showColoredSamples)
    {
        if (m_prevMode != m_selectedMode)
        {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
            if (m_selectedMode == 0)
            {
                glBufferData(GL_ARRAY_BUFFER, m_sampleColorsPDF.size() * sizeof(glm::vec3), m_sampleColorsPDF.data(), GL_STATIC_DRAW);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
#ifdef OPENPGL_RADIANCE_CACHES
            }
            else if (m_selectedMode == 1)
            {
                glBufferData(GL_ARRAY_BUFFER, m_sampleColorsRadiance.size() * sizeof(glm::vec3), m_sampleColorsRadiance.data(), GL_STATIC_DRAW);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
            }
            else if (m_selectedMode == 2)
            {
                glBufferData(GL_ARRAY_BUFFER, m_sampleColorsIrradiance.size() * sizeof(glm::vec3), m_sampleColorsIrradiance.data(), GL_STATIC_DRAW);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
            }
            else if (m_selectedMode == 3)
            {
                glBufferData(GL_ARRAY_BUFFER, m_sampleColorsOutgoingRadiance.size() * sizeof(glm::vec3), m_sampleColorsOutgoingRadiance.data(), GL_STATIC_DRAW);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);

#endif
            }
            else
            {
                glBufferData(GL_ARRAY_BUFFER, m_sampleColorsPDF.size() * sizeof(glm::vec3), m_sampleColorsPDF.data(), GL_STATIC_DRAW);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
            }
            glEnableVertexAttribArray(0);
            glBindVertexArray(0);
            m_prevMode = m_selectedMode;
        }

        Camera *cam = m_data->getCamera();
        if (m_samplePositions.size() > 0)
        {
            m_shader.bind();
            m_shader.setUniform("transform", (float *)&cam->m_transform);
            m_shader.setUniform("projection", (float *)&cam->m_projection);
            m_shader.setUniform("pointSize", m_coloredSampleSize);
            m_shader.setUniform("exposure", m_exp);
            m_shader.setUniform("gamma", m_gamma);
            m_shader.setUniform("viewport", cam->m_viewport[0], cam->m_viewport[1], cam->m_viewport[2], cam->m_viewport[3]);

            glBindVertexArray(vao_points);
            // glEnable(GL_DEPTH_TEST);
            // glDepthFunc(GL_LESS);
            // glDepthMask(GL_TRUE);
            glEnable(GL_PROGRAM_POINT_SIZE);
            glDrawArrays(GL_POINTS, 0, m_samplePositions.size());
            glBindVertexArray(0);
        }
    }
}
void SamplingDistributionView::draw()
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glViewport(0, 0, m_imgWidth, m_imgHeight);
    glClearColor(0.95f, 0.55f, 0.60f, 1.00f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_frameBufferShader.bind();
    m_frameBufferShader.setUniform("uselog", m_useLog);
    m_frameBufferShader.setUniform("gamma", m_gamma);
    m_frameBufferShader.setUniform("exposure", m_exp);
    glBindVertexArray(quadVAO);
    glActiveTexture(GL_TEXTURE0);
    if (m_selectedMode == 0)
    {
        glBindTexture(GL_TEXTURE_2D, m_pdf_texture);  // use the color attachment texture as the texture of the quad plane
    }
#ifdef OPENPGL_RADIANCE_CACHES
    else if (m_selectedMode == 1)
    {
        glBindTexture(GL_TEXTURE_2D, m_radiance_texture);  // use the color attachment texture as the texture of the quad plane
    }
    else if (m_selectedMode == 2)
    {
        glBindTexture(GL_TEXTURE_2D, m_irradiance_texture);  // use the color attachment texture as the texture of the quad plane
    }
    else if (m_selectedMode == 3)
    {
        glBindTexture(GL_TEXTURE_2D, m_outgoing_radiance_texture);  // use the color attachment texture as the texture of the quad plane
    }
#endif
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SamplingDistributionView::drawUI()
{
    if (m_distributionType == ESURFACE_DISTRIBUTION)
        ImGui::Begin("SurfaceSamplingDistribution:");
    else
        ImGui::Begin("VolumeSamplingDistribution:");

    ImVec2 viewportPanelSize = ImGui::GetContentRegionAvail();

    if (ImGui::BeginCombo("##combo", current_item))  // The second parameter is the label previewed before opening the combo.
    {
        for (int n = 0; n < m_modes.size(); n++)
        {
            bool is_selected = (n == m_selectedMode);  // You can store your selection however you want, outside or inside your objects
            if (ImGui::Selectable(m_modes[n].c_str(), is_selected))
            {
                current_item = m_modes[n].c_str();
                m_selectedMode = n;
            }
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::Image((void *)(intptr_t)m_texture_id, ImVec2(viewportPanelSize[0], viewportPanelSize[0] / 2));
    if (m_distributionType == EVOLUME_DISTRIBUTION)
        ImGui::SliderFloat("g:", &m_meanCosine, -1.f, 1.f);
    ImGui::Checkbox("log plot:", &m_useLog);
    // ImGui::Checkbox("normalize:", &m_normalize);
    ImGui::SliderFloat("exposure:", &m_exp, -5.f, 5.f);
    ImGui::SliderFloat("gamma:", &m_gamma, 1.f, 2.2f);
    ImGui::Checkbox("show colored samples:", &m_showColoredSamples);
    ImGui::SliderFloat("sample size:", &m_coloredSampleSize, 1.f, 10.f);

    ImGui::End();
}
void SamplingDistributionView::update(glm::vec3 pos)
{
    m_pos = pos;
    const Camera *camera = m_data->getCamera();
    m_ssd = m_data->getSurfaceSamplingDistribution(pos);
    float stepPhi = (2.0f * M_PI) / (float)m_imgWidth;
    float stepTheta = (M_PI) / (float)m_imgHeight;

    m_minValue = std::numeric_limits<float>::max();
    m_maxValue = std::numeric_limits<float>::min();
    if (m_ssd)
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, m_imgHeight), [&](tbb::blocked_range<int> r) {
            for (int y = r.begin(); y < r.end(); ++y)
            {
                // for (int y = 0; y < m_imgHeight; y++) {
                for (int x = 0; x < m_imgWidth; x++)
                {
                    int idx = (y * m_imgWidth) + x;
                    float theta = stepTheta * (0.5f + float(y));
                    float phi = stepPhi * (0.5f + float(x));
                    glm::vec3 dir;
                    switch (camera->m_upType)
                    {
                        case Camera::Z_UP:
                        {
                            dir = sphericalDirectionZUP(theta, phi);
                            break;
                        }
                        case Camera::Y_UP:
                        {
                            dir = sphericalDirectionYUP(theta, phi);
                            break;
                        }
                        case Camera::X_UP:
                        {
                            dir = sphericalDirectionZUP(theta, phi);
                            break;
                        }
                    }

                    pgl_vec3f pglDir;
                    pglDir.x = dir.x;
                    pglDir.y = dir.y;
                    pglDir.z = dir.z;
                    float pdf = m_ssd->PDF(pglDir);
#ifdef OPENPGL_RADIANCE_CACHES
                    pgl_vec3f radiance = m_ssd->IncomingRadiance(pglDir, false);
                    pgl_vec3f irradiance = m_ssd->Irradiance(pglDir, false);
                    pgl_vec3f outgoingradiance = m_ssd->OutgoingRadiance(pglDir);
#endif
                    m_minValue = std::min(m_minValue, pdf);
                    m_maxValue = std::max(m_maxValue, pdf);
                    m_pdfImg[idx] = glm::vec3(pdf, pdf, pdf);
#ifdef OPENPGL_RADIANCE_CACHES
                    m_radianceImg[idx] = glm::vec3(radiance.x, radiance.y, radiance.z);
                    m_irradianceImg[idx] = glm::vec3(irradiance.x, irradiance.y, irradiance.z);
                    m_outgoingRadianceImg[idx] = glm::vec3(outgoingradiance.x, outgoingradiance.y, outgoingradiance.z);
#endif
                }
            }
        });

        openpgl::cpp::SampleStorage *ss = m_data->getSamples();
        if (ss)
        {
            int nSamples = ss->GetSizeSurface();
            if (m_samplePositions.size() != nSamples)
            {
                m_samplePositions.resize(nSamples);
                m_sampleColorsPDF.resize(nSamples);
#ifdef OPENPGL_RADIANCE_CACHES
                m_sampleColorsRadiance.resize(nSamples);
                m_sampleColorsIrradiance.resize(nSamples);
                m_sampleColorsOutgoingRadiance.resize(nSamples);
#endif
            }
            // for (int n =0; n < ss->GetSizeSurface(); n++){
            tbb::parallel_for(tbb::blocked_range<int>(0, ss->GetSizeSurface()), [&](tbb::blocked_range<int> r) {
                for (int n = r.begin(); n < r.end(); ++n)
                {
                    openpgl::cpp::SampleData sd = ss->GetSampleSurface(n);
                    glm::vec3 p = glm::vec3(sd.position.x, sd.position.y, sd.position.z);
                    m_samplePositions[n] = p;
                    glm::vec3 dir = p - m_pos;
                    dir = glm::normalize(dir);
                    pgl_vec3f pglDir;
                    pglDir.x = dir.x;
                    pglDir.y = dir.y;
                    pglDir.z = dir.z;
                    float pdf = m_ssd->PDF(pglDir);
                    m_sampleColorsPDF[n] = glm::vec3(pdf, pdf, pdf);
#ifdef OPENPGL_RADIANCE_CACHES
                    pgl_vec3f radiance = m_ssd->IncomingRadiance(pglDir, false);
                    m_sampleColorsRadiance[n] = glm::vec3(radiance.x, radiance.y, radiance.z);
                    pgl_vec3f irradiance = m_ssd->Irradiance(pglDir, false);
                    m_sampleColorsIrradiance[n] = glm::vec3(irradiance.x, irradiance.y, irradiance.z);
                    pgl_vec3f outRadiance = m_ssd->OutgoingRadiance(pglDir);
                    m_sampleColorsOutgoingRadiance[n] = glm::vec3(outRadiance.x, outRadiance.y, outRadiance.z);
#endif
                }
            });
        }
    }

    glBindTexture(GL_TEXTURE_2D, m_pdf_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, m_imgWidth, m_imgHeight, 0, GL_RGB, GL_FLOAT, (float *)m_pdfImg);
    glBindTexture(GL_TEXTURE_2D, 0);

#ifdef OPENPGL_RADIANCE_CACHES
    glBindTexture(GL_TEXTURE_2D, m_radiance_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, m_imgWidth, m_imgHeight, 0, GL_RGB, GL_FLOAT, (float *)m_radianceImg);
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindTexture(GL_TEXTURE_2D, m_irradiance_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, m_imgWidth, m_imgHeight, 0, GL_RGB, GL_FLOAT, (float *)m_irradianceImg);
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindTexture(GL_TEXTURE_2D, m_outgoing_radiance_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, m_imgWidth, m_imgHeight, 0, GL_RGB, GL_FLOAT, (float *)m_outgoingRadianceImg);
    glBindTexture(GL_TEXTURE_2D, 0);
#endif
    if (m_samplePositions.size() > 0)
    {
        glGenVertexArrays(1, &vao_points);
        glGenBuffers(1, &vbo_points);
        glGenBuffers(1, &vbo_colors);

        glBindVertexArray(vao_points);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_points);
        glBufferData(GL_ARRAY_BUFFER, m_samplePositions.size() * sizeof(glm::vec3), m_samplePositions.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
        if (m_selectedMode == 0)
        {
            glBufferData(GL_ARRAY_BUFFER, m_sampleColorsPDF.size() * sizeof(glm::vec3), m_sampleColorsPDF.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
#ifdef OPENPGL_RADIANCE_CACHES
        }
        else if (m_selectedMode == 1)
        {
            glBufferData(GL_ARRAY_BUFFER, m_sampleColorsRadiance.size() * sizeof(glm::vec3), m_sampleColorsRadiance.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
        }
        else if (m_selectedMode == 2)
        {
            glBufferData(GL_ARRAY_BUFFER, m_sampleColorsIrradiance.size() * sizeof(glm::vec3), m_sampleColorsIrradiance.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
        }
        else if (m_selectedMode == 3)
        {
            glBufferData(GL_ARRAY_BUFFER, m_sampleColorsOutgoingRadiance.size() * sizeof(glm::vec3), m_sampleColorsOutgoingRadiance.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
#endif
        }
        else
        {
            glBufferData(GL_ARRAY_BUFFER, m_sampleColorsPDF.size() * sizeof(glm::vec3), m_sampleColorsPDF.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
        }
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);
    }
}
