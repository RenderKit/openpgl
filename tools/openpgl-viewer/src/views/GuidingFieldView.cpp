#include "GuidingFieldView.h"

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <sstream>

#include "../FileManager.h"
#include "shaders/cachesBBoxes_sc.h"
#include "shaders/caches_sc.h"

GuidingFieldView::GuidingFieldView(Data *data) : View(data)
{
    m_shader.init(caches_vs, caches_gs, caches_fs);
    m_shaderBBoxes.init(cachesBBoxes_vs, cachesBBoxes_gs, cachesBBoxes_fs);

    m_cacheColor = glm::vec3(1.0, .0, .50);
    m_cacheBoundsColor = glm::vec3(1.0, .50, .0);
    m_sampleBoundsColor = glm::vec3(.0, 1.0, .50);
    m_varianceBoundsColor = glm::vec3(.5, 0.75, 1.00);
}

void GuidingFieldView::drawUI()
{
    // Your ImGui code here
    ImGui::Begin("Guiding Field:");
    ImGui::Checkbox("show caches:", &m_drawCachePositions);
    ImGui::ColorEdit3("cache Color:", (float *)&m_cacheColor);

    ImGui::Checkbox("show cache bounds:", &m_drawCacheBounds);
    ImGui::ColorEdit3("cache bounds color:", (float *)&m_cacheBoundsColor);

    ImGui::Checkbox("show sample bounds:", &m_drawSampleBounds);
    ImGui::ColorEdit3("sample bounds color:", (float *)&m_sampleBoundsColor);

    ImGui::Checkbox("show sample variance bounds:", &m_drawVarianceBounds);
    ImGui::ColorEdit3("sample variance bounds color:", (float *)&m_varianceBoundsColor);
    // ImGui::Text("This is a simple ImGui application.");
    ImGui::End();
}

void GuidingFieldView::loadField()
{
    m_field = m_data->getField();
    std::cout << "Field: cache size = " << m_field->GetNumSurfaceCaches() << std::endl;

    // m_sampleBounds = AABB();
    m_cachePositions.clear();
    m_cacheSampleBoundsLowers.clear();
    m_cacheSampleBoundsUppers.clear();
    m_cacheBoundsLowers.clear();
    m_cacheBoundsUppers.clear();
    m_cacheVarianceBoundsLowers.clear();
    m_cacheVarianceBoundsUppers.clear();
    m_cacheIndices.clear();
    for (int n = 0; n < m_field->GetNumSurfaceCaches(); n++)
    {
        pgl_cacheInfo cacheInfo = m_field->GetSurfaceCacheInfo(n);
        // pgl_vec3f cachePos = cacheInfo.distributionPivot;
        pgl_vec3f cachePos = cacheInfo.sampleMean;
        pgl_box3f cacheSampleBounds = cacheInfo.sampleBounds;
        pgl_box3f cacheBounds = cacheInfo.cacheBounds;
        glm::vec3 p = glm::vec3(cachePos.x, cachePos.y, cachePos.z);

        glm::vec3 variance = glm::vec3(cacheInfo.sampleVariance.x, cacheInfo.sampleVariance.y, cacheInfo.sampleVariance.z);
        variance.x = 2.0 * sqrt(variance.x);
        variance.y = 2.0 * sqrt(variance.y);
        variance.z = 2.0 * sqrt(variance.z);

        glm::vec3 sampleLower = glm::vec3(cacheSampleBounds.lower.x, cacheSampleBounds.lower.y, cacheSampleBounds.lower.z);
        glm::vec3 sampleUpper = glm::vec3(cacheSampleBounds.upper.x, cacheSampleBounds.upper.y, cacheSampleBounds.upper.z);

        glm::vec3 cacheLower = glm::vec3(cacheBounds.lower.x, cacheBounds.lower.y, cacheBounds.lower.z);
        glm::vec3 cacheUpper = glm::vec3(cacheBounds.upper.x, cacheBounds.upper.y, cacheBounds.upper.z);

        glm::vec3 varianceLower = glm::vec3(cachePos.x - variance.x, cachePos.y - variance.y, cachePos.z - variance.z);
        glm::vec3 varianceUpper = glm::vec3(cachePos.x + variance.x, cachePos.y + variance.y, cachePos.z + variance.z);

        varianceLower.x = std::max(varianceLower.x, sampleLower.x);
        varianceLower.y = std::max(varianceLower.y, sampleLower.y);
        varianceLower.z = std::max(varianceLower.z, sampleLower.z);

        varianceUpper.x = std::min(varianceUpper.x, sampleUpper.x);
        varianceUpper.y = std::min(varianceUpper.y, sampleUpper.y);
        varianceUpper.z = std::min(varianceUpper.z, sampleUpper.z);

        // glm::vec3 p = glm::vec3(n*1.0f/float(m_sampleStorage->GetSizeSurface()), n*1.0f/float(m_sampleStorage->GetSizeSurface()), 0.0f);
        // m_sampleBounds.extend(p);
        m_cachePositions.push_back(p);
        m_cacheSampleBoundsLowers.push_back(sampleLower);
        m_cacheSampleBoundsUppers.push_back(sampleUpper);
        m_cacheBoundsLowers.push_back(cacheLower);
        m_cacheBoundsUppers.push_back(cacheUpper);
        m_cacheVarianceBoundsLowers.push_back(varianceLower);
        m_cacheVarianceBoundsUppers.push_back(varianceUpper);
        m_cacheIndices.push_back(n);
    }

    // unsigned int VBO, VAO;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, m_cachePositions.size() * sizeof(glm::vec3), m_cachePositions.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    glGenVertexArrays(1, &vaoBB);
    glGenBuffers(1, &vboBBLowers);
    glGenBuffers(1, &vboBBUppers);

    glBindVertexArray(vaoBB);

    glBindBuffer(GL_ARRAY_BUFFER, vboBBLowers);
    glBufferData(GL_ARRAY_BUFFER, m_cacheBoundsLowers.size() * sizeof(glm::vec3), m_cacheBoundsLowers.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, vboBBUppers);
    glBufferData(GL_ARRAY_BUFFER, m_cacheBoundsUppers.size() * sizeof(glm::vec3), m_cacheBoundsUppers.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);

    glGenVertexArrays(1, &vaoSampleBB);
    glGenBuffers(1, &vboSampleBBPositions);
    glGenBuffers(1, &vboSampleBBLowers);
    glGenBuffers(1, &vboSampleBBUppers);

    glBindVertexArray(vaoSampleBB);
    glBindBuffer(GL_ARRAY_BUFFER, vboSampleBBPositions);
    glBufferData(GL_ARRAY_BUFFER, m_cachePositions.size() * sizeof(glm::vec3), m_cachePositions.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vboSampleBBLowers);
    glBufferData(GL_ARRAY_BUFFER, m_cacheSampleBoundsLowers.size() * sizeof(glm::vec3), m_cacheSampleBoundsLowers.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, vboSampleBBUppers);
    glBufferData(GL_ARRAY_BUFFER, m_cacheSampleBoundsUppers.size() * sizeof(glm::vec3), m_cacheSampleBoundsUppers.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);

    glGenVertexArrays(1, &vaoVarianceBB);
    glGenBuffers(1, &vboVarianceBBPositions);
    glGenBuffers(1, &vboVarianceBBLowers);
    glGenBuffers(1, &vboVarianceBBUppers);

    glBindVertexArray(vaoVarianceBB);
    glBindBuffer(GL_ARRAY_BUFFER, vboVarianceBBPositions);
    glBufferData(GL_ARRAY_BUFFER, m_cachePositions.size() * sizeof(glm::vec3), m_cachePositions.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vboVarianceBBLowers);
    glBufferData(GL_ARRAY_BUFFER, m_cacheVarianceBoundsLowers.size() * sizeof(glm::vec3), m_cacheVarianceBoundsLowers.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, vboVarianceBBUppers);
    glBufferData(GL_ARRAY_BUFFER, m_cacheVarianceBoundsUppers.size() * sizeof(glm::vec3), m_cacheVarianceBoundsUppers.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);
}

void GuidingFieldView::drawViewport()
{
    Camera *cam = m_data->getCamera();
    if (m_drawCachePositions)
    {
        m_shader.bind();
        m_shader.setUniform("transform", (float *)&cam->m_transform);
        m_shader.setUniform("projection", (float *)&cam->m_projection);
        m_shader.setUniform("pointSize", 5.0f);
        m_shader.setUniform("color", m_cacheColor[0], m_cacheColor[1], m_cacheColor[2]);
        // m_shader.setUniform("viewport", cam.m_viewport[0],cam.m_viewport[1],cam.m_viewport[2],cam.m_viewport[3]);
        glBindVertexArray(vao);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glDrawArrays(GL_POINTS, 0, m_cachePositions.size());
        // glDrawArrays(GL_POINTS, 0, 4);
        // glDrawElements(GL_POINTS, m_cachePositions.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

    if (m_drawCacheBounds)
    {
        m_shaderBBoxes.bind();
        m_shaderBBoxes.setUniform("transform", (float *)&cam->m_transform);
        m_shaderBBoxes.setUniform("projection", (float *)&cam->m_projection);
        m_shaderBBoxes.setUniform("pointSize", 5.0f);
        m_shader.setUniform("color", m_cacheBoundsColor[0], m_cacheBoundsColor[1], m_cacheBoundsColor[2]);

        glBindVertexArray(vaoBB);
        // glEnable(GL_PROGRAM_POINT_SIZE);
        glDrawArrays(GL_POINTS, 0, m_cachePositions.size());
        // glDrawArrays(GL_POINTS, 0, 4);
        // glDrawElements(GL_POINTS, m_cachePositions.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

    if (m_drawSampleBounds)
    {
        m_shaderBBoxes.bind();
        m_shaderBBoxes.setUniform("transform", (float *)&cam->m_transform);
        m_shaderBBoxes.setUniform("projection", (float *)&cam->m_projection);
        m_shaderBBoxes.setUniform("pointSize", 5.0f);
        m_shader.setUniform("color", m_sampleBoundsColor[0], m_sampleBoundsColor[1], m_sampleBoundsColor[2]);
        // m_shader.setUniform("viewport", cam.m_viewport[0],cam.m_viewport[1],cam.m_viewport[2],cam.m_viewport[3]);

        glBindVertexArray(vaoSampleBB);
        // glEnable(GL_PROGRAM_POINT_SIZE);
        glDrawArrays(GL_POINTS, 0, m_cachePositions.size());
        // glDrawArrays(GL_POINTS, 0, 4);
        // glDrawElements(GL_POINTS, m_cachePositions.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

    if (m_drawVarianceBounds)
    {
        m_shaderBBoxes.bind();
        m_shaderBBoxes.setUniform("transform", (float *)&cam->m_transform);
        m_shaderBBoxes.setUniform("projection", (float *)&cam->m_projection);
        m_shaderBBoxes.setUniform("pointSize", 5.0f);
        m_shader.setUniform("color", m_varianceBoundsColor[0], m_varianceBoundsColor[1], m_varianceBoundsColor[2]);
        // m_shader.setUniform("viewport", cam.m_viewport[0],cam.m_viewport[1],cam.m_viewport[2],cam.m_viewport[3]);

        glBindVertexArray(vaoVarianceBB);
        glDrawArrays(GL_POINTS, 0, m_cachePositions.size());
        glBindVertexArray(0);
    }
}