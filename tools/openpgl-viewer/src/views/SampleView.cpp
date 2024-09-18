#include "SampleView.h"

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <sstream>

#include "../FileManager.h"

SampleView::SampleView(Data *data) : View(data)
{
    m_pointSize = 1.5f;
    m_pointColor = glm::vec3(.0, .50, .50);
    m_shader.init(FileManager::read("samples.vs"), FileManager::read("samples.fs"));
}

void SampleView::loadSampleData()
{
    // m_samplesFileName = filename;
    m_sampleStorage = m_data->getSamples();
    if (m_sampleStorage)
    {
        std::cout << "SampleStorage: surface size = " << m_sampleStorage->GetSizeSurface() << std::endl;
        m_sampleBounds = AABB();
        m_samplePositions.clear();
        for (int n = 0; n < m_sampleStorage->GetSizeSurface(); n++)
        {
            openpgl::cpp::SampleData sample = m_sampleStorage->GetSampleSurface(n);
            glm::vec3 p = glm::vec3(sample.position.x, sample.position.y, sample.position.z);
            // glm::vec3 p = glm::vec3(n*1.0f/float(m_sampleStorage->GetSizeSurface()), n*1.0f/float(m_sampleStorage->GetSizeSurface()), 0.0f);
            m_sampleBounds.extend(p);
            m_samplePositions.push_back(p);
            m_sampleIndices.push_back(n);
        }
        std::cout << "AABB: min = " << m_sampleBounds.min[0] << " " << m_sampleBounds.min[1] << " " << m_sampleBounds.min[2] << "\t max = " << m_sampleBounds.max[0] << " "
                  << m_sampleBounds.max[1] << " " << m_sampleBounds.max[2] << std::endl;

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &bo);
        glGenBuffers(1, &ebo);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, bo);
        glBufferData(GL_ARRAY_BUFFER, m_samplePositions.size() * sizeof(glm::vec3), m_samplePositions.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_sampleIndices.size() * sizeof(unsigned int), m_sampleIndices.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
}

void SampleView::drawUI()
{
    /*
    ImGui::Begin("File:");
      // open Dialog Simple
      if (ImGui::Button("Open File Dialog"))
        ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".cpp,.h,.hpp", ".");

      // display
      if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey"))
      {
        // action if OK
        if (ImGuiFileDialog::Instance()->IsOk())
        {
          std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
          std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
          // action
        }

        // close
        ImGuiFileDialog::Instance()->Close();
      }
    ImGui::End();
    */

    // Your ImGui code here
    ImGui::Begin("Samples:");
    ImGui::Checkbox("show samples:", &m_showSamples);
    ImGui::SliderFloat("pointSize", &m_pointSize, 1, 20);
    ImGui::ColorEdit3("pointColor", (float *)&m_pointColor);
    ImGui::End();
}

void SampleView::drawViewport()
{
    if (m_showSamples)
    {
        Camera *cam = m_data->getCamera();
        m_shader.bind();
        m_shader.setUniform("transform", (float *)&cam->m_transform);
        m_shader.setUniform("projection", (float *)&cam->m_projection);
        m_shader.setUniform("pointSize", m_pointSize);
        m_shader.setUniform("pointColor", m_pointColor[0], m_pointColor[1], m_pointColor[2]);
        m_shader.setUniform("viewport", cam->m_viewport[0], cam->m_viewport[1], cam->m_viewport[2], cam->m_viewport[3]);

        glBindVertexArray(vao);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glDrawElements(GL_POINTS, m_samplePositions.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
}