#include "CameraView.h"

#include <imgui.h>

#include "../data/Data.h"
CameraView::CameraView(Data *data) : View(data)
{
    m_data = data;
    m_camera = m_data->getCamera();
}

void CameraView::drawUI()
{
    ImGui::Begin("Camera:");
    ImGui::Text("origin = %5.2f \t %5.2f \t %5.2f", m_camera->m_origin[0], m_camera->m_origin[1], m_camera->m_origin[2]);
    ImGui::Text("front = %5.2f \t %5.2f \t %5.2f", m_camera->m_front[0], m_camera->m_front[1], m_camera->m_front[2]);
    ImGui::Text("up = %5.2f \t %5.2f \t %5.2f", m_camera->m_worldUp[0], m_camera->m_worldUp[1], m_camera->m_worldUp[2]);
    ImGui::Text("fov = %5.2f", m_camera->m_fov);

    if (ImGui::Button("reset (R)"))
    {
        m_camera->reset();
    }
    ImGui::End();
}