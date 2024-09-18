#include "ViewportView.h"

ViewportView::ViewportView(Data *data, FrameBuffer *framebuffer) : View(data)
{
    init(framebuffer);
}

void ViewportView::init(FrameBuffer *framebuffer)
{
    m_framebuffer = framebuffer;
    firstRescale = true;
}

void ViewportView::draw() {}
void ViewportView::drawUI()
{
    ImGui::Begin("Viewport");
    m_viewportPanelSize = ImGui::GetContentRegionAvail();
    m_viewportMinRegion = ImGui::GetWindowContentRegionMin();
    m_viewportMaxRegion = ImGui::GetWindowContentRegionMax();
    m_viewportOffset = ImGui::GetWindowPos();

    if (m_width != m_viewportPanelSize[0] || m_height != m_viewportPanelSize[1])
    {
        m_width = m_viewportPanelSize[0];
        m_height = m_viewportPanelSize[1];
        if (firstRescale)
        {
            m_framebuffer->init(m_width, m_height);
        }
        else
        {
            m_framebuffer->rescale(m_width, m_height);
        }

        Camera *camera = m_data->getCamera();
        camera->updateProjection(m_width, m_height, camera->m_fov);
    }

    ImGui::Image((void *)(intptr_t)m_framebuffer->m_texture_id, ImVec2(m_width, m_height), ImVec2(0, 1), ImVec2(1, 0));
    if (ImGui::IsWindowHovered())
    {
        ImGuiIO &io = ImGui::GetIO();
        io.WantCaptureMouse = false;
        // io.WantCaptureKeyboard = false;
    }
    ImGui::End();
}