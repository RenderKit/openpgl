#pragma once

#include "View.h"

#include <imgui.h>

#include "../FrameBuffer.h"

struct ViewportView: public View {

    ViewportView(Data* data, FrameBuffer* framebuffer);

    void dataUpdated() override {};
    void draw() override;
    void drawUI()override;

private:
    void init(FrameBuffer* framebuffer);
    
    FrameBuffer* m_framebuffer;
    float m_width = 0.f;
    float m_height = 0.f;
    bool firstRescale = false;
public:
    ImVec2 m_viewportPanelSize;// = ImGui::GetContentRegionAvail();
    ImVec2 m_viewportMinRegion;// = ImGui::GetWindowContentRegionMin();
    ImVec2 m_viewportMaxRegion;// = ImGui::GetWindowContentRegionMax();
    ImVec2 m_viewportOffset; // = ImGui::GetWindowPos();
};