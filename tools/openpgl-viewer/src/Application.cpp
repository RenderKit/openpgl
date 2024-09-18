#include "Application.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <ImGuiFileDialog.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "FrameBuffer.h"
#include "imgui_internal.h"  // for DockBuilder API
#include "views/CameraView.h"
#include "views/GuidingFieldView.h"
#include "views/SampleView.h"
#include "views/SamplingDistributionView.h"
#include "views/ViewportView.h"

Application *Application::appPtr = nullptr;

static void glfw_error_callback(int error, const char *description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    Application *app = Application::GetInstance();
    app->scrollCallback(window, xoffset, yoffset);
}

void Application::scrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;

    m_camera->m_fov -= (float)yoffset;
    if (m_camera->m_fov < 1.0f)
        m_camera->m_fov = 1.0f;
    if (m_camera->m_fov > 120.0f)
        m_camera->m_fov = 120.0f;
}

void mouse_callback(GLFWwindow *window, double xposIn, double yposIn)
{
    Application *app = Application::GetInstance();
    app->mouseCallback(window, xposIn, yposIn);
}

void Application::mouseCallback(GLFWwindow *window, double xposIn, double yposIn)
{
    // convert to viewport positions
    xposIn -= m_viewportOffset[0] + m_viewportMinRegion[0];
    yposIn -= m_viewportOffset[1] + m_viewportMinRegion[1];

    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        if (!m_mouseDragOngoing)
        {
            m_lastX = xpos;
            m_lastY = ypos;
            m_mouseDragOngoing = true;
            return;
        }

        float xoffset = xpos - m_lastX;
        float yoffset = m_lastY - ypos;  // reversed since y-coordinates go from bottom to top
        m_lastX = xpos;
        m_lastY = ypos;

        float sensitivity = 0.1f;  // change this value to your liking
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        glm::vec3 front;
        switch (m_camera->m_upType)
        {
            case Camera::X_UP:
            {
                // rot = glm::rotate(rot, glm::radians(90.f), glm::vec3(0,1,0));
                break;
            }
            case Camera::Y_UP:
            {
                m_camera->m_yaw += xoffset;
                m_camera->m_pitch -= yoffset;
                front.x = cos(glm::radians(m_camera->m_yaw)) * sin(glm::radians(m_camera->m_pitch));
                front.y = cos(glm::radians(m_camera->m_pitch));
                front.z = sin(glm::radians(m_camera->m_yaw)) * sin(glm::radians(m_camera->m_pitch));
                break;
            }
            case Camera::Z_UP:
            {
                m_camera->m_yaw -= xoffset;
                m_camera->m_pitch -= yoffset;
                front.x = cos(glm::radians(m_camera->m_yaw)) * sin(glm::radians(m_camera->m_pitch));
                front.y = sin(glm::radians(m_camera->m_yaw)) * sin(glm::radians(m_camera->m_pitch));
                front.z = cos(glm::radians(m_camera->m_pitch));
                break;
            }
        }
        front = glm::normalize(front);
        m_camera->m_front = glm::vec3(front.x, front.y, front.z);
    }
    else
    {
        m_mouseDragOngoing = false;
    }
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    Application *app = Application::GetInstance();
    app->mouseButtonCallback(window, button, action, mods);
}

void Application::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
    {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        // convert to viewport positions
        xpos -= m_viewportOffset[0] + m_viewportMinRegion[0];
        ypos -= m_viewportOffset[1] + m_viewportMinRegion[1];

        m_mXY = glm::vec2(xpos, ypos);
        m_needToHandleRightPress = true;
    }
}

Application *Application::GetInstance()
{
    if (appPtr == nullptr)
    {
        appPtr = new Application();
    }
    return appPtr;
}

void Application::processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float cameraSpeed = static_cast<float>(m_camera->m_sensitivitySpatial * m_deltaTime);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        m_camera->m_origin += cameraSpeed * m_camera->m_front;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        m_camera->m_origin -= cameraSpeed * m_camera->m_front;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        m_camera->m_origin -= glm::normalize(glm::cross(m_camera->m_front, m_camera->m_worldUp)) * cameraSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        m_camera->m_origin += glm::normalize(glm::cross(m_camera->m_front, m_camera->m_worldUp)) * cameraSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
    {
        m_camera->reset();
    }
}

Application::Application()
{
    m_data.init();
}

void Application::loadScene(std::string sceneFile)
{
    if (sceneFile != "")
        m_data.load(sceneFile);
}

void Application::drawUI()
{
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Load Scene", nullptr, &m_loadScenePressed))
            {
                ImGui::OpenPopup("Load Scene");
            }
            if (ImGui::MenuItem("Exit", "Esc"))
            {
                glfwSetWindowShouldClose(m_window, true);
            }

            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
    if (m_loadScenePressed)
    {
        ImGui::OpenPopup("Load Scene");
        m_loadScenePressed = false;
    }

    if (ImGui::BeginPopup("Load Scene"))
    {
        ImGuiFileDialog::Instance()->OpenDialog("ChooseOPGLVFile", "Choose File", ".opglv,.*", m_lastFilePath);
        if (ImGuiFileDialog::Instance()->Display("ChooseOPGLVFile"))
        {
            // action if OK
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
                // std::cout << "filePathName: " << filePathName << std::endl;
                // std::cout << "filePath:" << filePath << std::endl;
                m_lastFilePath = filePath;
                // action
                m_data.load(filePathName);
            }

            ImGui::CloseCurrentPopup();
            ImGuiFileDialog::Instance()->Close();
        }
        ImGui::EndPopup();
    }
}

int Application::run()
{
    float width = 1920;
    float height = 1080;

    m_camera = m_data.getCamera();

    // Initialize GLFW
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Create a GLFW window
    m_window = glfwCreateWindow(width, height, "Open PGL Viewer", NULL, NULL);
    if (!m_window)
    {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(m_window);
    glfwSetCursorPosCallback(m_window, mouse_callback);
    glfwSetScrollCallback(m_window, scroll_callback);
    glfwSetMouseButtonCallback(m_window, mouse_button_callback);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    // ImGui::StyleColorsDark();
    setStyle();
    ImGuiStyle &style = ImGui::GetStyle();
    ImVec4 windowBgColor = style.Colors[ImGuiCol_WindowBg];
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Enable docking
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    bool err = glewInit() != GLEW_OK;

    if (err)
    {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return 1;
    }

    FrameBuffer framebuffer;

    ViewportView *vpv = new ViewportView(&m_data, &framebuffer);
    GuidingFieldView *gfv = new GuidingFieldView(&m_data);
    SamplingDistributionView *ssdv = new SamplingDistributionView(&m_data, SamplingDistributionView::ESURFACE_DISTRIBUTION);
    SamplingDistributionView *vsdv = new SamplingDistributionView(&m_data, SamplingDistributionView::EVOLUME_DISTRIBUTION);

    m_views.push_back(vpv);
    m_views.push_back(new SampleView(&m_data));
    m_views.push_back(gfv);
    m_views.push_back(ssdv);
    m_views.push_back(vsdv);
    m_views.push_back(new CameraView(&m_data));

    // Main loop
    while (!glfwWindowShouldClose(m_window))
    {
        glfwPollEvents();

        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        m_deltaTime = currentFrame - m_lastFrame;
        m_lastFrame = currentFrame;

        processInput(m_window);

        glClearColor(windowBgColor.x, windowBgColor.y, windowBgColor.z, windowBgColor.w);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (m_needToHandleRightPress)
        {
            Ray ray = m_camera->getRay(m_mXY.x, m_mXY.y);
            m_needToHandleRightPress = false;
            float cX, cY;

            float closestDist = std::numeric_limits<float>::max();
            int closestIdx = -1;

            for (int n = 0; n < gfv->m_cachePositions.size(); n++)
            {
                float pdist;
                ClosestDistance(ray, gfv->m_cachePositions[n], pdist);
                if (pdist < closestDist)
                {
                    closestDist = pdist;
                    closestIdx = n;
                }
            }
            ssdv->update(gfv->m_cachePositions[closestIdx]);
            m_camera->getPixel(gfv->m_cachePositions[closestIdx], cX, cY);

            // std::cout << "x: " << mXY.x << "\ty: " << mXY.y << "\tcX: " << cX << "\t cY: " << cY << std::endl;
            // std::cout << "idx: " << closestIdx << "\t pos: " << gfv.m_cachePositions[closestIdx].x << "\t" << gfv.m_cachePositions[closestIdx].y << "\t" <<
            // gfv.m_cachePositions[closestIdx].z << std::endl;
        }

        if (0)
        {
            ImGui::DockSpaceOverViewport();
        }
        else
        {
            // setup docking space
            static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;
            // dockspace_flags = ImGuiDockNodeFlags_AutoHideTabBar;
            dockspace_flags |= ImGuiDockNodeFlags_PassthruCentralNode;
            ImGuiViewport *iviewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(iviewport->WorkPos);
            ImGui::SetNextWindowSize(iviewport->WorkSize);
            ImGui::SetNextWindowViewport(iviewport->ID);
            ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
            window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
            window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus /* | ImGuiWindowFlags_NoNavFocus*/;

            if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
                window_flags |= ImGuiWindowFlags_NoBackground;

            // dockspace_flags &= ~ImGuiDockNodeFlags_PassthruCentralNode;
            // ImGui::DockSpaceOverViewport(viewport, dockspace_flags);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
            ImGui::Begin("DockSpace1", nullptr, window_flags);
            ImGui::PopStyleVar(3);

            ImGuiID dockspace_id = ImGui::GetID("dockspace1");
            ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);

            static bool once = false;
            if (!once)
            {
                once = true;

                ImGui::DockBuilderRemoveNode(dockspace_id);
                ImGui::DockBuilderAddNode(dockspace_id, dockspace_flags | ImGuiDockNodeFlags_DockSpace);
                ImGui::DockBuilderSetNodeSize(dockspace_id, iviewport->Size);

                const float marginLeft = 0.15f;
                const float marginRight = 0.25f;
                ImGuiID center_dock;
                // auto center_dock = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Down, 0.06f, nullptr, &dockspace_id);
                ImGuiID left_dock = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, marginLeft, nullptr, &center_dock);
                ImGuiID right_dock = ImGui::DockBuilderSplitNode(center_dock, ImGuiDir_Right, marginRight * (1.0f + marginLeft), nullptr, &center_dock);
                // ImGuiID top_dock = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Up, 1.0f, nullptr, &dockspace_id);

                ImGuiID right_bottom_dock;
                ImGuiID right_top_dock = ImGui::DockBuilderSplitNode(right_dock, ImGuiDir_Up, 0.5f, nullptr, &right_bottom_dock);

                ImGuiID left_top_dock = ImGui::DockBuilderSplitNode(left_dock, ImGuiDir_Up, 0.125f, nullptr, &left_dock);
                ImGuiID left_center_dock = ImGui::DockBuilderSplitNode(left_dock, ImGuiDir_Up, 0.125f, nullptr, &left_dock);

                ImGui::DockBuilderDockWindow("Camera:", left_top_dock);
                ImGui::DockBuilderDockWindow("Samples:", left_center_dock);
                ImGui::DockBuilderDockWindow("Guiding Field:", left_dock);
                ImGui::DockBuilderDockWindow("Viewport", center_dock);
                ImGui::DockBuilderDockWindow("SurfaceSamplingDistribution:", right_top_dock);
                ImGui::DockBuilderDockWindow("VolumeSamplingDistribution:", right_bottom_dock);
                ImGui::DockBuilderFinish(dockspace_id);
            }
            ImGui::End();
        }

        drawUI();
        for (auto view : m_views)
        {
            view->drawUI();
        }

        m_viewportPanelSize = vpv->m_viewportPanelSize;
        m_viewportMinRegion = vpv->m_viewportMinRegion;
        m_viewportMaxRegion = vpv->m_viewportMaxRegion;
        m_viewportOffset = vpv->m_viewportOffset;

        m_data.update();

        // Main Window
        ImGui::Render();

        for (auto view : m_views)
        {
            view->draw();
        }

        framebuffer.bind();
        framebuffer.clear();
        for (auto view : m_views)
        {
            view->drawViewport();
        }
        framebuffer.unbind();
        framebuffer.draw();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(m_window);
    }

    // Cleanup
    for (auto view : m_views)
    {
        delete view;
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(m_window);
    glfwTerminate();

    return 0;
}

void Application::setStyle()
{
    ImGuiStyle &style = ImGui::GetStyle();
    style.Colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
    style.Colors[ImGuiCol_ChildBg] = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
    style.Colors[ImGuiCol_PopupBg] = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
    style.Colors[ImGuiCol_Border] = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
    style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    style.Colors[ImGuiCol_FrameBg] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
    style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);
    style.Colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
    style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
    style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
    style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
    style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
    style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
    style.Colors[ImGuiCol_CheckMark] = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
    style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
    style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.08f, 0.50f, 0.72f, 1.00f);
    style.Colors[ImGuiCol_Button] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
    style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);
    style.Colors[ImGuiCol_Separator] = style.Colors[ImGuiCol_Border];
    style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.41f, 0.42f, 0.44f, 1.00f);
    style.Colors[ImGuiCol_SeparatorActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
    style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.29f, 0.30f, 0.31f, 0.67f);
    style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
    style.Colors[ImGuiCol_Tab] = ImVec4(0.08f, 0.08f, 0.09f, 0.83f);
    style.Colors[ImGuiCol_TabHovered] = ImVec4(0.33f, 0.34f, 0.36f, 0.83f);
    style.Colors[ImGuiCol_TabActive] = ImVec4(0.23f, 0.23f, 0.24f, 1.00f);
    style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
    style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
    style.Colors[ImGuiCol_DockingPreview] = ImVec4(0.26f, 0.59f, 0.98f, 0.70f);
    style.Colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    style.Colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
    style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
    style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
    style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
    style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
    style.Colors[ImGuiCol_DragDropTarget] = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
    style.Colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
    style.GrabRounding = style.FrameRounding = 2.3f;
}