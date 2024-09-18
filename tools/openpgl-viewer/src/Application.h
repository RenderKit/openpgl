#include "data/Data.h"
#include "math/Camera.h"
#include "views/View.h"

#include <imgui.h>
#include <glm/vec2.hpp>

#include <string>
#include <vector>

struct GLFWwindow;

struct Application{

protected:
    Application();

    static Application* appPtr;
public:

    static Application* GetInstance();

    void loadScene(std::string sceneFile);

    int run();

    void drawUI();

    void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

    void mouseCallback(GLFWwindow* window, double xposIn, double yposIn);

    void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

private:
    void processInput(GLFWwindow *window);

    void setStyle();

private:

    GLFWwindow* m_window;

    Data m_data;
    Camera* m_camera;

    std::vector<View*> m_views;

    bool m_firstMouse = true;
    float m_lastX =  0.f;
    float m_lastY =  0.f;

    float m_deltaTime = 0.0f;	// time between current frame and last frame
    float m_lastFrame = 0.0f; 

    bool m_loadScenePressed = false;  

    bool m_mouseDragOngoing = false;
    bool m_needToHandleRightPress = false;

    ImVec2 m_viewportPanelSize;
    ImVec2 m_viewportMinRegion;
    ImVec2 m_viewportMaxRegion;
    ImVec2 m_viewportOffset;  

    glm::vec2 m_mXY;

    std::string m_lastFilePath {"."};
};