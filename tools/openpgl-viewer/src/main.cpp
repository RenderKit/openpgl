#include "Application.h"
#include <string>

int main(int argc, char** argv) {

    Application* app = Application::GetInstance();

    if (argc == 2)
    {
        std::string sceneFile = std::string(argv[1]);
        app->loadScene(sceneFile);
    }

    return app->run();

}
