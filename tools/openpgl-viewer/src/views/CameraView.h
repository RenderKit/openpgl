#pragma once

#include "../math/Camera.h"

#include "View.h"

struct CameraView: public View{

    CameraView(Data* data);
    void dataUpdated() override {};
    void drawUI() override;

private:
    Camera* m_camera;
};