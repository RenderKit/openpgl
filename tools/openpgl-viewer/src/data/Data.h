#pragma once

#include <openpgl/cpp/OpenPGL.h>

#include <glm/vec3.hpp>
#include <vector>

#include "../math/Camera.h"
#include "../views/View.h"

struct Data
{
    void init();

    void update();

    void load(std::string filename);

    openpgl::cpp::Field *loadField(std::string filename);

    openpgl::cpp::SampleStorage *loadSamples(std::string filename);

    openpgl::cpp::SurfaceSamplingDistribution *getSurfaceSamplingDistribution(glm::vec3 pos);

    openpgl::cpp::VolumeSamplingDistribution *getVolumeSamplingDistribution(glm::vec3 pos);

    Camera *loadCamera(std::string filename);

    Camera *getCamera();

    openpgl::cpp::Field *getField();

    openpgl::cpp::SampleStorage *getSamples();

    void registerView(View *view);

   private:
    bool m_updated = false;
    openpgl::cpp::Device *m_device{nullptr};
    openpgl::cpp::Field *m_field{nullptr};
    openpgl::cpp::SampleStorage *m_sampleStorage{nullptr};

    openpgl::cpp::SurfaceSamplingDistribution *m_ssd{nullptr};
    openpgl::cpp::VolumeSamplingDistribution *m_vsd{nullptr};

    Camera m_camera;

    std::string m_lastFilePath{"."};

    std::vector<View *> m_views;
};