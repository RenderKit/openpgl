#include "Data.h"

#include <ImGuiFileDialog.h>
#include <imgui.h>

#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

std::string get_path(std::string filename)
{
    return filename.substr(0, filename.find_last_of("/\\"));
}

void Data::init()
{
    m_device = new openpgl::cpp::Device(PGL_DEVICE_TYPE_CPU_4);
}

void Data::load(std::string filename)
{
    std::ifstream f(filename);
    json data = json::parse(f);
    std::string rootDir = "";
    if(data.contains("root"))
        rootDir = data["root"];
    else {
        rootDir = get_path(filename);
    }
    std::string cameraFile = data["camera"];
    std::string fieldFile = data["field"];
    std::string samplesFile = data["samples"];

    loadCamera(rootDir + "/" + cameraFile);
    loadField(rootDir + "/" + fieldFile);
    loadSamples(rootDir + "/" + samplesFile);
    m_updated = true;
}

openpgl::cpp::Field *Data::loadField(std::string filename)
{
    m_field = new openpgl::cpp::Field(m_device, filename);
    m_ssd = new openpgl::cpp::SurfaceSamplingDistribution(m_field);
    m_vsd = new openpgl::cpp::VolumeSamplingDistribution(m_field);
    return m_field;
}

openpgl::cpp::SampleStorage *Data::loadSamples(std::string filename)
{
    std::cout << "ZeroValueSample: " << sizeof(openpgl::cpp::ZeroValueSampleData) << std::endl;
    m_sampleStorage = new openpgl::cpp::SampleStorage(filename);
    return m_sampleStorage;
}

openpgl::cpp::SurfaceSamplingDistribution *Data::getSurfaceSamplingDistribution(glm::vec3 pos)
{
    pgl_point3f pglP;
    pglP.x = pos.x;
    pglP.y = pos.y;
    pglP.z = pos.z;
    float sample = -1.0f;
    m_ssd->Init(m_field, pglP, sample);
    return m_ssd;
}

openpgl::cpp::VolumeSamplingDistribution *Data::getVolumeSamplingDistribution(glm::vec3 pos)
{
    pgl_point3f pglP;
    pglP.x = pos.x;
    pglP.y = pos.y;
    pglP.z = pos.z;
    float sample = -1.0f;
    m_vsd->Init(m_field, pglP, sample);
    return m_vsd;
}

Camera *Data::loadCamera(std::string filename)
{
    std::ifstream f(filename);
    json data = json::parse(f);
    json jcam = data["camera"];

    glm::vec3 origin = glm::vec3(jcam["origin"][0], jcam["origin"][1], jcam["origin"][2]);
    glm::vec3 target = glm::vec3(jcam["target"][0], jcam["target"][1], jcam["target"][2]);
    glm::vec3 up = glm::vec3(jcam["up"][0], jcam["up"][1], jcam["up"][2]);

    float fov = jcam["fov"];

    float sensitivity = jcam["sensitivitySpatial"];
    m_camera.init(origin, target, up, fov);
    // m_camera.m_fov = fov;
    m_camera.m_sensitivitySpatial = sensitivity;
    return &m_camera;
}

Camera *Data::getCamera()
{
    return &m_camera;
}

openpgl::cpp::Field *Data::getField()
{
    return m_field;
}

openpgl::cpp::SampleStorage *Data::getSamples()
{
    return m_sampleStorage;
}

void Data::registerView(View *view)
{
    m_views.push_back(view);
}

void Data::update()
{
    m_camera.update();
    if (m_updated)
    {
        for (auto view : m_views)
        {
            view->dataUpdated();
        }
        m_updated = false;
    }
}