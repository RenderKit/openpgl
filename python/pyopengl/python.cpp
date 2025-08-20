#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <openpgl/cpp/OpenPGL.h>

#include <cmath>
#include <algorithm>

namespace py = pybind11;


pgl_vec3f sphericalDirection(const float &theta, const float &phi)
{
    const float cosTheta = std::cos(theta);
    const float sinTheta = std::sin(theta);
    const float cosPhi = std::cos(phi);
    const float sinPhi = std::sin(phi);

    return openpgl::cpp::Vector3( sinTheta * cosPhi,
                    sinTheta * sinPhi,
                    cosTheta);

};

pgl_vec2f toSphericalCoordinates(const pgl_vec3f& dir)
{
    pgl_vec2f result = openpgl::cpp::Vector2(
        std::acos(dir.z),
        std::atan2(dir.y, dir.x)
    );
    if (result.y < 0)
        result.y += 2.0f*(float)M_PI;
    return result;
};

PYBIND11_MODULE(pyopenpgl, m) {

    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: libguiding
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("sphericalDirection", &sphericalDirection);

    m.def("toSphericalCoordinates", &toSphericalCoordinates);

/*
    py::class_<pgl_point3f>(m, "Point3")
        .def(py::init<>())
        .def_readwrite("x", &pgl_point3f::x)
        .def_readwrite("y", &pgl_point3f::y)
        .def_readwrite("z", &pgl_point3f::z);
*/
    py::class_<pgl_vec3f>(m, "Vector3")
        .def(py::init<>())
        .def_readwrite("x", &pgl_vec3f::x)
        .def_readwrite("y", &pgl_vec3f::y)
        .def_readwrite("z", &pgl_vec3f::z);

    py::class_<pgl_vec2f>(m, "Vector2")
        .def(py::init<>())
        .def_readwrite("x", &pgl_vec2f::x)
        .def_readwrite("y", &pgl_vec2f::y);

    py::enum_<PGL_DEVICE_TYPE>(m, "PGL_DEVICE_TYPE")
        .value("PGL_DEVICE_TYPE_CPU_4", PGL_DEVICE_TYPE::PGL_DEVICE_TYPE_CPU_4 )
        .value("PGL_DEVICE_TYPE_CPU_8",  PGL_DEVICE_TYPE::PGL_DEVICE_TYPE_CPU_8  )
        //.value("PGL_DEVICE_TYPE_CPU_16",  PGL_DEVICE_TYPE::PGL_DEVICE_TYPE_CPU_16  )
        .export_values();

    py::class_< openpgl::cpp::Device >(m, "Device")
        .def(py::init<PGL_DEVICE_TYPE>());
/*
    py::class_< PGLRange >(m, "Range")
        .def(py::init<>())
        .def_readwrite("start", &PGLRange::start)
        .def_readwrite("end", &PGLRange::end);
*/
    py::class_< openpgl::cpp::Field >(m, "Field")
        .def(py::init<openpgl::cpp::Device*, const openpgl::cpp::FieldConfig& >())
        //.def(py::init<openpgl::cpp::Device*, PGLFieldArguments >())
        .def(py::init<openpgl::cpp::Device*, const std::string& >())
        .def("Store", &openpgl::cpp::Field::Store)
        .def("SetSceneBounds", &openpgl::cpp::Field::SetSceneBounds)
        .def("GetSceneBounds", &openpgl::cpp::Field::GetSceneBounds)
        .def("Update", &openpgl::cpp::Field::Update)
        .def("Reset", &openpgl::cpp::Field::Reset)
        .def("GetIteration", &openpgl::cpp::Field::GetIteration)
        //.def("GetSurfaceSampleRange", &openpgl::cpp::Field::GetSurfaceSampleRange)
        //.def("GetVolumeSampleRange", &openpgl::cpp::Field::GetVolumeSampleRange)
        //.def("GetTotalSPP", &openpgl::cpp::Field::GetTotalSPP)
        .def("Validate", &openpgl::cpp::Field::Validate);

    py::class_< openpgl::cpp::SampleStorage >(m, "SampleStorage")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def("Store", &openpgl::cpp::SampleStorage::Store)
        .def("AddSample", &openpgl::cpp::SampleStorage::AddSample)
        .def("AddSamples", &openpgl::cpp::SampleStorage::AddSamples)
        .def("Reserve", &openpgl::cpp::SampleStorage::Reserve)
        .def("Clear", &openpgl::cpp::SampleStorage::Clear)
        .def("GetSizeSurface", &openpgl::cpp::SampleStorage::GetSizeSurface)
        .def("GetSizeVolume", &openpgl::cpp::SampleStorage::GetSizeVolume)
        .def("GetSampleSurface", &openpgl::cpp::SampleStorage::GetSampleSurface)
        .def("GetSampleVolume", &openpgl::cpp::SampleStorage::GetSampleVolume);

    py::class_< openpgl::cpp::SampleData >(m, "SampleData")
        .def(py::init<>())
        .def_readwrite("position", &PGLSampleData::position)
        .def_readwrite("direction", &PGLSampleData::direction)
        .def_readwrite("weight", &PGLSampleData::weight)
        .def_readwrite("pdf", &PGLSampleData::pdf)
        .def_readwrite("distance", &PGLSampleData::distance)
        .def_readwrite("flags", &PGLSampleData::flags);

    py::class_< openpgl::cpp::SurfaceSamplingDistribution >(m, "SurfaceSamplingDistribution")
        .def(py::init<const openpgl::cpp::Field*>())
        .def("Init", &openpgl::cpp::SurfaceSamplingDistribution::Init)
        .def("Clear", &openpgl::cpp::SurfaceSamplingDistribution::Clear)
        .def("Sample", &openpgl::cpp::SurfaceSamplingDistribution::Sample)
        .def("PDF", &openpgl::cpp::SurfaceSamplingDistribution::PDF)
        .def("SamplePDF", &openpgl::cpp::SurfaceSamplingDistribution::SamplePDF)
        .def("SupportsApplyCosineProduct", &openpgl::cpp::SurfaceSamplingDistribution::SupportsApplyCosineProduct)
        .def("ApplyCosineProduct", &openpgl::cpp::SurfaceSamplingDistribution::ApplyCosineProduct)
        .def("GetId", &openpgl::cpp::SurfaceSamplingDistribution::GetId)
        //.def("GetNumComponents", &openpgl::cpp::SurfaceSamplingDistribution::GetNumComponents)
        //.def("GetVarianceEstimateOfComponent", &openpgl::cpp::SurfaceSamplingDistribution::GetVarianceEstimateOfComponent)
        //.def("IncomingRadiance", &openpgl::cpp::SurfaceSamplingDistribution::IncomingRadiance)
        .def("Validate", &openpgl::cpp::SurfaceSamplingDistribution::Validate);
        //.def("GetRegion", &openpgl::cpp::SurfaceSamplingDistribution::GetRegion);


    py::class_< openpgl::cpp::VolumeSamplingDistribution >(m, "VolumeSamplingDistribution")
        .def(py::init<const openpgl::cpp::Field*>())
        .def("Init", &openpgl::cpp::VolumeSamplingDistribution::Init)
        .def("Clear", &openpgl::cpp::VolumeSamplingDistribution::Clear)
        .def("Sample", &openpgl::cpp::VolumeSamplingDistribution::Sample)
        .def("PDF", &openpgl::cpp::VolumeSamplingDistribution::PDF)
        .def("SamplePDF", &openpgl::cpp::VolumeSamplingDistribution::SamplePDF)
        .def("SupportsApplySingleLobeHenyeyGreensteinProduct", &openpgl::cpp::VolumeSamplingDistribution::SupportsApplySingleLobeHenyeyGreensteinProduct)
        .def("ApplySingleLobeHenyeyGreensteinProduct", &openpgl::cpp::VolumeSamplingDistribution::ApplySingleLobeHenyeyGreensteinProduct)
        .def("Validate", &openpgl::cpp::VolumeSamplingDistribution::Validate);
        //.def("GetRegion", &openpgl::cpp::VolumeSamplingDistribution::GetRegion);

#ifdef VERSION_INFO
    m.attr("__version__") = STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}