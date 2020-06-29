#include <pybind11/pybind11.h>
#include <rkguide/rkguide.h>
#include <rkguide/vmm/vmm.h>
#include <rkguide/vmm/vmmFactory.h>

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

int add(int i, int j) {
    return i + j;
}


float Vector2_get (embree::Vec2<float> &v, int index) { return v[index]; }
void Vector2_set (embree::Vec2<float> &v, int index, float value) { v[index] = value; }
const std::string Vector2_toString (embree::Vec2<float>& v){
	std::ostringstream oss;
	oss << "[" << v[0] << " " << v[1] << "]" << std::endl;
	return oss.str();
}


float Vector3_get (embree::Vec3<float> &v, int index) { return v[index]; }
void Vector3_set (embree::Vec3<float> &v, int index, float value) { v[index] = value; }
const std::string Vector3_toString (embree::Vec3<float>& v){
	std::ostringstream oss;
	oss << "[" << v[0] << " " << v[1] << " " << v[2]<< "]" << std::endl;
	return oss.str();
}

void test(){
    rkguide::VonMisesFisherMixture<4,32> vmm;
    std::cout << vmm.toString() << std::endl;
}

rkguide::Vector3 sphericalDirection(const float &theta, const float &phi)
{
    return rkguide::sphericalDirection(theta, phi);
}

namespace py = pybind11;

PYBIND11_MODULE(pyrkguide, m) {
    m.doc() = R"pbdoc(
        Python module for Intel's rkguide
        -----------------------
        .. currentmodule:: rkguide
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("test", &test, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");
/*
    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");
*/

m.def("toSphericalCoordinates", &rkguide::toSphericalCoordinates);

//m.def("sphericalDirection", overload_cast_<float, float>()(&rkguide::sphericalDirection));
m.def("sphericalDirection", &sphericalDirection);
m.def("squareToUniformSphere", &rkguide::squareToUniformSphere);


py::class_< embree::Vec2<float> >(m, "Vector2")
    .def(py::init<>())
    .def(py::init<float>())
    .def(py::init<float, float>())
    .def("__getitem__", &Vector2_get)
    .def("__setitem__", &Vector2_set)
    .def("__repr__", &Vector2_toString);

py::class_< embree::Vec3<float> >(m, "Vector3")
    .def(py::init<>())
    .def(py::init<float>())
    .def(py::init<float, float, float>())
    .def("__getitem__", &Vector3_get)
    .def("__setitem__", &Vector3_set)
    .def("__repr__", &Vector3_toString);

py::class_< rkguide::VonMisesFisherMixture<4,32> >(m, "VMM32v4")
    .def(py::init<>())
    .def("pdf", &rkguide::VonMisesFisherMixture<4,32>::pdf)
    .def("sample", &rkguide::VonMisesFisherMixture<4,32>::sample)
    .def("uniformInit", &rkguide::VonMisesFisherMixture<4,32>::uniformInit)
    .def("__repr__", &rkguide::VonMisesFisherMixture<4,32>::toString);


py::class_< rkguide::VonMisesFisherMixture<8,32> >(m, "VMM32v8")
    .def(py::init<>())
    .def("__repr__", &rkguide::VonMisesFisherMixture<8,32>::toString);

py::class_< rkguide::VonMisesFisherMixture<16,32> >(m, "VMM32v16")
    .def(py::init<>())
    .def("__repr__", &rkguide::VonMisesFisherMixture<16,32>::toString);

py::class_< rkguide::VonMisesFisherFactory<4, 32> >(m, "VMMFactory32v4")
    .def(py::init<>())
    .def("InitUniformVMM", &rkguide::VonMisesFisherFactory<4,32>::InitUniformVMM);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}