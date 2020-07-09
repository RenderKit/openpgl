#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <rkguide/rkguide.h>
#include <rkguide/data/DirectionalSampleData.h>
#include <rkguide/vmm/VMM.h>
#include <rkguide/vmm/VMMFactory.h>
#include <rkguide/vmm/WeightedEMVMMFactory.h>
#include <rkguide/vmm/VMMChiSquareComponentMerger.h>
#include <rkguide/vmm/VMMChiSquareComponentSplitter.h>

namespace py = pybind11;

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

int add(int i, int j) {
    return i + j;
}

static void DirectionalSampleData_storeDirectionalSampleData(const std::string& fileName, const py::list &listParticles){
	std::vector<rkguide::DirectionalSampleData> dataPoints = listParticles.cast<std::vector<rkguide::DirectionalSampleData>>();
	rkguide::StoreDirectionalSampleData(fileName, dataPoints.data(), dataPoints.size());
}

static py::list DirectionalSampleData_loadDirectionalSampleData(const std::string& fileName){
    size_t numData;
    rkguide::DirectionalSampleData* data = rkguide::LoadDirectionalSampleData(fileName, numData);
    std::vector<rkguide::DirectionalSampleData> dataPoints;
    for( size_t n = 0; n < numData; n++)
    {
        dataPoints.push_back(data[n]);
    }
	py::list list = py::cast(dataPoints);
	return list;
}

template< typename WeightedEMVMMFactory, typename VMM>
static void  WeightedEMVMMFactory_fit( WeightedEMVMMFactory *vmmFactory, VMM &model, const size_t &K, typename WeightedEMVMMFactory::SufficientStatisitcs &stats, const py::list &listParticles, typename WeightedEMVMMFactory::Configuration &cfg)
{
    std::vector<rkguide::DirectionalSampleData> dataPoints = listParticles.cast<std::vector<rkguide::DirectionalSampleData>>();
	//rkguide::StoreDirectionalSampleData(fileName, dataPoints.data(), dataPoints.size());
    vmmFactory->fitMixture(model, K, stats, dataPoints.data(), dataPoints.size(), cfg);
}


template< typename WeightedEMVMMFactory, typename VMM>
static void  WeightedEMVMMFactory_update( WeightedEMVMMFactory *vmmFactory, VMM &model, typename WeightedEMVMMFactory::SufficientStatisitcs &stats, const py::list &listParticles, typename WeightedEMVMMFactory::Configuration &cfg)
{
    std::vector<rkguide::DirectionalSampleData> dataPoints = listParticles.cast<std::vector<rkguide::DirectionalSampleData>>();
	//rkguide::StoreDirectionalSampleData(fileName, dataPoints.data(), dataPoints.size());
    vmmFactory->updateMixture(model, stats, dataPoints.data(), dataPoints.size(), cfg);
}

template< typename VMMChiSquareComponentSplitter, typename VMM>
static void  VMMChiSquareComponentSplitter_CalculateSplitStatistics( VMMChiSquareComponentSplitter *vmmSplitter, VMM &model, typename VMMChiSquareComponentSplitter::ComponentSplitStatistics &splitStats, const py::list &listParticles)
{
    std::vector<rkguide::DirectionalSampleData> dataPoints = listParticles.cast<std::vector<rkguide::DirectionalSampleData>>();
    float mcEstimate = 0.0f;
    for (size_t n = 0; n < dataPoints.size(); n++)
    {
        mcEstimate += dataPoints[n].weight;
    }
    mcEstimate /= (float) dataPoints.size();

	//rkguide::StoreDirectionalSampleData(fileName, dataPoints.data(), dataPoints.size());
    vmmSplitter->CalculateSplitStatistics(model, splitStats, mcEstimate, dataPoints.data(), dataPoints.size());
}

template< typename VMMChiSquareComponentSplitter, typename VMM>
static void  VMMChiSquareComponentSplitter_UpdateSplitStatistics( VMMChiSquareComponentSplitter *vmmSplitter, VMM &model, typename VMMChiSquareComponentSplitter::ComponentSplitStatistics &splitStats, const py::list &listParticles)
{
    std::vector<rkguide::DirectionalSampleData> dataPoints = listParticles.cast<std::vector<rkguide::DirectionalSampleData>>();
    float mcEstimate = 0.0f;
    for (size_t n = 0; n < dataPoints.size(); n++)
    {
        mcEstimate += dataPoints[n].weight;
    }
    mcEstimate /= (float) dataPoints.size();

	//rkguide::StoreDirectionalSampleData(fileName, dataPoints.data(), dataPoints.size());
    vmmSplitter->UpdateSplitStatistics(model, splitStats, mcEstimate, dataPoints.data(), dataPoints.size());
}

template< typename VMM>
static float  VMM_product( VMM *model, const float &weight, embree::Vec3<float> &meanDirection, const float &kappa)
{
    return model->product(weight, meanDirection, kappa);
}

template< typename VMM>
static VMM VMM_copy( VMM *model)
{
    VMM vmmCopy (*model);
    return vmmCopy;
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


m.def("toSphericalCoordinates", &rkguide::toSphericalCoordinates);

//m.def("sphericalDirection", overload_cast_<float, float>()(&rkguide::sphericalDirection));
m.def("sphericalDirection", &sphericalDirection);
m.def("squareToUniformSphere", &rkguide::squareToUniformSphere);

m.def("StoreDirectionalSampleData", &DirectionalSampleData_storeDirectionalSampleData);
m.def("LoadDirectionalSampleData", &DirectionalSampleData_loadDirectionalSampleData);

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


py::class_< rkguide::DirectionalSampleData >(m, "DirectionalSampleData")
    .def(py::init<>())
    .def_readwrite("position", &rkguide::DirectionalSampleData::position)
    .def_readwrite("direction", &rkguide::DirectionalSampleData::direction)
    .def_readwrite("weight", &rkguide::DirectionalSampleData::weight)
    .def_readwrite("pdf", &rkguide::DirectionalSampleData::pdf)
    .def_readwrite("distance", &rkguide::DirectionalSampleData::distance)
    .def("__repr__", &rkguide::DirectionalSampleData::toString);


py::class_< rkguide::VonMisesFisherMixture<4,32> >(m, "VMM32v4")
    .def(py::init<>())
    .def(py::init< rkguide::VonMisesFisherMixture<4,32> >())
    .def("pdf", &rkguide::VonMisesFisherMixture<4,32>::pdf)
    .def("sample", &rkguide::VonMisesFisherMixture<4,32>::sample)
    .def("mergeComponents", &rkguide::VonMisesFisherMixture<4,32>::mergeComponents)
    .def("swapComponents", &rkguide::VonMisesFisherMixture<4,32>::swapComponents)
    .def("product", &VMM_product< rkguide::VonMisesFisherMixture<4,32> >)
    .def("uniformInit", &rkguide::VonMisesFisherMixture<4,32>::uniformInit)
    .def("copy", &VMM_copy< rkguide::VonMisesFisherMixture<4,32> >)
    .def("__repr__", &rkguide::VonMisesFisherMixture<4,32>::toString);

/*
py::class_< rkguide::VonMisesFisherMixture<8,32> >(m, "VMM32v8")
    .def(py::init<>())
    .def("__repr__", &rkguide::VonMisesFisherMixture<8,32>::toString);

py::class_< rkguide::VonMisesFisherMixture<16,32> >(m, "VMM32v16")
    .def(py::init<>())
    .def("__repr__", &rkguide::VonMisesFisherMixture<16,32>::toString);
*/

py::class_< rkguide::VonMisesFisherFactory<4,32> >(m, "VMMFactory32v4")
    .def(py::init<>())
    .def("InitUniformVMM", &rkguide::VonMisesFisherFactory<4,32>::InitUniformVMM);

auto WEMVMMFactory32v4 = py::class_< rkguide::WeightedEMVonMisesFisherFactory<4,32> >(m, "WEMVMMFactory32v4")
    .def(py::init<>())
    .def("fit", &WeightedEMVMMFactory_fit< rkguide::WeightedEMVonMisesFisherFactory<4,32>, rkguide::VonMisesFisherMixture<4,32> >)
    .def("update", &WeightedEMVMMFactory_update< rkguide::WeightedEMVonMisesFisherFactory<4,32>, rkguide::VonMisesFisherMixture<4,32> >);

py::class_< rkguide::WeightedEMVonMisesFisherFactory<4,32>::Configuration >(WEMVMMFactory32v4, "Configuration")
    .def(py::init<>())
    .def_readwrite("maK", &rkguide::WeightedEMVonMisesFisherFactory<4,32>::Configuration::maK)
    .def_readwrite("maxEMIterrations", &rkguide::WeightedEMVonMisesFisherFactory<4,32>::Configuration::maxEMIterrations)
    .def_readwrite("maxKappa", &rkguide::WeightedEMVonMisesFisherFactory<4,32>::Configuration::maxKappa)
    .def_readonly("maxMeanCosine", &rkguide::WeightedEMVonMisesFisherFactory<4,32>::Configuration::maxMeanCosine)
    .def_readwrite("convergenceThreshold", &rkguide::WeightedEMVonMisesFisherFactory<4,32>::Configuration::convergenceThreshold)
    .def_readwrite("weightPrior", &rkguide::WeightedEMVonMisesFisherFactory<4,32>::Configuration::weightPrior)
    .def_readwrite("meanCosinePriorStrength", &rkguide::WeightedEMVonMisesFisherFactory<4,32>::Configuration::meanCosinePriorStrength)
    .def_readwrite("meanCosinePrior", &rkguide::WeightedEMVonMisesFisherFactory<4,32>::Configuration::meanCosinePrior)
    .def("init", &rkguide::WeightedEMVonMisesFisherFactory<4,32>::Configuration::init)
    .def("__repr__", &rkguide::WeightedEMVonMisesFisherFactory<4,32>::Configuration::toString);

py::class_< rkguide::WeightedEMVonMisesFisherFactory<4,32>::SufficientStatisitcs >(WEMVMMFactory32v4, "SufficientStatisitcs")
    .def(py::init<>())
    .def("__repr__", &rkguide::WeightedEMVonMisesFisherFactory<4,32>::SufficientStatisitcs::toString);
    //.def("update", &rkguide::WeightedEMVonMisesFisherFactory<4,32>::update);


py::class_< rkguide::VonMisesFisherChiSquareComponentMerger<4,32> >(m, "VMMChiSquareComponentMerger32v4")
    .def(py::init<>())
    .def("MergeNext", &rkguide::VonMisesFisherChiSquareComponentMerger<4,32>::MergeNext)
    .def("CalculateMergeCost", &rkguide::VonMisesFisherChiSquareComponentMerger<4,32>::CalculateMergeCost);


auto VMMChiSquareComponentSplitter32v4 = py::class_< rkguide::VonMisesFisherChiSquareComponentSplitter<4,32> >(m, "VMMChiSquareComponentSplitter32v4")
    .def(py::init<>())
    .def("CalculateSplitStatistics", &VMMChiSquareComponentSplitter_CalculateSplitStatistics< rkguide::VonMisesFisherChiSquareComponentSplitter<4,32>, rkguide::VonMisesFisherMixture<4,32> >)
    .def("UpdateSplitStatistics", &VMMChiSquareComponentSplitter_UpdateSplitStatistics< rkguide::VonMisesFisherChiSquareComponentSplitter<4,32>, rkguide::VonMisesFisherMixture<4,32> >);


py::class_< rkguide::VonMisesFisherChiSquareComponentSplitter<4,32>::ComponentSplitStatistics >(VMMChiSquareComponentSplitter32v4, "ComponentSplitStatistics")
    .def(py::init<>())
    .def("__repr__", &rkguide::VonMisesFisherChiSquareComponentSplitter<4,32>::ComponentSplitStatistics::toString);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}