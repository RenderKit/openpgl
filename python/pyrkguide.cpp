#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <rkguide/rkguide.h>
#include <rkguide/data/DirectionalSampleData.h>
#include <rkguide/vmm/VMM.h>
#include <rkguide/vmm/VMMFactory.h>
#include <rkguide/vmm/WeightedEMVMMFactory.h>
#include <rkguide/vmm/VMMChiSquareComponentMerger.h>
#include <rkguide/vmm/VMMChiSquareComponentSplitter.h>
#include <rkguide/vmm/AdaptiveSplitandMergeFactory.h>

namespace py = pybind11;

#define MODULENAME(x) x##PYGUIDE_MAX_COMPONENTS

namespace rkguide
{
typedef rkguide::VonMisesFisherMixture<4,PYGUIDE_MAX_COMPONENTS> VMM;
typedef rkguide::VonMisesFisherFactory<4,PYGUIDE_MAX_COMPONENTS> VMMFactory;
typedef rkguide::WeightedEMVonMisesFisherFactory<4,PYGUIDE_MAX_COMPONENTS>  VMMWEMFactory;
typedef rkguide::VonMisesFisherChiSquareComponentMerger<4,PYGUIDE_MAX_COMPONENTS> VMMChiSquareComponentMerger;
typedef rkguide::VonMisesFisherChiSquareComponentSplitter<4,PYGUIDE_MAX_COMPONENTS> VMMChiSquareComponentSplitter;
typedef rkguide::AdaptiveSplitAndMergeFactory<4,PYGUIDE_MAX_COMPONENTS> VMMAdaptiveSplitAndMergeFactory;
}

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

static rkguide::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics  AdaptiveSplitAndMergeFactory_fit( rkguide::VMMAdaptiveSplitAndMergeFactory *asmFactory, rkguide::VMM &model, const size_t &K, rkguide::VMMAdaptiveSplitAndMergeFactory::ASMStatistics &stats, const py::list &listParticles, rkguide::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration &cfg)
{
    rkguide::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics fitStats;
    std::vector<rkguide::DirectionalSampleData> dataPoints = listParticles.cast<std::vector<rkguide::DirectionalSampleData>>();
	//rkguide::StoreDirectionalSampleData(fileName, dataPoints.data(), dataPoints.size());
    asmFactory->fit(model, K, stats, dataPoints.data(), dataPoints.size(), cfg, fitStats);
    return fitStats;
}

static rkguide::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics  AdaptiveSplitAndMergeFactory_update( rkguide::VMMAdaptiveSplitAndMergeFactory *asmFactory, rkguide::VMM &model, rkguide::VMMAdaptiveSplitAndMergeFactory::ASMStatistics &stats, const py::list &listParticles, rkguide::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration &cfg)
{
    rkguide::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics fitStats;
    std::vector<rkguide::DirectionalSampleData> dataPoints = listParticles.cast<std::vector<rkguide::DirectionalSampleData>>();
	//rkguide::StoreDirectionalSampleData(fileName, dataPoints.data(), dataPoints.size());
    asmFactory->update(model, stats, dataPoints.data(), dataPoints.size(), cfg, fitStats);
    return fitStats;
}


template< typename WeightedEMVMMFactory, typename VMM>
static rkguide::VMMWEMFactory::FittingStatistics  WeightedEMVMMFactory_fit( WeightedEMVMMFactory *vmmFactory, VMM &model, const size_t &K, typename WeightedEMVMMFactory::SufficientStatisitcs &stats, const py::list &listParticles, typename WeightedEMVMMFactory::Configuration &cfg)
{
    rkguide::VMMWEMFactory::FittingStatistics fitStats;
    std::vector<rkguide::DirectionalSampleData> dataPoints = listParticles.cast<std::vector<rkguide::DirectionalSampleData>>();
	//rkguide::StoreDirectionalSampleData(fileName, dataPoints.data(), dataPoints.size());
    vmmFactory->fitMixture(model, K, stats, dataPoints.data(), dataPoints.size(), cfg, fitStats);
    return fitStats;
}


template< typename WeightedEMVMMFactory, typename VMM>
static rkguide::VMMWEMFactory::FittingStatistics  WeightedEMVMMFactory_partialUpdate( WeightedEMVMMFactory *vmmFactory, VMM &model, const typename WeightedEMVMMFactory::PartialFittingMask &mask, typename WeightedEMVMMFactory::SufficientStatisitcs &stats, const py::list &listParticles, typename WeightedEMVMMFactory::Configuration &cfg)
{
    rkguide::VMMWEMFactory::FittingStatistics fitStats;
    std::vector<rkguide::DirectionalSampleData> dataPoints = listParticles.cast<std::vector<rkguide::DirectionalSampleData>>();
	//rkguide::StoreDirectionalSampleData(fileName, dataPoints.data(), dataPoints.size());
    vmmFactory->partialUpdateMixture(model, mask, stats, dataPoints.data(), dataPoints.size(), cfg, fitStats);
    return fitStats;
}

template< typename WeightedEMVMMFactory, typename VMM>
static rkguide::VMMWEMFactory::FittingStatistics  WeightedEMVMMFactory_update( WeightedEMVMMFactory *vmmFactory, VMM &model, typename WeightedEMVMMFactory::SufficientStatisitcs &stats, const py::list &listParticles, typename WeightedEMVMMFactory::Configuration &cfg)
{
    rkguide::VMMWEMFactory::FittingStatistics fitStats;
    std::vector<rkguide::DirectionalSampleData> dataPoints = listParticles.cast<std::vector<rkguide::DirectionalSampleData>>();
	//rkguide::StoreDirectionalSampleData(fileName, dataPoints.data(), dataPoints.size());
    vmmFactory->updateMixture(model, stats, dataPoints.data(), dataPoints.size(), cfg, fitStats);
    return fitStats;
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


template< typename VMMChiSquareComponentSplitter, typename VMM>
static py::list VMMChiSquareComponentSplitter_GetProjectedLocalDirections( VMMChiSquareComponentSplitter *vmmSplitter, VMM &model, const size_t &idx, const py::list &listParticles)
{
    std::vector<rkguide::DirectionalSampleData> dataPoints = listParticles.cast<std::vector<rkguide::DirectionalSampleData>>();
	//rkguide::StoreDirectionalSampleData(fileName, dataPoints.data(), dataPoints.size());
    rkguide::Vector3 *localDirections2D =new rkguide::Vector3 [dataPoints.size()];
    rkguide::ComponentSplitinfo cmpSplit = vmmSplitter->GetProjectedLocalDirections(model, idx, dataPoints.data(), dataPoints.size(), localDirections2D);

    py::list list;

    for (size_t n = 0; n < dataPoints.size(); n++)
    {
        py::list location2D;
        location2D.append(localDirections2D[n].x);
        location2D.append(localDirections2D[n].y);
        location2D.append(localDirections2D[n].z);

        list.append(location2D);
    }

    py::list resultList;
    resultList.append(cmpSplit);
    resultList.append(list);
    return resultList;

}

template< typename VMMChiSquareComponentSplitter, typename VMM>
static void  VMMChiSquareComponentSplitter_performSplitting( VMMChiSquareComponentSplitter *vmmSplitter, VMM &vmm, const float &splitThreshold, const py::list &listParticles, typename VMMChiSquareComponentSplitter::VMMFactory::Configuration &cfg, const bool &doPartialRefit, const int &maxSplittingItr)
{
    std::vector<rkguide::DirectionalSampleData> dataPoints = listParticles.cast<std::vector<rkguide::DirectionalSampleData>>();
    float mcEstimate = 0.0f;
    for (size_t n = 0; n < dataPoints.size(); n++)
    {
        mcEstimate += dataPoints[n].weight;
    }
    mcEstimate /= (float) dataPoints.size();

    vmmSplitter->PerformSplitting(vmm, splitThreshold, mcEstimate, dataPoints.data(), dataPoints.size(), cfg, doPartialRefit, maxSplittingItr);

}


template< typename VMM>
static float  VMM_product( VMM *model, const float &weight, embree::Vec3<float> &meanDirection, const float &kappa)
{
    return model->product(weight, meanDirection, kappa);
}

template< typename VMM, int VecSize>
static py::list VMM_softAssignment( VMM *model, const embree::Vec3<float> &direction)
{
    typename VMM::SoftAssignment softAssign;
    bool success = model->softAssignment(direction, softAssign);

    //std::cout << softAssign.toString() << std::endl;

    std::vector<float> assigns;
    for ( size_t k = 0; k < model->_numComponents; k++ )
    {
        const div_t tmp = div(k, static_cast<int>(VecSize));
        assigns.push_back(softAssign.assignments[tmp.quot][tmp.rem]);
    }

    py::list results;
    results.append(success);
    results.append(softAssign.pdf);
    results.append(py::cast(assigns));
    return results;

}


template< typename VMM, int VecSize>
static py::list VMM_getWeights( VMM *model)
{
    std::vector<float> weights;
    for ( size_t k = 0; k < model->_numComponents; k++ )
    {
        const div_t tmp = div(k, static_cast<int>(VecSize));
        weights.push_back(model->_weights[tmp.quot][tmp.rem]);
    }
    return py::cast(weights);

}


template< typename VMM, int VecSize>
static void VMM_setWeights( VMM *model, const py::list &listWeights)
{   

    std::cout << "VMM_setWeights: " <<model->_numComponents << std::endl;
    //std::vector<rkguide::DirectionalSampleData> dataPoints = listParticles.cast<std::vector<rkguide::DirectionalSampleData>>();
    std::vector<double> weights = listWeights.cast<std::vector<double>>();
    for ( size_t k = 0; k < model->_numComponents; k++ )
    {
        const div_t tmp = div(k, static_cast<int>(VecSize));
        //weights.push_back(model->_weights[tmp.quot][tmp.rem]);
        std::cout << "w["<<k<<"]" << weights[k] << std::endl;
        model->_weights[tmp.quot][tmp.rem] = weights[k];
    }
    //return py::cast(weights);

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
    rkguide::VMM vmm;
    std::cout << vmm.toString() << std::endl;
}

rkguide::Vector3 sphericalDirection(const float &theta, const float &phi)
{
    return rkguide::sphericalDirection(theta, phi);
}

//replace by macro
#if PYGUIDE_MAX_COMPONENTS == 32
PYBIND11_MODULE(pyrkguide32, m) {
#elif PYGUIDE_MAX_COMPONENTS == 64
PYBIND11_MODULE(pyrkguide64, m) {
#elif PYGUIDE_MAX_COMPONENTS == 128
PYBIND11_MODULE(pyrkguide128, m) {
#else
PYBIND11_MODULE(pyrkguide32, m) {
#endif
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


py::class_< rkguide::ComponentSplitinfo >(m, "ComponentSplitinfo")
    .def(py::init<>())
    .def_readwrite("mean", &rkguide::ComponentSplitinfo::mean)
    .def_readwrite("covariance", &rkguide::ComponentSplitinfo::covariance)
    .def_readwrite("eigenValue0", &rkguide::ComponentSplitinfo::eigenValue0)
    .def_readwrite("eigenValue1", &rkguide::ComponentSplitinfo::eigenValue1)
    .def_readwrite("eigenVector0", &rkguide::ComponentSplitinfo::eigenVector0)
    .def_readwrite("eigenVector1", &rkguide::ComponentSplitinfo::eigenVector1)
    .def("__repr__", &rkguide::ComponentSplitinfo::toString);


py::class_< rkguide::VMM >(m, "VMM")
    .def(py::init<>())
    .def(py::init< rkguide::VMM >())
    .def_readonly("nComponents",  &rkguide::VMM::_numComponents)
    .def("pdf", &rkguide::VMM::pdf)
    .def("sample", &rkguide::VMM::sample)
    .def("mergeComponents", &rkguide::VMM::mergeComponents)
    .def("swapComponents", &rkguide::VMM::swapComponents)
    .def("product", &VMM_product< rkguide::VMM >)
    .def("uniformInit", &rkguide::VMM::uniformInit)
    .def("softAssignment", &VMM_softAssignment<rkguide::VMM, 4 >)
    .def("getWeights", &VMM_getWeights<rkguide::VMM, 4 >)
    .def("setWeights", &VMM_setWeights<rkguide::VMM, 4 >)
    .def("setNumComponents", &rkguide::VMM::setNumComponents)
    .def("setComponentWeight", &rkguide::VMM::setComponentWeight)
    .def("setComponentMeanDirection", &rkguide::VMM::setComponentMeanDirection)
    .def("setComponentKappa", &rkguide::VMM::setComponentKappa)
    .def("getComponentWeight", &rkguide::VMM::getComponentWeight)
    .def("getComponentMeanDirection", &rkguide::VMM::getComponentMeanDirection)
    .def("getComponentKappa", &rkguide::VMM::getComponentKappa)
    .def("normalizeWeights", &rkguide::VMM::_normalizeWeights)
    .def("copy", &VMM_copy< rkguide::VMM >)
    .def("__repr__", &rkguide::VMM::toString);

py::class_< rkguide::VMMFactory >(m, "VMMFactory")
    .def(py::init<>())
    .def("InitUniformVMM", &rkguide::VMMFactory::InitUniformVMM);

auto VMMWEMFactory = py::class_< rkguide::VMMWEMFactory >(m, "VMMWEMFactory")
    .def(py::init<>())
    .def("fit", &WeightedEMVMMFactory_fit< rkguide::VMMWEMFactory, rkguide::VMM >)
    .def("update", &WeightedEMVMMFactory_update< rkguide::VMMWEMFactory, rkguide::VMM >)
    .def("partialUpdate", &WeightedEMVMMFactory_partialUpdate< rkguide::VMMWEMFactory, rkguide::VMM >)
    .def("VMMfromSufficientStatisitcs", &rkguide::VMMWEMFactory::VMMfromSufficientStatisitcs);


py::class_< rkguide::VMMWEMFactory::Configuration >(VMMWEMFactory, "Configuration")
    .def(py::init<>())
    .def_readwrite("maK", &rkguide::VMMWEMFactory::Configuration::maK)
    .def_readwrite("maxEMIterrations", &rkguide::VMMWEMFactory::Configuration::maxEMIterrations)
    .def_readwrite("maxKappa", &rkguide::VMMWEMFactory::Configuration::maxKappa)
    .def_readonly("maxMeanCosine", &rkguide::VMMWEMFactory::Configuration::maxMeanCosine)
    .def_readwrite("convergenceThreshold", &rkguide::VMMWEMFactory::Configuration::convergenceThreshold)
    .def_readwrite("weightPrior", &rkguide::VMMWEMFactory::Configuration::weightPrior)
    .def_readwrite("meanCosinePriorStrength", &rkguide::VMMWEMFactory::Configuration::meanCosinePriorStrength)
    .def_readwrite("meanCosinePrior", &rkguide::VMMWEMFactory::Configuration::meanCosinePrior)
    .def("init", &rkguide::VMMWEMFactory::Configuration::init)
    .def("__repr__", &rkguide::VMMWEMFactory::Configuration::toString);

py::class_< rkguide::VMMWEMFactory::SufficientStatisitcs >(VMMWEMFactory, "SufficientStatisitcs")
    .def(py::init<>())
    .def(py::init< rkguide::VMMWEMFactory::SufficientStatisitcs >())
    .def_readwrite("sumWeights", &rkguide::VMMWEMFactory::SufficientStatisitcs::sumWeights)
    .def_readwrite("numSamples", &rkguide::VMMWEMFactory::SufficientStatisitcs::numSamples)
    .def("clear", &rkguide::VMMWEMFactory::SufficientStatisitcs::clear)
    .def("decay", &rkguide::VMMWEMFactory::SufficientStatisitcs::decay)
    .def("__repr__", &rkguide::VMMWEMFactory::SufficientStatisitcs::toString);
    //.def("update", &rkguide::WEMVMMFactory::update);

py::class_< rkguide::VMMWEMFactory::FittingStatistics >(VMMWEMFactory, "FittingStatistics")
    .def(py::init<>())
    .def(py::init< rkguide::VMMWEMFactory::FittingStatistics >())
    .def_readwrite("numSamples", &rkguide::VMMWEMFactory::FittingStatistics::numSamples)
    .def_readwrite("numIterations", &rkguide::VMMWEMFactory::FittingStatistics::numIterations)
    .def_readwrite("summedWeightedLogLikelihood", &rkguide::VMMWEMFactory::FittingStatistics::summedWeightedLogLikelihood);


py::class_< rkguide::VMMWEMFactory::PartialFittingMask >(VMMWEMFactory, "PartialFittingMask")
    .def(py::init<>())
    .def(py::init< rkguide::VMMWEMFactory::PartialFittingMask >())
    .def("resetToFalse", &rkguide::VMMWEMFactory::PartialFittingMask::resetToFalse)
    .def("resetToTrue", &rkguide::VMMWEMFactory::PartialFittingMask::resetToTrue)
    .def("setToTrue", &rkguide::VMMWEMFactory::PartialFittingMask::setToTrue)
    .def("setToFalse", &rkguide::VMMWEMFactory::PartialFittingMask::setToFalse)
    .def("__repr__", &rkguide::VMMWEMFactory::PartialFittingMask::toString);
    //.def("update", &rkguide::WEMVMMFactory::update);



py::class_< rkguide::VMMChiSquareComponentMerger >(m, "VMMChiSquareComponentMerger")
    .def(py::init<>())
    .def("MergeNext", &rkguide::VMMChiSquareComponentMerger::MergeNext)
    //.def("PerformMerging", &rkguide::VMMChiSquareComponentMerger::PerformMerging)
    .def("PerformMerging", (size_t (rkguide::VMMChiSquareComponentMerger::*)(rkguide::VMM &, const float &) const) &rkguide::VMMChiSquareComponentMerger::PerformMerging)
    .def("PerformMerging", (size_t (rkguide::VMMChiSquareComponentMerger::*)(rkguide::VMM &, const float &, rkguide::VMMWEMFactory::SufficientStatisitcs &, rkguide::VMMChiSquareComponentSplitter::ComponentSplitStatistics &) const) &rkguide::VMMChiSquareComponentMerger::PerformMerging)
    .def("CalculateMergeCost", &rkguide::VMMChiSquareComponentMerger::CalculateMergeCost);


auto VMMChiSquareComponentSplitter = py::class_< rkguide::VMMChiSquareComponentSplitter >(m, "VMMChiSquareComponentSplitter")
    .def(py::init<>())
    .def("CalculateSplitStatistics", &VMMChiSquareComponentSplitter_CalculateSplitStatistics< rkguide::VMMChiSquareComponentSplitter, rkguide::VMM >)
    .def("UpdateSplitStatistics", &VMMChiSquareComponentSplitter_UpdateSplitStatistics< rkguide::VMMChiSquareComponentSplitter, rkguide::VMM >)
    .def("GetProjectedLocalDirections", &VMMChiSquareComponentSplitter_GetProjectedLocalDirections< rkguide::VMMChiSquareComponentSplitter, rkguide::VMM >)
    .def("PerformSplitting", &VMMChiSquareComponentSplitter_performSplitting< rkguide::VMMChiSquareComponentSplitter, rkguide::VMM >)
    .def("SplitComponent", &rkguide::VMMChiSquareComponentSplitter::SplitComponent);


py::class_< rkguide::VMMChiSquareComponentSplitter::ComponentSplitStatistics >(VMMChiSquareComponentSplitter, "ComponentSplitStatistics")
    .def(py::init<>())
    .def_readonly("numComponents", &rkguide::VMMChiSquareComponentSplitter::ComponentSplitStatistics::numComponents)
    //.def_readonly("mcEstimate", &rkguide::VMMChiSquareComponentSplitter::ComponentSplitStatistics::mcEstimate)
    //.def_readonly("numSamples", &rkguide::VMMChiSquareComponentSplitter::ComponentSplitStatistics::numSamplesOld)
    .def("getSumChiSquareEst", &rkguide::VMMChiSquareComponentSplitter::ComponentSplitStatistics::getSumChiSquareEst)
    .def("getChiSquareEst", &rkguide::VMMChiSquareComponentSplitter::ComponentSplitStatistics::getChiSquareEst)
    .def("getHighestChiSquareIdx", &rkguide::VMMChiSquareComponentSplitter::ComponentSplitStatistics::getHighestChiSquareIdx)
    .def("decay", &rkguide::VMMChiSquareComponentSplitter::ComponentSplitStatistics::decay)
    .def("mergeComponentStats", &rkguide::VMMChiSquareComponentSplitter::ComponentSplitStatistics::mergeComponentStats)
    .def("getSplitMean", &rkguide::VMMChiSquareComponentSplitter::ComponentSplitStatistics::getSplitMean)
    .def("getSplitCovariance", &rkguide::VMMChiSquareComponentSplitter::ComponentSplitStatistics::getSplitCovariance)
    .def("__repr__", &rkguide::VMMChiSquareComponentSplitter::ComponentSplitStatistics::toString);


auto AdaptiveSplitAndMergeFactory = py::class_< rkguide::VMMAdaptiveSplitAndMergeFactory >(m, "VMMAdaptiveSplitAndMergeFactory")
    .def(py::init<>())
    .def("fit", &AdaptiveSplitAndMergeFactory_fit)
    .def("update", &AdaptiveSplitAndMergeFactory_update);


py::class_< rkguide::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration >(AdaptiveSplitAndMergeFactory, "ASMConfiguration")
    .def(py::init<>())
    .def_readwrite("weightedEMCfg", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration::weightedEMCfg)
    .def_readwrite("splittingThreshold", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration::splittingThreshold)
    .def_readwrite("mergingThreshold", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration::mergingThreshold)
    .def_readwrite("partialReFit", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration::partialReFit)
    .def_readwrite("maxSplitItr", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration::maxSplitItr)
    .def("__repr__", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration::toString);

py::class_< rkguide::VMMAdaptiveSplitAndMergeFactory::ASMStatistics >(AdaptiveSplitAndMergeFactory, "ASMStatistics")
    .def(py::init<>())
    .def_readwrite("sufficientStatistics", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMStatistics::sufficientStatistics)
    .def_readwrite("splittingStatistics", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMStatistics::splittingStatistics)
    .def("decay", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMStatistics::decay)
    //.def_readwrite("clear", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMStatistics::clear)
    //.def_readwrite("clearAll", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMStatistics::clearAll)
    .def("__repr__", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMStatistics::toString);


py::class_< rkguide::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics >(AdaptiveSplitAndMergeFactory, "ASMFittingStatistics")
    .def(py::init<>())
    .def_readwrite("numSamples", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics::numSamples)
    .def_readwrite("numSplits", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics::numSplits)
    .def_readwrite("numMerges", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics::numMerges)
    .def_readwrite("numComponents", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics::numComponents)
    .def_readwrite("numUpdateWEMIterations", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics::numUpdateWEMIterations)
    .def_readwrite("numPartialUpdateWEMIterations", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics::numPartialUpdateWEMIterations)
    .def("__repr__", &rkguide::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics::toString);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}