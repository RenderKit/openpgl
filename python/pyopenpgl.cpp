#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <openpgl/openpgl_common.h>
#include <openpgl/data/SampleData.h>
#include <openpgl/vmm/ParallaxAwareVMM.h>
#include <openpgl/vmm/VMM.h>
#include <openpgl/vmm/VMMFactory.h>
#include <openpgl/vmm/WeightedEMVMMFactory.h>
#include <openpgl/vmm/WeightedEMParallaxAwareVMMFactory.h>
#include <openpgl/vmm/VMMChiSquareComponentMerger.h>
#include <openpgl/vmm/VMMChiSquareComponentSplitter.h>
#include <openpgl/vmm/AdaptiveSplitandMergeFactory.h>

namespace py = pybind11;

#define MODULENAME(x) x##PYGUIDE_MAX_COMPONENTS

namespace openpgl
{
typedef openpgl::ParallaxAwareVonMisesFisherMixture<4,PYGUIDE_MAX_COMPONENTS> VMM;
typedef openpgl::VonMisesFisherFactory<VMM> VMMFactory;
typedef openpgl::WeightedEMParallaxAwareVonMisesFisherFactory<VMM>  VMMWEMFactory;
typedef openpgl::VonMisesFisherChiSquareComponentMerger<VMMWEMFactory> VMMChiSquareComponentMerger;
typedef openpgl::VonMisesFisherChiSquareComponentSplitter<VMMWEMFactory> VMMChiSquareComponentSplitter;
typedef openpgl::AdaptiveSplitAndMergeFactory<VMM> VMMAdaptiveSplitAndMergeFactory;
}

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

int add(int i, int j) {
    return i + j;
}

static void SampleData_storeSampleData(const std::string& fileName, const py::list &listParticles){
	std::vector<openpgl::SampleData> dataPoints = listParticles.cast<std::vector<openpgl::SampleData>>();
	openpgl::StoreSampleData(fileName, dataPoints.data(), dataPoints.size());
}

static py::list SampleData_loadSampleData(const std::string& fileName){
    size_t numData;
    openpgl::SampleData* data = openpgl::LoadSampleData(fileName, numData);
    std::vector<openpgl::SampleData> dataPoints;
    for( size_t n = 0; n < numData; n++)
    {
        dataPoints.push_back(data[n]);
    }
	py::list list = py::cast(dataPoints);
	return list;
}

static openpgl::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics  AdaptiveSplitAndMergeFactory_fit( openpgl::VMMAdaptiveSplitAndMergeFactory *asmFactory, openpgl::VMM &model, const size_t &K, openpgl::VMMAdaptiveSplitAndMergeFactory::ASMStatistics &stats, const py::list &listParticles, openpgl::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration &cfg)
{
    openpgl::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics fitStats;
    std::vector<openpgl::SampleData> dataPoints = listParticles.cast<std::vector<openpgl::SampleData>>();
	//openpgl::StoreSampleData(fileName, dataPoints.data(), dataPoints.size());
    asmFactory->fit(model, K, stats, dataPoints.data(), dataPoints.size(), cfg, fitStats);
    return fitStats;
}

static openpgl::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics  AdaptiveSplitAndMergeFactory_update( openpgl::VMMAdaptiveSplitAndMergeFactory *asmFactory, openpgl::VMM &model, openpgl::VMMAdaptiveSplitAndMergeFactory::ASMStatistics &stats, const py::list &listParticles, openpgl::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration &cfg)
{
    openpgl::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics fitStats;
    std::vector<openpgl::SampleData> dataPoints = listParticles.cast<std::vector<openpgl::SampleData>>();
	//openpgl::StoreSampleData(fileName, dataPoints.data(), dataPoints.size());
    asmFactory->update(model, stats, dataPoints.data(), dataPoints.size(), cfg, fitStats);
    return fitStats;
}


template< typename WeightedEMVMMFactory, typename VMM>
static openpgl::VMMWEMFactory::FittingStatistics  WeightedEMVMMFactory_fit( WeightedEMVMMFactory *vmmFactory, VMM &model, const size_t &K, typename WeightedEMVMMFactory::SufficientStatisitcs &stats, const py::list &listParticles, typename WeightedEMVMMFactory::Configuration &cfg)
{
    openpgl::VMMWEMFactory::FittingStatistics fitStats;
    std::vector<openpgl::SampleData> dataPoints = listParticles.cast<std::vector<openpgl::SampleData>>();
	//openpgl::StoreSampleData(fileName, dataPoints.data(), dataPoints.size());
    vmmFactory->fitMixture(model, K, stats, dataPoints.data(), dataPoints.size(), cfg, fitStats);
    return fitStats;
}


template< typename WeightedEMVMMFactory, typename VMM>
static openpgl::VMMWEMFactory::FittingStatistics  WeightedEMVMMFactory_partialUpdate( WeightedEMVMMFactory *vmmFactory, VMM &model, typename WeightedEMVMMFactory::PartialFittingMask &mask, typename WeightedEMVMMFactory::SufficientStatisitcs &stats, const py::list &listParticles, typename WeightedEMVMMFactory::Configuration &cfg)
{
    openpgl::VMMWEMFactory::FittingStatistics fitStats;
    std::vector<openpgl::SampleData> dataPoints = listParticles.cast<std::vector<openpgl::SampleData>>();
	//openpgl::StoreSampleData(fileName, dataPoints.data(), dataPoints.size());
    vmmFactory->partialUpdateMixture(model, mask, stats, dataPoints.data(), dataPoints.size(), cfg, fitStats);
    return fitStats;
}

template< typename WeightedEMVMMFactory, typename VMM>
static openpgl::VMMWEMFactory::FittingStatistics  WeightedEMVMMFactory_update( WeightedEMVMMFactory *vmmFactory, VMM &model, typename WeightedEMVMMFactory::SufficientStatisitcs &stats, const py::list &listParticles, typename WeightedEMVMMFactory::Configuration &cfg)
{
    openpgl::VMMWEMFactory::FittingStatistics fitStats;
    std::vector<openpgl::SampleData> dataPoints = listParticles.cast<std::vector<openpgl::SampleData>>();
	//openpgl::StoreSampleData(fileName, dataPoints.data(), dataPoints.size());
    vmmFactory->updateMixture(model, stats, dataPoints.data(), dataPoints.size(), cfg, fitStats);
    return fitStats;
}

template< typename VMMChiSquareComponentSplitter, typename VMM>
static void  VMMChiSquareComponentSplitter_CalculateSplitStatistics( VMMChiSquareComponentSplitter *vmmSplitter, VMM &model, typename VMMChiSquareComponentSplitter::ComponentSplitStatistics &splitStats, const py::list &listParticles)
{
    std::vector<openpgl::SampleData> dataPoints = listParticles.cast<std::vector<openpgl::SampleData>>();
    float mcEstimate = 0.0f;
    for (size_t n = 0; n < dataPoints.size(); n++)
    {
        mcEstimate += dataPoints[n].weight;
    }
    mcEstimate /= (float) dataPoints.size();

	//openpgl::StoreSampleData(fileName, dataPoints.data(), dataPoints.size());
    vmmSplitter->CalculateSplitStatistics(model, splitStats, mcEstimate, dataPoints.data(), dataPoints.size());
}

template< typename VMMChiSquareComponentSplitter, typename VMM>
static void  VMMChiSquareComponentSplitter_UpdateSplitStatistics( VMMChiSquareComponentSplitter *vmmSplitter, VMM &model, typename VMMChiSquareComponentSplitter::ComponentSplitStatistics &splitStats, const py::list &listParticles)
{
    std::vector<openpgl::SampleData> dataPoints = listParticles.cast<std::vector<openpgl::SampleData>>();
    float mcEstimate = 0.0f;
    for (size_t n = 0; n < dataPoints.size(); n++)
    {
        mcEstimate += dataPoints[n].weight;
    }
    mcEstimate /= (float) dataPoints.size();

	//openpgl::StoreSampleData(fileName, dataPoints.data(), dataPoints.size());
    vmmSplitter->UpdateSplitStatistics(model, splitStats, mcEstimate, dataPoints.data(), dataPoints.size());
}


template< typename VMMChiSquareComponentSplitter, typename VMM>
static py::list VMMChiSquareComponentSplitter_GetProjectedLocalDirections( VMMChiSquareComponentSplitter *vmmSplitter, VMM &model, const size_t &idx, const py::list &listParticles)
{
    std::vector<openpgl::SampleData> dataPoints = listParticles.cast<std::vector<openpgl::SampleData>>();
	//openpgl::StoreSampleData(fileName, dataPoints.data(), dataPoints.size());
    openpgl::Vector3 *localDirections2D =new openpgl::Vector3 [dataPoints.size()];
    openpgl::ComponentSplitinfo cmpSplit = vmmSplitter->GetProjectedLocalDirections(model, idx, dataPoints.data(), dataPoints.size(), localDirections2D);

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
    std::vector<openpgl::SampleData> dataPoints = listParticles.cast<std::vector<openpgl::SampleData>>();
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
    //std::vector<openpgl::SampleData> dataPoints = listParticles.cast<std::vector<openpgl::SampleData>>();
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
    openpgl::VMM vmm;
    std::cout << vmm.toString() << std::endl;
}

openpgl::Vector3 sphericalDirection(const float &theta, const float &phi)
{
    return openpgl::sphericalDirection(theta, phi);
}

//replace by macro
#if PYGUIDE_MAX_COMPONENTS == 32
PYBIND11_MODULE(pyopenpgl32, m) {
#elif PYGUIDE_MAX_COMPONENTS == 64
PYBIND11_MODULE(pyopenpgl64, m) {
#elif PYGUIDE_MAX_COMPONENTS == 128
PYBIND11_MODULE(pyopenpgl128, m) {
#else
PYBIND11_MODULE(pyopenpgl32, m) {
#endif
    m.doc() = R"pbdoc(
        Python module for Intel's openpgl
        -----------------------
        .. currentmodule:: openpgl
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("test", &test, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");


m.def("toSphericalCoordinates", &openpgl::toSphericalCoordinates);

//m.def("sphericalDirection", overload_cast_<float, float>()(&openpgl::sphericalDirection));
m.def("sphericalDirection", &sphericalDirection);
m.def("squareToUniformSphere", &openpgl::squareToUniformSphere);

m.def("StoreSampleData", &SampleData_storeSampleData);
m.def("LoadSampleData", &SampleData_loadSampleData);

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


py::class_< openpgl::SampleData >(m, "SampleData")
    .def(py::init<>())
    .def_readwrite("position", &openpgl::SampleData::position)
    .def_readwrite("direction", &openpgl::SampleData::direction)
    .def_readwrite("weight", &openpgl::SampleData::weight)
    .def_readwrite("pdf", &openpgl::SampleData::pdf)
    .def_readwrite("distance", &openpgl::SampleData::distance);
 //   .def("__repr__", &openpgl::SampleData::toString);


py::class_< openpgl::ComponentSplitinfo >(m, "ComponentSplitinfo")
    .def(py::init<>())
    .def_readwrite("mean", &openpgl::ComponentSplitinfo::mean)
    .def_readwrite("covariance", &openpgl::ComponentSplitinfo::covariance)
    .def_readwrite("eigenValue0", &openpgl::ComponentSplitinfo::eigenValue0)
    .def_readwrite("eigenValue1", &openpgl::ComponentSplitinfo::eigenValue1)
    .def_readwrite("eigenVector0", &openpgl::ComponentSplitinfo::eigenVector0)
    .def_readwrite("eigenVector1", &openpgl::ComponentSplitinfo::eigenVector1)
    .def("__repr__", &openpgl::ComponentSplitinfo::toString);


py::class_< openpgl::VMM >(m, "VMM")
    .def(py::init<>())
    .def(py::init< openpgl::VMM >())
    .def_readonly("nComponents",  &openpgl::VMM::_numComponents)
    .def("pdf", &openpgl::VMM::pdf)
    .def("sample", &openpgl::VMM::sample)
    .def("mergeComponents", &openpgl::VMM::mergeComponents)
    .def("swapComponents", &openpgl::VMM::swapComponents)
    .def("product", &VMM_product< openpgl::VMM >)
    .def("uniformInit", &openpgl::VMM::uniformInit)
    .def("softAssignment", &VMM_softAssignment<openpgl::VMM, 4 >)
    .def("getWeights", &VMM_getWeights<openpgl::VMM, 4 >)
    .def("setWeights", &VMM_setWeights<openpgl::VMM, 4 >)
    .def("setNumComponents", &openpgl::VMM::setNumComponents)
    .def("setComponentWeight", &openpgl::VMM::setComponentWeight)
    .def("setComponentMeanDirection", &openpgl::VMM::setComponentMeanDirection)
    .def("setComponentKappa", &openpgl::VMM::setComponentKappa)
    .def("getComponentWeight", &openpgl::VMM::getComponentWeight)
    .def("getComponentMeanDirection", &openpgl::VMM::getComponentMeanDirection)
    .def("getComponentKappa", &openpgl::VMM::getComponentKappa)
    .def("normalizeWeights", &openpgl::VMM::_normalizeWeights)
    .def("copy", &VMM_copy< openpgl::VMM >)
    .def("__repr__", &openpgl::VMM::toString);

py::class_< openpgl::VMMFactory >(m, "VMMFactory")
    .def(py::init<>())
    .def("InitUniformVMM", &openpgl::VMMFactory::InitUniformVMM);

auto VMMWEMFactory = py::class_< openpgl::VMMWEMFactory >(m, "VMMWEMFactory")
    .def(py::init<>())
    .def("fit", &WeightedEMVMMFactory_fit< openpgl::VMMWEMFactory, openpgl::VMM >)
    .def("update", &WeightedEMVMMFactory_update< openpgl::VMMWEMFactory, openpgl::VMM >)
    .def("partialUpdate", &WeightedEMVMMFactory_partialUpdate< openpgl::VMMWEMFactory, openpgl::VMM >)
    .def("VMMfromSufficientStatisitcs", &openpgl::VMMWEMFactory::VMMfromSufficientStatisitcs);


py::class_< openpgl::VMMWEMFactory::Configuration >(VMMWEMFactory, "Configuration")
    .def(py::init<>())
    .def_readwrite("maK", &openpgl::VMMWEMFactory::Configuration::maxK)
    .def_readwrite("maxEMIterrations", &openpgl::VMMWEMFactory::Configuration::maxEMIterrations)
    .def_readwrite("maxKappa", &openpgl::VMMWEMFactory::Configuration::maxKappa)
    .def_readonly("maxMeanCosine", &openpgl::VMMWEMFactory::Configuration::maxMeanCosine)
    .def_readwrite("convergenceThreshold", &openpgl::VMMWEMFactory::Configuration::convergenceThreshold)
    .def_readwrite("weightPrior", &openpgl::VMMWEMFactory::Configuration::weightPrior)
    .def_readwrite("meanCosinePriorStrength", &openpgl::VMMWEMFactory::Configuration::meanCosinePriorStrength)
    .def_readwrite("meanCosinePrior", &openpgl::VMMWEMFactory::Configuration::meanCosinePrior)
    .def("init", &openpgl::VMMWEMFactory::Configuration::init)
    .def("__repr__", &openpgl::VMMWEMFactory::Configuration::toString);

py::class_< openpgl::VMMWEMFactory::SufficientStatisitcs >(VMMWEMFactory, "SufficientStatisitcs")
    .def(py::init<>())
    .def(py::init< openpgl::VMMWEMFactory::SufficientStatisitcs >())
    //.def_readwrite("sumWeights", &openpgl::VMMWEMFactory::SufficientStatisitcs::sumWeights)
    //.def_readwrite("numSamples", &openpgl::VMMWEMFactory::SufficientStatisitcs::numSamples)
    .def("clear", &openpgl::VMMWEMFactory::SufficientStatisitcs::clear)
    .def("decay", &openpgl::VMMWEMFactory::SufficientStatisitcs::decay)
    .def("__repr__", &openpgl::VMMWEMFactory::SufficientStatisitcs::toString);
    //.def("update", &openpgl::WEMVMMFactory::update);

py::class_< openpgl::VMMWEMFactory::FittingStatistics >(VMMWEMFactory, "FittingStatistics")
    .def(py::init<>())
    .def(py::init< openpgl::VMMWEMFactory::FittingStatistics >())
    .def_readwrite("numSamples", &openpgl::VMMWEMFactory::FittingStatistics::numSamples)
    .def_readwrite("numIterations", &openpgl::VMMWEMFactory::FittingStatistics::numIterations)
    .def_readwrite("summedWeightedLogLikelihood", &openpgl::VMMWEMFactory::FittingStatistics::summedWeightedLogLikelihood);


py::class_< openpgl::VMMWEMFactory::PartialFittingMask >(VMMWEMFactory, "PartialFittingMask")
    .def(py::init<>())
    .def(py::init< openpgl::VMMWEMFactory::PartialFittingMask >())
    .def("resetToFalse", &openpgl::VMMWEMFactory::PartialFittingMask::resetToFalse)
    .def("resetToTrue", &openpgl::VMMWEMFactory::PartialFittingMask::resetToTrue)
    .def("setToTrue", &openpgl::VMMWEMFactory::PartialFittingMask::setToTrue)
    .def("setToFalse", &openpgl::VMMWEMFactory::PartialFittingMask::setToFalse)
    .def("__repr__", &openpgl::VMMWEMFactory::PartialFittingMask::toString);
    //.def("update", &openpgl::WEMVMMFactory::update);



py::class_< openpgl::VMMChiSquareComponentMerger >(m, "VMMChiSquareComponentMerger")
    .def(py::init<>())
    .def("MergeNext", &openpgl::VMMChiSquareComponentMerger::MergeNext)
    //.def("PerformMerging", &openpgl::VMMChiSquareComponentMerger::PerformMerging)
    .def("PerformMerging", (size_t (openpgl::VMMChiSquareComponentMerger::*)(openpgl::VMM &, const float &) const) &openpgl::VMMChiSquareComponentMerger::PerformMerging)
    .def("PerformMerging", (size_t (openpgl::VMMChiSquareComponentMerger::*)(openpgl::VMM &, const float &, openpgl::VMMWEMFactory::SufficientStatisitcs &, openpgl::VMMChiSquareComponentSplitter::ComponentSplitStatistics &) const) &openpgl::VMMChiSquareComponentMerger::PerformMerging)
    .def("CalculateMergeCost", &openpgl::VMMChiSquareComponentMerger::CalculateMergeCost);


auto VMMChiSquareComponentSplitter = py::class_< openpgl::VMMChiSquareComponentSplitter >(m, "VMMChiSquareComponentSplitter")
    .def(py::init<>())
    .def("CalculateSplitStatistics", &VMMChiSquareComponentSplitter_CalculateSplitStatistics< openpgl::VMMChiSquareComponentSplitter, openpgl::VMM >)
    .def("UpdateSplitStatistics", &VMMChiSquareComponentSplitter_UpdateSplitStatistics< openpgl::VMMChiSquareComponentSplitter, openpgl::VMM >)
    .def("GetProjectedLocalDirections", &VMMChiSquareComponentSplitter_GetProjectedLocalDirections< openpgl::VMMChiSquareComponentSplitter, openpgl::VMM >)
    .def("PerformSplitting", &VMMChiSquareComponentSplitter_performSplitting< openpgl::VMMChiSquareComponentSplitter, openpgl::VMM >)
    .def("SplitComponent", &openpgl::VMMChiSquareComponentSplitter::SplitComponent);


py::class_< openpgl::VMMChiSquareComponentSplitter::ComponentSplitStatistics >(VMMChiSquareComponentSplitter, "ComponentSplitStatistics")
    .def(py::init<>())
    .def_readonly("numComponents", &openpgl::VMMChiSquareComponentSplitter::ComponentSplitStatistics::numComponents)
    //.def_readonly("mcEstimate", &openpgl::VMMChiSquareComponentSplitter::ComponentSplitStatistics::mcEstimate)
    //.def_readonly("numSamples", &openpgl::VMMChiSquareComponentSplitter::ComponentSplitStatistics::numSamplesOld)
    .def("getSumChiSquareEst", &openpgl::VMMChiSquareComponentSplitter::ComponentSplitStatistics::getSumChiSquareEst)
    .def("getChiSquareEst", &openpgl::VMMChiSquareComponentSplitter::ComponentSplitStatistics::getChiSquareEst)
    .def("getHighestChiSquareIdx", &openpgl::VMMChiSquareComponentSplitter::ComponentSplitStatistics::getHighestChiSquareIdx)
    .def("decay", &openpgl::VMMChiSquareComponentSplitter::ComponentSplitStatistics::decay)
    .def("mergeComponentStats", &openpgl::VMMChiSquareComponentSplitter::ComponentSplitStatistics::mergeComponentStats)
    .def("getSplitMean", &openpgl::VMMChiSquareComponentSplitter::ComponentSplitStatistics::getSplitMean)
    .def("getSplitCovariance", &openpgl::VMMChiSquareComponentSplitter::ComponentSplitStatistics::getSplitCovariance)
    .def("__repr__", &openpgl::VMMChiSquareComponentSplitter::ComponentSplitStatistics::toString);


auto AdaptiveSplitAndMergeFactory = py::class_< openpgl::VMMAdaptiveSplitAndMergeFactory >(m, "VMMAdaptiveSplitAndMergeFactory")
    .def(py::init<>())
    .def("fit", &AdaptiveSplitAndMergeFactory_fit)
    .def("update", &AdaptiveSplitAndMergeFactory_update);


py::class_< openpgl::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration >(AdaptiveSplitAndMergeFactory, "ASMConfiguration")
    .def(py::init<>())
    .def_readwrite("weightedEMCfg", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration::weightedEMCfg)
    .def_readwrite("splittingThreshold", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration::splittingThreshold)
    .def_readwrite("mergingThreshold", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration::mergingThreshold)
    .def_readwrite("partialReFit", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration::partialReFit)
    .def_readwrite("maxSplitItr", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration::maxSplitItr)
    .def_readwrite("minSamplesForSplitting", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration::minSamplesForSplitting)
    .def_readwrite("minSamplesForPartialRefitting", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration::minSamplesForPartialRefitting)
    .def_readwrite("minSamplesForMerging", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration::minSamplesForMerging)
    .def("__repr__", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMConfiguration::toString);

py::class_< openpgl::VMMAdaptiveSplitAndMergeFactory::ASMStatistics >(AdaptiveSplitAndMergeFactory, "ASMStatistics")
    .def(py::init<>())
    .def_readwrite("sufficientStatistics", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMStatistics::sufficientStatistics)
    .def_readwrite("splittingStatistics", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMStatistics::splittingStatistics)
    .def("decay", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMStatistics::decay)
    //.def_readwrite("clear", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMStatistics::clear)
    //.def_readwrite("clearAll", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMStatistics::clearAll)
    .def("__repr__", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMStatistics::toString);


py::class_< openpgl::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics >(AdaptiveSplitAndMergeFactory, "ASMFittingStatistics")
    .def(py::init<>())
    .def_readwrite("numSamples", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics::numSamples)
    .def_readwrite("numSplits", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics::numSplits)
    .def_readwrite("numMerges", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics::numMerges)
    .def_readwrite("numComponents", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics::numComponents)
    .def_readwrite("numUpdateWEMIterations", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics::numUpdateWEMIterations)
    .def_readwrite("numPartialUpdateWEMIterations", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics::numPartialUpdateWEMIterations)
    .def("__repr__", &openpgl::VMMAdaptiveSplitAndMergeFactory::ASMFittingStatistics::toString);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}