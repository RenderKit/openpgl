#include "device/Device.h"

namespace openpgl {

IDevice* newDeviceCPU4() {
    return (IDevice*) new Device<4>();
}

}