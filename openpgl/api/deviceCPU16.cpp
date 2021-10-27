#include "device/Device.h"

namespace openpgl {

IDevice* newDeviceCPU16() {
    return (IDevice*) new Device<16>();
}

}