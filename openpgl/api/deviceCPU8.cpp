#include "device/Device.h"

namespace openpgl {

IDevice* newDeviceCPU8() {
    return (IDevice*) new Device<8>();
}

}