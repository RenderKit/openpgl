#include "field/Device.h"

namespace openpgl {

IDevice* newDevice8() {
    return (IDevice*) new Device<8>();
}

}