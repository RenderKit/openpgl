#include "field/Device.h"

namespace openpgl {

IDevice* newDevice16() {
    return (IDevice*) new Device<16>();
}

}