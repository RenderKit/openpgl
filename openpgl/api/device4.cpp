#include "field/Device.h"

namespace openpgl {

IDevice* newDevice4() {
    return (IDevice*) new Device<4>();
}

}