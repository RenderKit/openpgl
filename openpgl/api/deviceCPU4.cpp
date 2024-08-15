#include "device/Device.h"

namespace openpgl
{

IDevice *newDeviceCPU4(size_t numThreads)
{
    return (IDevice *)new Device<4>(numThreads);
}

}  // namespace openpgl