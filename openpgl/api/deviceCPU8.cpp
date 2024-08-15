#include "device/Device.h"

namespace openpgl
{

IDevice *newDeviceCPU8(size_t numThreads)
{
    return (IDevice *)new Device<8>(numThreads);
}

}  // namespace openpgl