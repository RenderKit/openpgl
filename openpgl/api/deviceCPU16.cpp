#include "device/Device.h"

namespace openpgl
{

IDevice *newDeviceCPU16(size_t numThreads)
{
    return (IDevice *)new Device<16>(numThreads);
}

}  // namespace openpgl