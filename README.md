# Intel® Open Path Guiding Library

This is release v0.7.0 of Intel® Open PGL. For changes and new features,
see the [changelog](CHANGELOG.md). Visit http://www.openpgl.org for more
information.

# Overview

The Intel® Open Path Guiding Library (Intel® Open PGL) implements a set
of representations and training algorithms needed to integrate path
guiding into a renderer. Open PGL offers implementations of current
state-of-the-art path guiding methods, which increase the sampling
quality and, therefore, the efficiency of a renderer. The goal of Open
PGL is to provide implementations that are well tested and robust enough
to be used in a production environment.

The representation of the guiding field is learned during rendering and
updated on a per-frame basis using radiance/importance samples generated
during rendering. At each vertex of a random path/walk, the guiding
field is queried for a local distribution (e.g., incident radiance),
guiding local sampling decisions (e.g., directions).

Currently supported path guiding methods include: guiding directional
sampling decisions on surfaces and inside volumes based on a learned
incident radiance distribution or its product with BSDF components
(i.e., cosine lobe) or phase functions (i.e., single lobe HG).

Open PGL offers a C API and a C++ wrapper API for higher-level
abstraction. The current implementation is optimized for the latest
Intel® processors with support for SSE, AVX, AVX2, and AVX-512
instructions.
<!--, and for ARM processors with support for NEON instructions.-->

Open PGL is part of the [Intel® oneAPI Rendering
Toolkit](https://software.intel.com/en-us/rendering-framework) and has
been released under the permissive [Apache 2.0
license](http://www.apache.org/licenses/LICENSE-2.0).

|                                                                ![Example rendering without and with Open PGL](/doc/images/example.png)                                                                 |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Path traced image of a variation of the Nishita Sky Demo scene from Blender Studio (CC0) without and with using Open PGL to guide directional samples (i.e., on surfaces and inside the water volume). |

# Disclaimer

The current version of Open PGL is still in a pre v1.0 stage and should
be used with caution in any production related environment. The API
specification is still in flux and might change with upcoming releases.

# Latest Updates

The full version history can be found [here](./CHANGELOG.md)

## Open PGL 0.7.0

- New (**Experimental**) Features:

  - Radiance Caching (RC):
    - If RC is enabled, the guiding structure (i.e., `Field`) learns an
      approximation of multiple radiance quantities (in linear RGB)
      ,such as outgoing and incoming radiance, irradiance, fluence, and
      in-scattered radiance. These quantities can be queried using the
      `SurfaceSamplingDistribution` and `VolumeSamplingDistribution`
      classes. RC support can be enabled using the
      `OPENPGL_EF_RADIANCE_CACHES` CMake option. **Note:** Since the RC
      quantities are Monte-Carlo estimates, zero-value samples
      (`ZeroValueSampleData`) that are generated during
      rendering/training have to be passed/stored in the `SampleStorage`
      as well.
  - Guided/Adjoint-driven Russian Roulette (GRR):
    - The information stored in radiance caches can be used to optimize
      stochastic path termination decisions (a.k.a. Russian roulette) to
      avoid a significant increase in variance (i.e., noise) caused by
      early terminations, which can occur when using standard
      throughput-based RR strategies. We, therefore, added to example
      implementation for guided
      (`openpgl::cpp::util::GuidedRussianRoulette(...)`) and standard
      (`openpgl::cpp::util::StandardThroughputBasedRussianRoulette(...)`)
      RR, which can be found in the `openpgl/cpp/RussianRoulette.h`
      header.
  - Image-space guiding buffer (ISGB):
    - The ISGB can be used to store and approximate per-pixel guiding
      information (e.g., a pixel estimate used in guided Russian
      roulette). The ISGB class
      (`openpgl::cpp::util::ImageSpaceGuidingBuffer`) is defined in the
      `openpgl/cpp/ImageSpaceGuidingBuffer.h` header file. The support
      can be enabled using the `OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER`
      CMake option.

- API changes:

  - `pgl_direction`: A new **wrapper** type for directional data. When
    using C++ `pgl_direction` can directly be assigned by and to
    `pgl_vec3f`.
  - `pgl_spectrum`: A new **wrapper** type for spetral (i.e., linear
    RGB) data. When using C++ `pgl_spectrum` can directly be assigned by
    and to `pgl_vec3f`.
  - `SampleData`:
    - New enum `EDirectLight` flag that identifies if the radiance
      stored in this sample comes directly from an emitter (e.g.,
      emissive surface, volume, or light source).
    - `direction`: Changes the type `pgl_vec3f` to `pgl_direction`.
  - `ZeroValueSampleData`: This new structure is a simplified and more
    compact representation of the `SampleData` struct representing a
    zero-value sample. It contains the following members:
    - `position`: The position of the sample (type `pgl_point3f`).
    - `direction`: The incoming direction of the sample (type
      `pgl_direction`).
    - `volume`: If the sample is a volume sample (type `bool`).
  - `SampleStorage`: To add, query, and get the number of
    `ZeroValueSampleData`, the following functions were added.
    - `AddZeroValueSample` and `AddZeroValueSamples`: These functions
      add one or multiple `ZeroValueSampleData`.
    - `GetSizeZeroValueSurface` and `GetSizeZeroValueVolume`: These
      functions return the number of collected/stored surface or volume
      `Ze1roValueSampleData`.
    - `GetZeroValueSampleSurface` and `GetZeroValueSampleVolume`: Return
      a given `ZeroValueSampleData` from either the surface or volume
      storage.

- API changes (`OPENPGL_EF_RADIANCE_CACHES=ON`): When the RC feature is
  enabled, additional functions and members are available for the
  following structures:

  - `SurfaceSamplingDistribution`:
    - `IncomingRadiance`: The incoming radiance estimate arriving at the
      current cache position from a specific direction.
    - `OutgoingRadiance`: The outgoing radiance at the current cache
      position to a specific direction.
    - `Irradiance`: The irradiance at the current cache position and for
      a given surface normal.
  - `VolumeSamplingDistribution`:
    - `IncomingRadiance`: The incoming radiance estimate arriving at the
      current cache position from a specific direction.
    - `OutgoingRadiance`: The outgoing radiance at the current cache
      position to a specific direction.
    - `InscatteredRadiance`: The in-scattered radiance at the current
      cache position to a specific direction and for a given HG mean
      cosine.
    - `Fluence`: The volume fluence at the current cache position.
  - `SampleData`:
    - `radianceIn`: The incoming radiance arriving at the sample
      position from `direction` (type `pgl_spectrum`).
    - `radianceInMISWeight`: The MIS weight of the `radianceIn` if the
      source of it is a light source, if not it is `1.0` (type `float`).
    - `directionOut`: The outgoing direction of the sample (type
      `pgl_direction`).
    - `radianceOut`: The outgoing radiance estimate of the sample (type
      `pgl_direction`).

  `ZeroValueSampleData`: - `directionOut`: The outgoing direction of the
  sample (type `pgl_direction`).

- API changes (`OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER=ON`): When the
  ISGB feature is enabled, additional functions and members are
  available for the following structures:

  - `ImageSpaceGuidingBuffer`: This is the main structure for storing
    image-space, per-pixel guiding information approximated from pixel
    samples. -`AddSample`: Add a pixel sample of type
    `ImageSpaceGuidingBuffer::Sample` to the buffer.
    - `Update`: Updates the image-space guiding
      information/approximations from the previously collected samples
      (e.g., denoises the pixel contribution estimates using OIDN). For
      efficiency reasons, it makes sense not to update the buffer after
      every rendering progression but in an exponential fashion (e.g.,
      at progression `2^0`,`2^1`,…,`2^N`).
    - `IsReady`: If the ISGB is ready (i.e., at least one `Update` step
      was performed).
    - `GetPixelContributionEstimate`: Returns the pixel contibution
      estimate for a given pixel, which can be used, for example, for
      guided RR.
    - `Reset`: Resets the ISGB.
  - `ImageSpaceGuidingBuffer::Sample`: This structure is used to store
    information about a per-pixel sample that is passed to the ISGB.
    - `contribution`: The contribution estimate of the pixel value of a
      given sample (type `pgl_vec3f`).
    - `albedo`: The albedo of the surface or the volume at the first
      scattering event (type `pgl_vec3f`).
    - `normal`: The normal at the first surface scattering event or the
      ray dairection towards the camers if the first event is a volume
      event (type `pgl_vec3f`).
    - `flags`: Bit encoded information about the sample (e.g., if the
      first scattering event is a volume event `Sample::EVolumeEvent`).

- Optimizations:

  - Compression for spectral and directional: To reduce the size of the
    `SampleData` and `ZeroValueSampleData` data types it is possible to
    enable 32-Bit compression, which is mainly adviced when enabling the
    RC feature via `OPENPGL_EF_RADIANCE_CACHES=ON`.
    - `OPENPGL_DIRECTION_COMPRESSION`: Enables 32-Bit compression for
      `pgl_direction`.
    - `OPENPGL_RADIANCE_COMPRESSION`: Enables 32-Bit compression for
      `pgl_spectrum`.

- Bugfixes:

  - Numerical accuracy problem during sampling when using parametric
    mixtures.

- Platform support:

  - Added support for Windows on ARM (by [Anthony
    Roberts](https://github.com/anthony-linaro)
    [PR17](https://github.com/RenderKit/openpgl/pull/17)). **Note:**
    Requires using LLVM and `clang-cl.exe` as C and C++ compiler.

# Support and Contact

Open PGL is under active development. Though we do our best to guarantee
stable release versions, a certain number of bugs, as-yet-missing
features, inconsistencies, or any other issues are still possible.
Should you find any such issues, please report them immediately via
[Open PGL’s GitHub Issue
Tracker](https://github.com/OpenPathGuidingLibrary/openpgl/issues) (or,
if you should happen to have a fix for it, you can also send us a pull
request).

# Reference

``` code
@misc{openpgl,
   Author = {Herholz, Sebastian and Dittebrandt, Addis},
   Year = {2022},
   Note = {http://www.openpgl.org},
   Title = {Intel{\textsuperscript{\tiny\textregistered}}
 Open Path Guiding Library}
}
```

# Building Open PGL from source

The latest Open PGL sources are always available at the [Open PGL GitHub
repository](https://github.com/RenderKit/openpgl). The default `main`
branch should always point to the latest tested bugfix release.

## Prerequisites

Open PGL currently supports Linux, Windows and MacOS. In addition,
before building Open PGL you need the following prerequisites:

- You can clone the latest Open PGL sources via:

      git clone https://github.com/RenderKit/openpgl.git

- To build Open PGL you need [CMake](http://www.cmake.org), any form of
  C++11 compiler (we recommend using GCC, but also support Clang and
  MSVC), and standard Linux development tools.

- Open PGL depends on TBB, which is available at the [TBB GitHub
  repository](https://github.com/oneapi-src/oneTBB).

- Open PGL depends on OIDN, if the **Image-space Guiding Buffer**
  feature is enabled, which is available at the [OIDN GitHub
  repository](https://github.com/RenderKit/oidn).

Depending on your Linux distribution, you can install these dependencies
using `yum` or `apt-get`. Some of these packages might already be
installed or might have slightly different names.

## CMake Superbuild

For convenience, Open PGL provides a CMake Superbuild script which will
pull down Open PGL’s dependencies and build Open PGL itself. The result
is an install directory including all dependencies.

Run with:

``` bash
    mkdir build
    cd build
    cmake ../superbuild
    cmake  --build .
```

The resulting `install` directory (or the one set with
`CMAKE_INSTALL_PREFIX`) will have everything in it, with one
subdirectory per dependency.

CMake options to note (all have sensible defaults):

- `CMAKE_INSTALL_PREFIX`: The root directory where everything gets
  installed to.
- `BUILD_JOBS`: Sets the number given to `make -j` for parallel builds.
- `BUILD_STATIC`: Builds Open PGL as static library (default `OFF`).
- `BUILD_TOOLS`: Builds Open PGL’s tools (default `OFF`).
- `BUILD_DEPENDENCIES_ONLY`: Only builds Open PGL’s dependencies
  (default `OFF`).
- `BUILD_TBB`: Builds or downloads TBB (default `ON`).
- `BUILD_TBB_FROM_SOURCE`: Specifies whether TBB should be built from
  source or the releases on GitHub should be used. This must be ON when
  compiling for ARM (default `OFF`).
- `BUILD_OIDN`: Builds or downloads Intel’s Open Image Denoise (OIDN)
  (default `ON`).
- `BUILD_OIDN_FROM_SOURCE`: Builds OIDN from source. This must be ON
  when compiling for ARM. (default `ON`).
- `DOWNLOAD_ISPC`: Downloads Intel’s ISPC compiler which is needed to
  build OIDN (default `ON` when building OIDN from source).

For the full set of options, run `ccmake [<PGL_ROOT>/superbuild]`.

## Standard CMake build

Assuming the above prerequisites are all fulfilled, building Open PGL
through CMake is easy:

Create a build directory, and go into it:

``` bash
    mkdir build
    cd build
```

Configure the Open PGL build using:

``` bash
    cmake -DCMAKE_INSTALL_PREFIX=[openpgl_install] ..
```

- CMake options to note (all have sensible defaults):

  - `CMAKE_INSTALL_PREFIX`: The root directory where everything gets
    installed to.

  - `OPENPGL_BUILD_STATIC`: Builds Open PGL as a static or shared
    library (default `OFF`).

  - `OPENPGL_ISA_AVX512`: Compiles Open PGL with AVX-512 support
    (default `OFF`).

  - `OPENPGL_ISA_NEON` and `OPENPGL_ISA_NEON2X`: Compiles Open PGL with
    NEON or double pumped NEON support (default `OFF`).

  - `OPENPGL_LIBRARY_NAME`: Specifies the name of the Open PGL library
    file created. By default the name `openpgl` is used.

  - `OPENPGL_BUILD_STATIC`: Builds Open PGL as static library (default
    `OFF`).

  - `OPENPGL_BUILD_TOOLS`: Builds additional tools such as:
    `openpgl_bench` and `openpgl_debug` for benchmarking and debuging
    guiding caches (default `OFF`).

  - `OPENPGL_EF_RADIANCE_CACHES`: Enables the **experimental** radiance
    caching feature (default `OFF`).

  - `OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER`: Enables the
    **experimental** image-space guiding buffer feature (default `OFF`).

  - `OPENPGL_DIRECTION_COMPRESSION`: Enables the 32Bit compression for
    directional data stored in `pgl_direction` (default `OFF`).

  - `OPENPGL_RADIANCE_COMPRESSION`: Enables the 32Bit compression for
    RGB data stored in `pgl_spectrum` (default `OFF`).

  - `OPENPGL_TBB_ROOT`: Location of the TBB installation.

  - `OPENPGL_TBB_COMPONENT`: The name of the TBB component/library
    (default `tbb`).

Build and install Open PGL using:

``` bash
    cmake build
    cmake install
```

# Including Open PGL into a project

## Including into CMake build scripts.

To include Open PGL into a project which is using CMake as a build
system, one can simply use the CMake configuration files provided by
Open PGL.

To make CMake aware of Open PGL’s CMake configuration scripts the
`openpgl_DIR` has to be set to their location during configuration:

``` bash
cmake -Dopenpgl_DIR=[openpgl_install]/lib/cmake/openpgl-0.7.0 ..
```

After that, adding OpenPGL to a CMake project/target is done by first
finding Open PGL using `find_package()` and then adding the
`openpgl:openpgl` targets to the project/target:

``` cmake
# locating Open PGL library and headers 
find_package(openpgl REQUIRED)

# setting up project/target
...
add_executable(myProject ...)
...

# adding Open PGL to the project/target
target_include_directories(myProject openpgl::openpgl)

target_link_libraries(myProject openpgl::openpgl)
```

## Including Open PGL API headers

Open PGL offers two types of APIs.

The C API is C99 conform and is the basis for interacting with Open PGL.
To use the C API of Open PGL, one only needs to include the following
header:

``` c
#include <openpgl/openpgl.h>
```

The C++ API is a header-based wrapper of the C API, which offers a more
comfortable, object-oriented way of using Open PGL. To use the C++ API
of Open PGL, one only needs to include the following header:

``` c++
#include <openpgl/cpp/OpenPGL.h>
```

# Open PGL API

The API specification of Open PGL is currently still in a “work in
progress” stage and might change with the next releases - depending on
the community feedback and library evolution.

We, therefore, only give here a small overview of the C++ class
structures and refer to the individual class header files for detailed
information.

## Device

``` c++
#include <openpgl/cpp/Device.h>
```

The `Device` class is a key component of OpenPGL. It defines the backend
used by Open PGL. OpenPGL supports different CPU backends using SSE,
AVX, or AVX-512 optimizations.

Note: support for different GPU backends is planned in future releases.

## Field

``` c++
#include <openpgl/cpp/Field.h>
```

The `Field` class is a key component of Open PGL. An instance of this
class holds the spatio-directional guiding information (e.g.,
approximation of the incoming radiance field) for a scene. The `Field`
is responsible for storing, learning, and accessing the guiding
information. This information can be the incidence radiance field
learned from several training iterations across the whole scene. The
`Field` holds separate approximations for surface and volumetric
radiance distributions, which can be accessed separately. The
representation of a scene’s radiance distribution is usually separated
into a positional and directional representation using a spatial
subdivision structure. Each spatial leaf node (a.k.a. Region) contains a
directional representation for the local incident radiance distribution.

## SurfaceSamplingDistribution

``` c++
#include <openpgl/cpp/SurfaceSamplingDistribution.h>
```

The `SurfaceSamplingDistribution` class represents the guiding
distribution used for sampling directions on surfaces. The sampling
distribution is often proportional to the incoming radiance distribution
or its product with components of a BSDF model (e.g., cosine term). The
class supports functions for sampling and PDF evaluations.

## VolumeSamplingDistribution

``` c++
#include <openpgl/cpp/VolumeSamplingDistribution.h>
```

The `VolumeSamplingDistribution` class represents the guiding
distribution used for sampling directions inside volumes. The sampling
distribution is often proportional to the incoming radiance distribution
or its product with the phase function (e.g., single lobe HG). The class
supports functions for sampling and PDF evaluations.

## SampleData

``` c++
#include <openpgl/cpp/SampleData.h>
```

The `SampleData` struct represents a radiance sample (e.g., position,
direction, value). Radiance samples are generated during rendering and
are used to train/update the guiding field (e.g., after each rendering
progression). A `SampleData` object is created at each vertex of a
random walk/path. To collect the data at a specific vertex, the whole
path (from its endpoint to the current vertex) must be considered, and
information (e.g., radiance) must be backpropagated.

## SampleStorage

``` c++
#include <openpgl/cpp/SampleStorage.h>
```

The `SampleStorage` class is a storage container collecting all
SampleData generated during rendering. It stores the (radiance/photon)
samples generated during rendering. The implementation is thread save
and supports concurrent adding of samples from multiple threads. As a
result, only one instance of this container is needed per rendering
process. The stored samples are later used by the Field class to
train/learn the guiding field (i.e., radiance field) for a scene.

## PathSegmentStorage

``` c++
#include <openpgl/cpp/PathSegmentStorage.h>
```

The `PathSegmentStorage` is a utility class to help generate multiple
`SampleData` objects during the path/random walk generation process. For
the construction of a path/walk, each new `PathSegment` is stored in the
`PathSegmentStorage`. When the walk is finished or terminated, the
-radiance- SampleData is generated using a backpropagation process. The
resulting samples are then be passed to the global `SampleDataStorage`.

Note: The `PathSegmentStorage` is just a utility class meaning its usage
is not required. It is possible to for the users to use their own method
for generating `SampleData` objects during rendering.

## PathSegment

``` c++
#include <openpgl/cpp/PathSegment.h>
```

The `PathSegment` struct stores all required information for a path
segment (e.g., position, direction, PDF, BSDF evaluation). A list of
succeeding segments (stored in a `PathSegmentStorage`) is used to
generate `SampleData` for training the guiding field.

# Projects that make use of Open PGL

TBA

# Projects that are closely related to Open PGL

- The [Intel® oneAPI Rendering
  Toolkit](https://software.intel.com/en-us/rendering-framework)

- The [Intel® Embree](http://embree.github.io) Ray Tracing Kernel
  Framework
