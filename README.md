# Intel® Open Path Guiding Library

This is release v0.3.0 of Intel® Open PGL. For changes and new features,
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
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Path traced image of a variation of the Nishita Sky Demo scene from Blender Studio (CC0) without and with using Open PGL to guide directional samples (i.e., on surfaces and inside the water volume). |

# Disclaimer

The current version of Open PGL is still in a beta stage and should be
used with caution in any production related environment. The API
specification is still in flux and might change with upcoming releases.

# Version History

## Open PGL 0.3.0

  - Added CMake Superbuild script to build Open PGL, including all its
    dependencies.  
    The dependencies (e.g., TBB and Embree) are downloaded, built, and
    installed automatically.

  - Added support for different SIMD optimizations (SSE, AVX2, AVX-512).
    The optimization type can be chosen when initializing the `Device`.

  - Added support for directional quadtrees for the directional
    representation.

  - `PathSegmentStorage`:
    
      - Added debug function `CalculatePixelEstimate` to validate if the
        stored path segment information represents the sampling behavior
        of the render (i.e., the resulting RGB value should match the
        pixel value the renderer adds to the framebuffer)

  - `SurfaceSamplingDistribution`:
    
      - Added support for guiding based on the product of a
        normal-oriented cosine lobe and the incident radiance
        distribution: `ApplySingleLobeHenyeyGreensteinProduct()` This
        feature is only supported for VMM-based directional
        distributions. Support can be checked with
        `SupportsApplyCosineProduct()`.

  - `VolumeSamplingDistribution`:
    
      - Added support for guiding based on the product of a single lobe
        HG phase function and the incident radiance distribution:
        `ApplyCosineProduct()` This feature is only supported for
        VMM-based directional distributions. Support can be checked with
        `SupportsApplySingleLobeHenyeyGreensteinProduct()`.

## Open PGL 0.1.0

  - Initial release of Open PGL Features:
      - Incremental learning/updating of a 5D spatio-directional
        radiance field from radiance samples (see `Field`).
      - Directional representation based on (parallax-aware) von
        Mises-Fisher mixtures.
      - `PathSegmentStorage` is a utility class to help keep track of
        all path segment information and generate radiance samples when
        a path/random walk is finished/terminated.
      - Support for guided importance sampling of directions on surfaces
        (see `SurfaceSamplingDistribution`) and inside volumes (see
        `VolumeSamplingDistribution`)
  - Added C-API and C++-API headers
      - C-API: `#include <openpgl/openpgl.h>`
      - C++-API: `#include <openpgl/cpp/OpenPGL.h>` and the namespace
        `openpgl::cpp::`

# Support and Contact

Open PGL is under active development. Though we do our best to guarantee
stable release versions, a certain number of bugs, as-yet-missing
features, inconsistencies, or any other issues are still possible.
Should you find any such issues, please report them immediately via
[Open PGL’s GitHub Issue
Tracker](https://github.com/OpenPathGuidingLibrary/openpgl/issues) (or,
if you should happen to have a fix for it, you can also send us a pull
request).

# Building Open PGL from source

The latest Open PGL sources are always available at the [Open PGL GitHub
repository](http://github.com/openpathguidinglibrary/openpgl). The
default `main` branch should always point to the latest tested bugfix
release.

## Prerequisites

Open PGL currently supports Linux and Windows. In addition, before
building Open PGL you need the following prerequisites:

  - You can clone the latest Open PGL sources via:
    
        git clone https://github.com/openpathguidinglibrary/openpgl.git

  - To build Open PGL you need [CMake](http://www.cmake.org), any form
    of C++11 compiler (we recommend using GCC, but also support Clang
    and MSVC), and standard Linux development tools.

  - Open PGL depends on Embree, which is available at the [Embree GitHub
    repository](https://github.com/embree/embree).

  - Open PGL depends on TBB, which is available at the [TBB GitHub
    repository](https://github.com/oneapi-src/oneTBB).

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

  - `CMAKE_INSTALL_PREFIX` will be the root directory where everything
    gets installed.
  - `BUILD_JOBS` sets the number given to `make -j` for parallel builds.
  - `INSTALL_IN_SEPARATE_DIRECTORIES` toggles installation of all
    libraries in separate or the same directory.
  - `BUILD_TBB_FROM_SOURCE` specifies whether TBB should be built from
    source or the releases on GitHub should be used. This must be ON
    when compiling for ARM.

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
    
      - `CMAKE_INSTALL_PREFIX` will be the root directory where
        everything gets installed.
    
      - `BUILD_STATIC` if Open PGL should be built as a static or shared
        library (default `OFF`).
    
      - `ISA_AVX512` if Open PGL is compiled with AVX-512 support
        (default `OFF`).
    
      - `embree_DIR` location of the Embree CMake configuration file
        (e.g., \[embree\_install\]/lib/cmake/embree-3.6.1).
    
      - `TBB_ROOT` location of the TBB installation.

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
cmake -Dopenpgl_DIR=[openpgl_install]/lib/cmake/openpgl-0.3.0 ..
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
