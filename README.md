# Intel® Open Path Guiding Library

This is release v0.1.0 of Intel® Open PGL. For changes and new features
see the [changelog](CHANGELOG.md). Visit http://www.openpgl.org for more
information.

# Overview

The Intel® Open Path Guiding Library (Intel® Open PGL) implements a set
of representations and training algorithms needed to integrate path
guiding into a renderer. Open PGL offers implementations of current
state-of-the-art path guiding methods which can be used to increase the
sampling quality, and therefore the efficiency of a renderer. The goal
of Open PGL is to provide implementations that well tested and robust
enough to be used in a production environment.

The representation of the guiding field is learned during rendering and
can be updated on a per frame basis. Radiance/importance samples which
are generated during rendering by the renderer passed to Open PGL to
train and update a guiding field covering all surfaces and volumes of
the current scene. For each vertex of a random path/walk the guiding
field can be queried for a local distriubtion (e.g., incident radiance)
which can be used to guide local sampling decisions (e.g., directions).

Currently supported path guiding methods include: guiding directional
sampling decisions on surfaces as well as inside volumes based on a
learned incident radiance distribution or its product with BSDF
components (i.e., cosine lobe) or phase functions (i.e., single lobe
HG).

Open PGL offers a C API as well as a C++ wrapper API for higher level
abstraction. The current implementation is optimized for the latest
Intel® processors with support for SSE, AVX, AVX2, and AVX-512
instructions, and for ARM processors with support for NEON instructions.
Open PGL is part of the [Intel® oneAPI Rendering
Toolkit](https://software.intel.com/en-us/rendering-framework) and is
released under the permissive [Apache 2.0
license](http://www.apache.org/licenses/LICENSE-2.0).

# Version History

## Open PGL 0.3.0

  - Added CMake superbuild script to build Open PGL including all it
    dependencies.  
    The dependencies (e.g., TBB and Embree) are downloaded, built, and
    installed automatically.

  - Added support for different SIMD optimizations (SSE, AVX2, AVX-512).
    The optimization type can be chosen when initializing the `Device`.

  - Added support for directional quad trees for the directional
    representation.

  - `PathSegmentStorage`:
    
      - Added debug function `CalculatePixelEstimate` to validate if the
        stored path segment information represent the sampling behavior
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

  - Added `vklSetParam()` API function which can set parameters of any
    supported type
  - Structured regular volumes:
      - Added support for cell-centered data via the `cellCentered`
        parameter; vertex-centered remains the default
      - Added support for more general transformations via the
        `indexToObject` parameter
      - Added `indexOrigin` parameter which applies an index-space vec3i
        translation

# Building Open PGL from source

The latest Open PGL sources are always available at the [Open PGL GitHub
repository](http://github.com/openpathguidinglibrary/openpgl). The
default `main` branch should always point to the latest tested bugfix
release.

## Prerequisites

Open PGL currently supports Linux and Windows. In addition, before you
can build Open PGL you need the following prerequisites:

  - You can clone the latest Open PGL sources via:
    
        git clone https://github.com/openpathguidinglibrary/openpgl.git

  - To build Open PGL you need [CMake](http://www.cmake.org), any form
    of C++11 compiler (we recommend using GCC, but also support Clang
    and MSVC), and standard Linux development tools.

  - Open PGL depends on Embree, which is available at the [Embree GitHub
    repository](https://github.com/embree/embree).

  - Open PGL depends on TBB, which is available at the [TBB GitHub
    repository](https://github.com/oneapi-src/oneTBB).

Depending on your Linux distribution you can install these dependencies
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

Configure the Open PGL build:

``` bash
        cmake -DCMAKE_INSTALL_PREFIX=[openpgl_install] ..
```

  - CMake options to note (all have sensible defaults):
    
      - `CMAKE_INSTALL_PREFIX` will be the root directory where
        everything gets installed.
    
      - `BUILD_STATIC` if Open PGL should be build as static or shared
        library (default `OFF`)
    
      - `ISA_AVX512` if Open PGL is compiled with AVX-512 support
        (default `OFF`).
    
      - `embree_DIR` location of the Embree CMake configuration file
        (e.g., \[embree\_install\]/lib/cmake/embree-3.6.1)
    
      - `TBB_ROOT` location of the TBB installation.

Build and install Open PGL:

``` bash
        cmake build
        cmake install
```

# Including Open PGL into a project

## Including into CMake build scripts

`-Dopenpgl_DIR=[openpgl_install]/lib/cmake/openpgl-0.1.0`

``` cmake
# locating Open PGL library and headers 
FIND_PACKAGE(openpgl REQUIRED)

# setting up project/target
...
...

# adding Open PGL to the project/target
target_include_directories([project] openpgl::openpgl)

target_link_libraries([project] openpgl::openpgl)
```

## Including Open PGL API headers

Open PGL offers two types of APIs

The C API is C99 conform and is the basis for interacting with Open PGL.
To use the C API of Open PGL one only needs to include the following
header:

``` c
#include <openpgl/openpgl.h>
```

The C++ API is a header based wrapper of the C API which offers a more
comfortable, object-oriented way of using Open PGL. To use the C++ API
of Open PGL one only needs to include the following header:

``` c++
#include <openpgl/cpp/OpenPGL.h>
```

# Open PGL API

The API specification of Open PGL is currently still in a “work in
progress” and might change with the next releases - depending on the
community feedback and library evolution.

We therefore only give here a small overview of the C++ class structures
and refer to the individual class header files for detailed information.

## Device

``` c++
#include <openpgl/cpp/Device.h>
```

## Field

``` c++
#include <openpgl/cpp/Field.h>
```

## SurfaceSamplingDistriubtion

``` c++
#include <openpgl/cpp/SurfaceSamplingDistriubtion.h>
```

## VolumeSamplingDistriubtion

``` c++
#include <openpgl/cpp/VolumeSamplingDistriubtion.h>
```

## SampleData

``` c++
#include <openpgl/cpp/SampleData.h>
```

## SampleDataStorage

``` c++
#include <openpgl/cpp/SampleDataStorage.h>
```

# Support and Contact

Open PGL is under active development, and though we do our best to
guarantee stable release versions a certain number of bugs,
as-yet-missing features, inconsistencies, or any other issues are still
possible. Should you find any such issues please report them immediately
via [Open PGL’s GitHub Issue
Tracker](https://github.com/OpenPathGuidingLibrary/openpgl/issues) (or,
if you should happen to have a fix for it, you can also send us a pull
request).
