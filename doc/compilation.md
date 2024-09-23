Building Open PGL from source
=============================

The latest Open PGL sources are always available at the [Open PGL GitHub
repository](https://github.com/RenderKit/openpgl). The default `main` branch
should always point to the latest tested bugfix release.

Prerequisites
-------------

Open PGL currently supports Linux, Windows and MacOS. In addition, before
building Open PGL you need the following prerequisites:

-   You can clone the latest Open PGL sources via:

        git clone https://github.com/RenderKit/openpgl.git

-   To build Open PGL you need [CMake](http://www.cmake.org), any form of C++11 compiler (we recommend using GCC, but also support Clang and MSVC), and standard Linux development tools.

-   Open PGL depends on TBB, which is available at the [TBB GitHub
    repository](https://github.com/oneapi-src/oneTBB).

-   Open PGL depends on OIDN, if the **Image-space Guiding Buffer** feature is enabled, which is available at the [OIDN GitHub
    repository](https://github.com/RenderKit/oidn).

Depending on your Linux distribution, you can install these dependencies using `yum` or `apt-get`. Some of these packages might already be installed or might have slightly different names.

CMake Superbuild
----------------

For convenience, Open PGL provides a CMake Superbuild script which will pull
down Open PGL's dependencies and build Open PGL itself. The result is an install
directory including all dependencies.

Run with:

```bash
    mkdir build
    cd build
    cmake ../superbuild
    cmake  --build .
```

The resulting `install` directory (or the one set with `CMAKE_INSTALL_PREFIX`)
will have everything in it, with one subdirectory per dependency.

CMake options to note (all have sensible defaults):

- `CMAKE_INSTALL_PREFIX`: The root directory where everything gets installed to.
- `BUILD_JOBS`: Sets the number given to `make -j` for parallel builds.
- `BUILD_STATIC`: Builds Open PGL as static library (default `OFF`).
- `BUILD_TOOLS`: Builds Open PGL's tools (default `OFF`).
- `BUILD_DEPENDENCIES_ONLY`: Only builds Open PGL's dependencies (default `OFF`).
- `BUILD_TBB`: Builds or downloads TBB (default `ON`).
- `BUILD_TBB_FROM_SOURCE`: Specifies whether TBB should be built from source or the releases on GitHub should be used. This must be ON
   when compiling for ARM (default `OFF`).
- `BUILD_OIDN`: Builds or downloads Intel's Open Image Denoise (OIDN) (default `ON`).
- `BUILD_OIDN_FROM_SOURCE`: Builds OIDN from source. This must be ON when compiling for ARM. (default `ON`).
- `DOWNLOAD_ISPC`: Downloads Intel's ISPC compiler which is needed to build OIDN (default `ON` when building OIDN from source).

For the full set of options, run `ccmake [<PGL_ROOT>/superbuild]`.

Standard CMake build
--------------------

Assuming the above prerequisites are all fulfilled, building Open PGL through
CMake is easy:

Create a build directory, and go into it:

```bash
    mkdir build
    cd build
```

Configure the Open PGL build using:

```bash
    cmake -DCMAKE_INSTALL_PREFIX=[openpgl_install] ..
```

-  CMake options to note (all have sensible defaults):

    - `CMAKE_INSTALL_PREFIX`: The root directory where everything gets installed to.

    - `OPENPGL_BUILD_STATIC`: Builds Open PGL as a static or shared library (default `OFF`).

    - `OPENPGL_ISA_AVX512`: Compiles Open PGL with AVX-512 support (default `OFF`).

    - `OPENPGL_ISA_NEON` and `OPENPGL_ISA_NEON2X`: Compiles Open PGL with NEON or double
       pumped NEON support (default `OFF`).

    - `OPENPGL_LIBRARY_NAME`: Specifies the name of the Open PGL library file
        created. By default the name `openpgl` is used.

    - `OPENPGL_BUILD_STATIC`: Builds Open PGL as static library (default `OFF`).

    - `OPENPGL_BUILD_TOOLS`: Builds additional tools such as: `openpgl_bench` and `openpgl_debug` for benchmarking and debuging guiding caches (default `OFF`).

    - `OPENPGL_EF_RADIANCE_CACHES`: Enables the **experimental** radiance caching feature (default `OFF`).

    - `OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER`: Enables the **experimental** image-space guiding buffer feature (default `OFF`).

    - `OPENPGL_DIRECTION_COMPRESSION`: Enables the 32Bit compression for directional data stored in `pgl_direction` (default `OFF`).

    - `OPENPGL_RADIANCE_COMPRESSION`: Enables the 32Bit compression for RGB data stored in `pgl_spectrum` (default `OFF`).

    - `OPENPGL_TBB_ROOT`: Location of the TBB installation.

    - `OPENPGL_TBB_COMPONENT`: The name of the TBB component/library (default `tbb`).

Build and install Open PGL using:

```bash
    cmake build
    cmake install
```
