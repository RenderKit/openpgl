Building Open PGL from source
=============================

The latest Open PGL sources are always available at the [Open PGL GitHub
repository](http://github.com/openpathguidinglibrary/openpgl). The default `main` branch
should always point to the latest tested bugfix release.

Prerequisites
-------------

Open PGL currently supports Linux and Windows. In addition, before
building Open PGL you need the following prerequisites:

-   You can clone the latest Open PGL sources via:

        git clone https://github.com/openpathguidinglibrary/openpgl.git

-   To build Open PGL you need [CMake](http://www.cmake.org), any form of C++11 compiler (we recommend using GCC, but also support Clang and MSVC), and standard Linux development tools.

-   Open PGL depends on Embree, which is available at the [Embree GitHub
    repository](https://github.com/embree/embree).

-   Open PGL depends on TBB, which is available at the [TBB GitHub
    repository](https://github.com/oneapi-src/oneTBB).

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

- `CMAKE_INSTALL_PREFIX` will be the root directory where everything gets
  installed.
- `BUILD_JOBS` sets the number given to `make -j` for parallel builds.
- `INSTALL_IN_SEPARATE_DIRECTORIES` toggles installation of all libraries in
  separate or the same directory.
- `BUILD_TBB_FROM_SOURCE` specifies whether TBB should be built from source or the releases on GitHub should be used. This must be ON
   when compiling for ARM.

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

    - `CMAKE_INSTALL_PREFIX` will be the root directory where everything gets installed.

    - `OPENPGL_BUILD_STATIC` if Open PGL should be built as a static or shared library (default `OFF`).

    - `OPENPGL_ISA_AVX512` if Open PGL is compiled with AVX-512 support (default `OFF`).

    - `embree_DIR` location of the Embree CMake configuration file 
    (e.g., [embree_install]/lib/cmake/embree-3.6.1).

    - `OPENPGL_TBB_ROOT` location of the TBB installation.

Build and install Open PGL using:

```bash
        cmake build
        cmake install
```
