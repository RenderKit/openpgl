Overview
========

The Intel® Open Path Guiding Library (Intel® Open PGL) implements a set of representations and training algorithms needed to integrate path guiding into a renderer. Open PGL offers implementations of current state-of-the-art path guiding methods, which increase the sampling quality and, therefore, the efficiency of a renderer. The goal of Open PGL is to provide implementations that are well tested and robust enough to be used in a production environment.

The representation of the guiding field is learned during rendering and updated on a per-frame basis using radiance/importance samples generated during rendering. At each vertex of a random path/walk, the guiding field is queried for a local distribution (e.g., incident radiance), guiding local sampling decisions (e.g., directions).

Currently supported path guiding methods include: guiding directional sampling decisions on surfaces and inside volumes based on a learned incident radiance distribution or its product with BSDF components (i.e., cosine lobe) or phase functions (i.e., single lobe HG).

Open PGL offers a C API and a C++ wrapper API for higher-level abstraction. 
The current implementation is optimized for the latest Intel® processors with support for SSE, AVX, AVX2, and AVX-512 instructions.
<!--, and for ARM processors with support for NEON instructions.--> 

Open PGL is part of the [Intel® oneAPI Rendering Toolkit](https://software.intel.com/en-us/rendering-framework) and has been released
under the permissive [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0).
