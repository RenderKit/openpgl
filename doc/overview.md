Overview
========

The Intel® Open Path Guiding Library (Intel® Open PGL) implements a set of representations and training algorithms needed to integrate path guiding
into a renderer. Open PGL offers implementations of current state-of-the-art path guiding methods which can be used to increase the sampling quality, and therefore the efficiency of a renderer. The goal of Open PGL is to provide implementations that well tested and robust enough to be used in a production environment. 

The representation of the guiding field is learned during rendering and can be updated on a per frame basis. Radiance/importance samples which are generated during rendering by the renderer passed to Open PGL to train and update a guiding field
covering all surfaces and volumes of the current scene. For each vertex of a random path/walk the guiding field can be queried for a local distriubtion (e.g., incident radiance) which can be used to guide local sampling decisions (e.g., directions).

Currently supported path guiding methods include: guiding directional sampling decisions on surfaces as well as inside volumes based on a learned incident radiance distribution or its product with BSDF components (i.e., cosine lobe) or phase functions (i.e., single lobe HG).

Open PGL offers a C API as well as a C++ wrapper API for higher level abstraction. 
The current implementation is optimized for the latest Intel® processors with support for SSE, AVX, AVX2, and AVX-512 instructions.
<!--, and for ARM processors with support for NEON instructions.--> 

Open PGL is part of the [Intel® oneAPI Rendering Toolkit](https://software.intel.com/en-us/rendering-framework) and is released
under the permissive [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0).
