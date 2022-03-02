Open PGL API
============

The API specification of Open PGL is currently still in a "work in progress" and might change with the next releases - depending on the community feedback and library evolution.  

We therefore only give here a small overview of the C++ class structures and refer to the individual class header files for detailed information.

Device
------
```C++
#include <openpgl/cpp/Device.h>
```
The `Device` class is a key component of OpenPGL. The Device defines the backend used by Open PGL.
Currently OpenPGL supports different CPU backends using either SSE, AVX, or AVX-512 optimizations.

Note: support for different GPU backends is planned in future releases. 


Field
-----
```C++
#include <openpgl/cpp/Field.h>
```
The `Field` class is a key component of Open PGL. An instance of this class holds the spatio-directional guiding information (e.g., approximation of the incoming radiance field) for a scene. The Field is responsible for storing, learning and accessing the guiding information. This information can be the incidence radiance field across the whole scene learned from several training. The Field holds separate approximations for surface and volumetric radiance distributions which can be accessed separately. The representation of a scenes radiance distriubtion is usually separated into a positional and directional representation using a spatial subdivision structure, where each spatial leaf node (a.k.a. Region) contains a directional representation for the local incident radiance distribution.


SurfaceSamplingDistriubtion
---------------------------
```C++
#include <openpgl/cpp/SurfaceSamplingDistriubtion.h>
```
The `SurfaceSamplingDistriubtion` class represents the guiding distribution used for sampling directions on surfaces. The sampling distribution is often be proportional to the incoming radiance or its product with components of a BSDF model (e.g., cosine term). The class supports function for sampling and PDF evaluations. 


VolumeSamplingDistriubtion
--------------------------
```C++
#include <openpgl/cpp/VolumeSamplingDistriubtion.h>
```
 The `VolumeSamplingDistriubtion` class represents the guiding distriubtion used for sampling directions inside volumes. The sampling distribution is often proportional to the incoming radiance or to its product with the phase function (e.g., single lobe HG). The class supports function for sampling and PDF evaluations.


SampleData
----------
```C++
#include <openpgl/cpp/SampleData.h>
```
The `SampleData` struct represent a radiance sample (e.g., position, direction, value). Radiance samples
are generated during rendering and are used to train/update the guiding field (e.g., after each rendering progression). A SampleData object can be created at each vertex of a random walk/path. To fill-in the required information the whole path (from its end point to the current vertex) has to be collected and backpropagated.


SampleStorage
-----------------
```C++
#include <openpgl/cpp/SampleStorage.h>
```
The `SampleStorage` class is a storage container collecting all SampleData generated during rendering.
It stores the (radiance/photon) samples generated during rendering. The implementation is thread save and supports concurrent adding of samples from multiple threads. As a result only one instance of this container is needed per rendering process. The stored samples are later used by the Field class to train/learn the guiding field (i.e., radiance field) for a scene.  


PathSegmentStorage
-----------------
```C++
#include <openpgl/cpp/PathSegmentStorage.h>
```
The `PathSegmentStorage` is a utility class to help generating SampleData during 
the path/random walk generation process. For the construction of a path/walk each new PathSegment
is stored the PathSegmentStorage. When the walk is finished or terminated the -radiance- SampleData is 
generated using a back propagation process. The resulting samples can then be passed to the global SampleDataStorage. 


PathSegment
-----------------
```C++
#include <openpgl/cpp/PathSegment.h>
```
The `PathSegment` struct stores all required information for a path segment (e.g., position, direction, PDF, BSDF evaluation). A list of succeeding segments (stored in a `PathSegmentStorage`) can be used to generate SampleData for training the guiding Field.  








