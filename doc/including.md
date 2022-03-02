Including Open PGL into a project
=================================


Including into CMake build scripts 
----------------------------------

`-Dopenpgl_DIR=[openpgl_install]/lib/cmake/openpgl-<OPENPGL_VERSION>`

```CMake
# locating Open PGL library and headers 
FIND_PACKAGE(openpgl REQUIRED)

# setting up project/target
...
...

# adding Open PGL to the project/target
target_include_directories([project] openpgl::openpgl)

target_link_libraries([project] openpgl::openpgl)
```

Including Open PGL API headers
------------------------------

Open PGL offers two types of APIs

The C API is C99 conform and is the basis for interacting with Open PGL. To use the C API of Open PGL one only needs to include the following header: 

```C
#include <openpgl/openpgl.h>
```

The C++ API is a header based wrapper of the C API which offers a more comfortable, object-oriented way of using Open PGL. 
To use the C++ API of Open PGL one only needs to include the following header:

```C++
#include <openpgl/cpp/OpenPGL.h>
```