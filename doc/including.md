Including Open PGL into a project
=================================


Including into CMake build scripts. 
----------------------------------

To include Open PGL into a project which is using CMake as a build system, one can simply use the CMake configuration files provided by Open PGL. 

To make CMake aware of Open PGL's CMake configuration scripts the 
`openpgl_DIR` has to be set to their location during configuration:

```Bash
cmake -Dopenpgl_DIR=[openpgl_install]/lib/cmake/openpgl-<OPENPGL_VERSION> ..
```

After that, adding OpenPGL to a CMake project/target is done by first
finding Open PGL using `find_package()` and then adding the `openpgl:openpgl`
targets to the project/target: 

```CMake
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


Including Open PGL API headers
------------------------------

Open PGL offers two types of APIs.

The C API is C99 conform and is the basis for interacting with Open PGL. To use the C API of Open PGL, one only needs to include the following header: 

```C
#include <openpgl/openpgl.h>
```

The C++ API is a header-based wrapper of the C API, which offers a more comfortable, object-oriented way of using Open PGL. 
To use the C++ API of Open PGL, one only needs to include the following header:

```C++
#include <openpgl/cpp/OpenPGL.h>
```
