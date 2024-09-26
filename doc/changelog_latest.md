## Open PGL 0.7.0

- New (**Experimental**) Features:
    - Radiance Caching (RC):
        - If RC is enabled, the guiding structure (i.e., `Field`) learns an approximation of multiple radiance quantities (in linear RGB), such as outgoing and incoming radiance, irradiance, fluence, and in-scattered radiance. These quantities can be queried using the `SurfaceSamplingDistribution` and `VolumeSamplingDistribution` classes.
        RC support can be enabled using the `OPENPGL_EF_RADIANCE_CACHES` CMake option. **Note:** Since the RC quantities are Monte-Carlo estimates, zero-value samples (`ZeroValueSampleData`) that are generated during rendering/training have to be passed/stored in the `SampleStorage` as well. 
    - Guided/Adjoint-driven Russian Roulette (GRR):
        - The information stored in radiance caches can be used to optimize stochastic path termination decisions (a.k.a. Russian roulette) to avoid a significant increase in variance (i.e., noise) caused by early terminations, which can occur when using standard throughput-based RR strategies.
        We, therefore, added to example implementation for guided (`openpgl::cpp::util::GuidedRussianRoulette(...)`) and standard (`openpgl::cpp::util::StandardThroughputBasedRussianRoulette(...)`) RR, which can be found in  the `openpgl/cpp/RussianRoulette.h` header.
    - Image-space guiding buffer (ISGB):
        - The ISGB can be used to store and approximate per-pixel guiding information (e.g., a pixel estimate used in guided Russian roulette). 
        The ISGB class (`openpgl::cpp::util::ImageSpaceGuidingBuffer`) is defined in the `openpgl/cpp/ImageSpaceGuidingBuffer.h` header file. 
        The support can be enabled using the `OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER` CMake option.

- API changes:
    - `pgl_direction`: A new **wrapper** type for directional data. When using C++ `pgl_direction` can directly be assigned by and to `pgl_vec3f`.
    - `pgl_spectrum`: A new **wrapper** type for spetral (i.e., linear RGB) data. When using C++ `pgl_spectrum` can directly be assigned by and to `pgl_vec3f`.
    - `SampleData`:
        - New enum `EDirectLight` flag that identifies if the radiance stored in this sample comes directly from an emitter (e.g., emissive surface, volume, or light source). 
        - `direction`: Changes the type `pgl_vec3f` to `pgl_direction`.
    - `ZeroValueSampleData`: This new structure is a simplified and more compact representation of the `SampleData` struct representing a zero-value sample. It contains the following members:
        - `position`: The position of the sample (type `pgl_point3f`).
        - `direction`: The incoming direction of the sample (type `pgl_direction`).
        - `volume`: If the sample is a volume sample (type `bool`).
    - `SampleStorage`: To add, query, and get the number of `ZeroValueSampleData`, the following functions were added.
        - `AddZeroValueSample` and `AddZeroValueSamples`: These functions add one or multiple `ZeroValueSampleData`.
        - `GetSizeZeroValueSurface` and `GetSizeZeroValueVolume`: These functions return the number of collected/stored surface or volume `Ze1roValueSampleData`.
        - `GetZeroValueSampleSurface` and `GetZeroValueSampleVolume`: Return a given `ZeroValueSampleData` from either the surface or volume storage.

- API changes (`OPENPGL_EF_RADIANCE_CACHES=ON`):
    When the RC feature is enabled, additional functions and members are available for the following structures:
    - `SurfaceSamplingDistribution`: 
        - `IncomingRadiance`: The incoming radiance estimate arriving at the current cache position from a specific direction.
        - `OutgoingRadiance`: The outgoing radiance at the current cache position to a specific direction.
        - `Irradiance`: The irradiance at the current cache position and for a given surface normal.

    - `VolumeSamplingDistribution`:
        - `IncomingRadiance`: The incoming radiance estimate arriving at the current cache position from a specific direction.
        - `OutgoingRadiance`: The outgoing radiance at the current cache position to a specific direction.
        - `InscatteredRadiance`: The in-scattered radiance at the current cache position to a specific direction and for a given HG mean cosine.
        - `Fluence`: The volume fluence at the current cache position.

    - `SampleData`:
        - `radianceIn`: The incoming radiance arriving at the sample position from `direction` (type `pgl_spectrum`).
        - `radianceInMISWeight`: The MIS weight of the `radianceIn` if the source of it is a light source, if not it is `1.0` (type `float`).
        - `directionOut`: The outgoing direction of the sample (type `pgl_direction`).
        - `radianceOut`: The outgoing radiance estimate of the sample (type `pgl_direction`).

     `ZeroValueSampleData`:
        - `directionOut`: The outgoing direction of the sample (type `pgl_direction`).

- API changes (`OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER=ON`):
    When the ISGB feature is enabled, additional functions and members are available for the following structures:
    - `ImageSpaceGuidingBuffer`: This is the main structure for storing image-space, per-pixel guiding information approximated from pixel samples.
        -`AddSample`: Add a pixel sample of type `ImageSpaceGuidingBuffer::Sample` to the buffer.
        - `Update`: Updates the image-space guiding information/approximations from the previously collected samples (e.g., denoises the pixel contribution estimates using OIDN). For efficiency reasons, it makes sense not to update the buffer after every rendering progression but in an exponential fashion (e.g., at progression `2^0`,`2^1`,...,`2^N`).
        - `IsReady`: If the ISGB is ready (i.e., at least one `Update` step was performed).
        - `GetPixelContributionEstimate`: Returns the pixel contibution estimate for a given pixel, which can be used, for example, for guided RR.
        - `Reset`: Resets the ISGB.

    - `ImageSpaceGuidingBuffer::Sample`: This structure is used to store information about a per-pixel sample that is passed to the ISGB.
        - `contribution`: The contribution estimate of the pixel value of a given sample (type `pgl_vec3f`).
        - `albedo`: The albedo of the surface or the volume at the first scattering event (type `pgl_vec3f`).
        - `normal`: The normal at the first surface scattering event or the ray dairection towards the camers if the first event is a volume event (type `pgl_vec3f`).
        - `flags`: Bit encoded information about the sample (e.g., if the first scattering event is a volume event `Sample::EVolumeEvent`).

- Optimizations:
    - Compression for spectral and directional: 
        To reduce the size of the `SampleData` and `ZeroValueSampleData` data types it is possible to enable 32-Bit compression, which is mainly adviced when enabling the RC feature via `OPENPGL_EF_RADIANCE_CACHES=ON`.
        - `OPENPGL_DIRECTION_COMPRESSION`: Enables 32-Bit compression for `pgl_direction`.
        - `OPENPGL_RADIANCE_COMPRESSION`: Enables 32-Bit compression for `pgl_spectrum`.
- Bugfixes:
    - Numerical accuracy problem during sampling when using parametric mixtures.

- Platform support:
    - Added support for Windows on ARM (by [Anthony Roberts](https://github.com/anthony-linaro) [PR17](https://github.com/RenderKit/openpgl/pull/17)). **Note:** Requires using LLVM and `clang-cl.exe` as C and C++ compiler.  