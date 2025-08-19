# Version History

## Open PGL 0.8.0

- New (**Experimental**) Feature:
  - Volume Scatter Probability Guiding (VSPG):
    - This feature allows guiding the optimal volume scattering
      probability (VSP) and is based on Xu et al. work “Volume Scatter
      Probability Guiding”. This **experimental** feature can be enabled
      by setting the CMake variable `OPENPGL_EF_VSP_GUIDING=ON`. The
      volume scattering probability for a given direction can be queried
      using the `VolumeScatterProbability` function of the
      `SurfaceSamplingDistribution` and `VolumeSamplingDistribution`
      classes.
- API changes:
  - `SampleData`:
  - New enum `ENextEventVolume` flag that identifies if the radiance
    stored in this sample comes from a volume or surface scatting event
    (e.g., if the next event is inside a volume or on a surface).
- API changes (`OPENPGL_EF_VSP_GUIDING=ON`):
  - `FieldConfig`:
    - `SetVarianceBasedVSP` when set to `true` the VSP value is
      calculated based on the `variance` and not the `contribution` of
      the nested volume and surface estimators. The default is `false`
      (i.e., `contribution`).
  - `VolumeScatterProbability` and `SurfaceSamplingDistribution`:
    - `VolumeScatterProbability` this function returns the optimal VSP
      probability for a given direction. Based on the type the VSP value
      is either calculated based on the `contribution` or the `variance`
      of the nested (surface and volume) estimators.
- API changes (`OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER=ON`):
  - `ImageSpaceGuidingBuffer`: Moving to a config-based initialization
    system and adding support to query the VSP for each pixel (i.e.,
    primary ray) of the image-space guiding buffer. The
    `ImageSpaceGuidingBuffer` constructor now takes a
    `ImageSpaceGuidingBuffer::Config` instead of a `Point2i` parameter.

  - The function to query the estimate of a pixel’s contribution got
    renamed from `ContributionEstimate` to `GetContributionEstimate`.

  - `ImageSpaceGuidingBuffer::Config`: Class for setting up the
    `ImageSpaceGuidingBuffer` (e.g., resolution, enabling contribution,
    or VSP buffers).

    - The `Config` constructor initializes the config class. It takes a
      `Point2i` defining the resolution of the desired
      `ImageSpaceGuidingBuffer`.
    - The resolution of the desired `ImageSpaceGuidingBuffer` can be
      queried using `GetResolution`.
    - Estimating the image contribution can be enabled or disables using
      `EnableContributionEstimate`.
    - If the estimation of the image contribution is enabled can be
      checked using `ContributionEstimate`.
    - The type of the estimated image contribution is defined via
      `SetContributionType`. The type is defined via the
      `PGLContributionTypes` enum and can be based on the contribution
      (`EContribContribution`) or variance (`EContribVariance`).
    - The type of the image contribution can be queried using
      `GetContributionType`.
- API changes (`OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER=ON` and
  `OPENPGL_EF_VSP_GUIDING=ON`):
  - `ImageSpaceGuidingBuffer`: Adding the possibility to query the VSP
    value.
    - The VSP for a given pixel (i.e., primary ray) can be queried using
      the `GetVolumeScatterProbabilityEstimate` function.
  - `ImageSpaceGuidingBuffer::Config`:
    - Estimating the image space VSP values can be activated and
      deactivated using `EnableVolumeScatterProbabilityEstimate)`.
    - If estimating of the image-space VSP values is enabled can be
      checked using `VolumeScatterProbabilityEstimate`.
    - The type of the estimated image-space VSP values is defined via
      `SetVolumeScatterProbabilityType`. The type is defined via the
      `PGLVSPTypes` enum and can be based on the contribution
      (`EVSPContribution`) or variance (`EVSPVariance`).
    - The type of the image-space VSP values can be queried using
      `GetVolumeScatterProbabilityType`.

## Open PGL 0.7.1

- Bugfixes:
  - Fixing invalidation of the guiding field on initial creation if a
    cell contains no samples
    [\#23](https://github.com/RenderKit/openpgl/issues/23).
  - Fixing noisy stdout printouts
    [\#19](https://github.com/RenderKit/openpgl/issues/19).
  - Improving robustness of the integer arithmetic used during the
    deterministic multi-threaded building of the spatial subdivision
    structure.
  - Improving numerical stability of fitting process of the VMM-based
    guiding models.

## Open PGL 0.7.0

- New (**Experimental**) Features:

  - Radiance Caching (RC):
    - If RC is enabled, the guiding structure (i.e., `Field`) learns an
      approximation of multiple radiance quantities (in linear RGB),
      such as outgoing and incoming radiance, irradiance, fluence, and
      in-scattered radiance. These quantities can be queried using the
      `SurfaceSamplingDistribution` and `VolumeSamplingDistribution`
      classes. RC support can be enabled using the
      `OPENPGL_EF_RADIANCE_CACHES` CMake option. **Note:** Since the RC
      quantities are Monte-Carlo estimates, zero-value samples
      (`ZeroValueSampleData`) that are generated during
      rendering/training have to be passed/stored in the `SampleStorage`
      as well.
  - Guided/Adjoint-driven Russian Roulette (GRR):
    - The information stored in radiance caches can be used to optimize
      stochastic path termination decisions (a.k.a. Russian roulette) to
      avoid a significant increase in variance (i.e., noise) caused by
      early terminations, which can occur when using standard
      throughput-based RR strategies. We, therefore, added to example
      implementation for guided
      (`openpgl::cpp::util::GuidedRussianRoulette(...)`) and standard
      (`openpgl::cpp::util::StandardThroughputBasedRussianRoulette(...)`)
      RR, which can be found in the `openpgl/cpp/RussianRoulette.h`
      header.
  - Image-space guiding buffer (ISGB):
    - The ISGB can be used to store and approximate per-pixel guiding
      information (e.g., a pixel estimate used in guided Russian
      roulette). The ISGB class
      (`openpgl::cpp::util::ImageSpaceGuidingBuffer`) is defined in the
      `openpgl/cpp/ImageSpaceGuidingBuffer.h` header file. The support
      can be enabled using the `OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER`
      CMake option.

- API changes:

  - `pgl_direction`: A new **wrapper** type for directional data. When
    using C++ `pgl_direction` can directly be assigned by and to
    `pgl_vec3f`.
  - `pgl_spectrum`: A new **wrapper** type for spectral (i.e., linear
    RGB) data. When using C++ `pgl_spectrum` can directly be assigned by
    and to `pgl_vec3f`.
  - `SampleData`:
    - New enum `EDirectLight` flag that identifies if the radiance
      stored in this sample comes directly from an emitter (e.g.,
      emissive surface, volume, or light source).
    - `direction`: Changes the type `pgl_vec3f` to `pgl_direction`.
  - `ZeroValueSampleData`: This new structure is a simplified and more
    compact representation of the `SampleData` struct representing a
    zero-value sample. It contains the following members:
    - `position`: The position of the sample (type `pgl_point3f`).
    - `direction`: The incoming direction of the sample (type
      `pgl_direction`).
    - `volume`: If the sample is a volume sample (type `bool`).
  - `SampleStorage`: To add, query, and get the number of
    `ZeroValueSampleData`, the following functions were added.
    - `AddZeroValueSample` and `AddZeroValueSamples`: These functions
      add one or multiple `ZeroValueSampleData`.
    - `GetSizeZeroValueSurface` and `GetSizeZeroValueVolume`: These
      functions return the number of collected/stored surface or volume
      `Ze1roValueSampleData`.
    - `GetZeroValueSampleSurface` and `GetZeroValueSampleVolume`: Return
      a given `ZeroValueSampleData` from either the surface or volume
      storage.

- API changes (`OPENPGL_EF_RADIANCE_CACHES=ON`): When the RC feature is
  enabled, additional functions and members are available for the
  following structures:

  - `SurfaceSamplingDistribution`:
    - `IncomingRadiance`: The incoming radiance estimate arriving at the
      current cache position from a specific direction.
    - `OutgoingRadiance`: The outgoing radiance at the current cache
      position to a specific direction.
    - `Irradiance`: The irradiance at the current cache position and for
      a given surface normal.
  - `VolumeSamplingDistribution`:
    - `IncomingRadiance`: The incoming radiance estimate arriving at the
      current cache position from a specific direction.
    - `OutgoingRadiance`: The outgoing radiance at the current cache
      position to a specific direction.
    - `InscatteredRadiance`: The in-scattered radiance at the current
      cache position to a specific direction and for a given HG mean
      cosine.
    - `Fluence`: The volume fluence at the current cache position.
  - `SampleData`:
    - `radianceIn`: The incoming radiance arriving at the sample
      position from `direction` (type `pgl_spectrum`).
    - `radianceInMISWeight`: The MIS weight of the `radianceIn` if the
      source of it is a light source, if not it is `1.0` (type `float`).
    - `directionOut`: The outgoing direction of the sample (type
      `pgl_direction`).
    - `radianceOut`: The outgoing radiance estimate of the sample (type
      `pgl_direction`).

  `ZeroValueSampleData`: - `directionOut`: The outgoing direction of the
  sample (type `pgl_direction`).

- API changes (`OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER=ON`): When the
  ISGB feature is enabled, additional functions and members are
  available for the following structures:

  - `ImageSpaceGuidingBuffer`: This is the main structure for storing
    image-space, per-pixel guiding information approximated from pixel
    samples. -`AddSample`: Add a pixel sample of type
    `ImageSpaceGuidingBuffer::Sample` to the buffer.
    - `Update`: Updates the image-space guiding
      information/approximations from the previously collected samples
      (e.g., denoises the pixel contribution estimates using OIDN). For
      efficiency reasons, it makes sense not to update the buffer after
      every rendering progression but in an exponential fashion (e.g.,
      at progression `2^0`,`2^1`,…,`2^N`).
    - `IsReady`: If the ISGB is ready (i.e., at least one `Update` step
      was performed).
    - `GetPixelContributionEstimate`: Returns the pixel contribution
      estimate for a given pixel, which can be used, for example, for
      guided RR.
    - `Reset`: Resets the ISGB.
  - `ImageSpaceGuidingBuffer::Sample`: This structure is used to store
    information about a per-pixel sample that is passed to the ISGB.
    - `contribution`: The contribution estimate of the pixel value of a
      given sample (type `pgl_vec3f`).
    - `albedo`: The albedo of the surface or the volume at the first
      scattering event (type `pgl_vec3f`).
    - `normal`: The normal at the first surface scattering event or the
      ray direction towards the cameras if the first event is a volume
      event (type `pgl_vec3f`).
    - `flags`: Bit encoded information about the sample (e.g., if the
      first scattering event is a volume event `Sample::EVolumeEvent`).

- Optimizations:

  - Compression for spectral and directional: To reduce the size of the
    `SampleData` and `ZeroValueSampleData` data types it is possible to
    enable 32-Bit compression, which is mainly advised when enabling the
    RC feature via `OPENPGL_EF_RADIANCE_CACHES=ON`.
    - `OPENPGL_DIRECTION_COMPRESSION`: Enables 32-Bit compression for
      `pgl_direction`.
    - `OPENPGL_RADIANCE_COMPRESSION`: Enables 32-Bit compression for
      `pgl_spectrum`.

- Bugfixes:

  - Numerical accuracy problem during sampling when using parametric
    mixtures.

- Platform support:

  - Added support for Windows on ARM (by [Anthony
    Roberts](https://github.com/anthony-linaro)
    [PR17](https://github.com/RenderKit/openpgl/pull/17)). **Note:**
    Requires using LLVM and `clang-cl.exe` as C and C++ compiler.

## Open PGL 0.6.0

- Api changes:
  - `Device` added `numThread` parameter (default = 0) to the
    constructor to set the number of threads used by `Open PGL` during
    training. The default value of `0` uses all threads provided by
    `TBB`. If the renderer uses `TBB` as well and regulates the thread
    count this count is also used by `Open PGL`.
  - `SurfaceSamplingDistribution` and `VolumeSamplingDistribution`:
    - Added `GetId` function to return the unique id of the spatial
      structure used to query the sampling distribution.
  - `Field` and `SampleStorage`, added `Compare` function to check if
    the data stored in different instances (e.g., generated by two
    separate runs) are similar (i.e., same spatial subdivisions and
    directional distributions).
  - `Field`:
    - The constructor of the `Field` class now takes a `FieldConfig`
      instead of a `PGLFieldArguments` object. **(BREAKING API CHANGE)**
    - `GetSurfaceStatistics` and `GetVolumeStatistics` functions are
      added to query statistics about the surface and volume guiding
      field. The functions return a `FieldStatistics` object. Note,
      querying the statistics of a `Field` introduces a small overhead.
  - `FieldStatistics`:
    - This class store different statistics about a `Field`, such as,
      number and size of spatial nodes, statistics about the directional
      distributions, and the times spend for full and separate steps of
      the last `Update` step. The statistics can be queried as a full
      string (useful for logging) or as CSV strings (useful for analysis
      and plotting).
    - `ToString`: Returns a string printing all statistics.
    - `HeaderCSVString`: Returns the CSV header sting with the names of
      each statistic.
    - `ToCSVString`: Returns the CSV value sting of each statistic.
  - `FieldConfig`:
    - This class is added to replace the `PGLFieldArguments` struct when
      using the C++ API. -`Init`: the function initializes the
      parameters of the `FieldConfig` (i.e., similar to
      `pglFieldArgumentsSetDefaults`). Additional parameters
      (`deterministic` and `maxSamplesPerLeaf`) are introduced to enable
      deterministic behavior and to control the spatial subdivision
      granularity. -`SetSpatialStructureArgMaxDepth`: this function can
      be called after `Init` to the the maximum tree depth of the
      spatial structure.
  - `pglFieldArgumentsSetDefaults`: Adding two additional parameters
    `deterministic` and `maxSamplesPerLeaf`. **(BREAKING API CHANGE)**
- Tools:
  - Added a set of command line tools which are build when enabling the
    `OPENPGL_BUILD_TOOLS` Cmake flag.
    - `openpgl_bench`: Tool to time different components of `Open PGL`
      such as the full training of a `Field` or the querying
      (initialization) of `SamplingDistributions`.
    - `openpgl_debug`: Tool to `validate` and `compare` stored
      `SampleStorage` and `Field` objects or retrain a `Field` from
      scratch using multiple stored sets (iterations) of stored samples.
- Optimizations:
  - Spatial structure (Kd-tree) build is now fully multithreaded. This
    improves training performance on machines with higher core counts,
    especially when using `deterministic` training.
  - Kd-tree switched to use cache-friendlier `TreeLets` instead of
    single `TreeNode` structures.
- Bugfixes:
  - `Field` fixed some non-deterministic behavior when spatial cache
    does not receive any training data during a training iteration due
    to a large number of training iterations.
  - Removed legacy/broken support for `OpenMP` threading since there is
    a dependency to `TBB` anyway.
  - Fixed build problems on (non-Mac) `ARM` systems.

## Open PGL 0.5.0

- Api changes:

  - `PathSegmentStorage`:
    - Removed support for splatting training samples due to the fact
      that knn-lookups have proven to be better. Therefore, the function
      attributes `splatSamples` and `sampler` have been removed from the
      `PrepareSamples` function.

    - Added `PropagateSamples` method prepare and push samples to the
      `SampleStorage` The goal is to replace `PrepareSamples`,
      `GetSamples` and `AddSamples`.
  - `Sampler`:
    - Removed since it is not used/needed anymore.
  - `SurfaceSamplingDistribution` and `VolumeSamplingDistribution`:
    - The usage of parallax-compensation is now connected to the guiding
      distribution type. Therefore the explicit
      `useParallaxCompensation` parameter is removed from the `Init`
      functions of the `SamplingDistributions`.  
    - Added `IncomingRadiancePDF` function that returns an approximation
      of the incoming radiance distribution. This PDF does not need to
      be related to the actual sampling PDF but can be used for
      Resampled Importance Sampling (RIS).
  - `Field`:
    - Adding `UpdateSurface` and `UpdateVolume` function to update/train
      the surface and volume field separately.
  - `SampleStorage`:
    - Adding `ClearSurface` and `ClearVolume` function to clear the
      surface and volume samples separately. This allows to wait until a
      specific number of samples is collected for the surface or volume
      cache before updating/fitting the `Field`.

- Deactivating/removing `OMP` threading support since it would still
  have a dependency on `TBB`

- Bugfixes:

  - Fixing bug causing crash during `Field::Update` when in previous
    iterations no volume or surface samples were present.

## Open PGL 0.4.1

- Bugfixes:
  - Fixing bug introduced in `0.4.0` when using
    `ApplySingleLobeHenyeyGreensteinProduct()` for VMM-based
    representations

## Open PGL 0.4.0

- Performance:

  - Optimized KNN lookup of guiding caches (x3 speed-up).
  - Optimized Cosine product for VMM based representations.

- Dependencies:

  - Removed the Embree library dependency for KNN lookups in favour of
    the header-only library nanoflann.

- Adding ARM Neon support (e.g., Apple M1).

- Fixing memory alignment bug for higher SIMD widths.

- `PathSegmentStorage`:

  - Fixing bug when multiple refracted/reflected events hit a distant
    source (i.e., environment map) by clamping to a max distance.
  - Adding `GetMaxDistance` and `SetMaxDistance` methods.
  - Adding `GetNumSegments` and `GetNumSamples` methods.

- `Field`:

  - Stopped tracing a total number of spp statistic since it is not
    really useful.
    - Removed the `GetTotalSPP` function.
    - Removed the `numPerPixelSamples` parameter from the `Update`
      function.

## Open PGL 0.3.1

- `Field`:
  - Added `Reset()` function to reset a guiding field (e.g., when the
    lighting or the scene geometry changed)
- `PathSegmentStorage`:
  - Fixed bug when using `AddSample()`

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
    stored path segment information represents the sampling behavior of
    the render (i.e., the resulting RGB value should match the pixel
    value the renderer adds to the framebuffer)

- `SurfaceSamplingDistribution`:

  - Added support for guiding based on the product of a normal-oriented
    cosine lobe and the incident radiance distribution:
    `(ApplyCosineProduct)` This feature is only supported for VMM-based
    directional distributions. Support can be checked with
    `SupportsApplyCosineProduct()`.

- `VolumeSamplingDistribution`:

  - Added support for guiding based on the product of a single lobe HG
    phase function and the incident radiance distribution:
    `ApplySingleLobeHenyeyGreensteinProduct()` This feature is only
    supported for VMM-based directional distributions. Support can be
    checked with `SupportsApplySingleLobeHenyeyGreensteinProduct()`.

## Open PGL 0.1.0

- Initial release of Open PGL Features:
  - Incremental learning/updating of a 5D spatio-directional radiance
    field from radiance samples (see `Field`).
  - Directional representation based on (parallax-aware) von
    Mises-Fisher mixtures.
  - `PathSegmentStorage` is a utility class to help keep track of all
    path segment information and generate radiance samples when a
    path/random walk is finished/terminated.
  - Support for guided importance sampling of directions on surfaces
    (see `SurfaceSamplingDistribution`) and inside volumes (see
    `VolumeSamplingDistribution`)
- Added C-API and C++-API headers
  - C-API: `#include <openpgl/openpgl.h>`
  - C++-API: `#include <openpgl/cpp/OpenPGL.h>` and the namespace
    `openpgl::cpp::`
