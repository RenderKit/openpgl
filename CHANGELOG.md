Version History
===============
## Open PGL 0.5.0

- Api changes:
    - `PathSegmentStorage`:
        - Removed support for splatting training samples due
          to the fact that knn-lookups have proven to be better.
          Therefore, the function attributes `splatSamples` and `sampler`
          have been removed from the `PrepareSamples` function.

    - `Sampler`:
        - Removed since it is not used/needed anymore.

    - `SurfaceSamplingDistribution` and `VolumeSamplingDistribution`:
        - The usage of parallax-compensation is now connected to the guiding distribution type. Therefore the explicit `useParallaxCompensation` parameter is removed from the `Init` functions of the `SamplingDistributions`.  
        - Added `IncomingRadiancePDF` function that returns an approximation of the incoming radiance distribution. 
          This PDF does not need to be related to the actual sampling PDF but can be used for Resampled Importance Sampling (RIS).

    - `Field`:
        - Adding `UpdateSurface` and `UpdateVolume` function to update/train the surface and volume field separately.
    
    - `SampleStorage`:
        - Adding `ClearSurface` and `ClearVolume` function to clear the surface and volume samples separately.
          This allows to wait until a specific number of samples is collected for the surface or volume cache before updating/fitting the `Field`.


## Open PGL 0.4.1
- Bugfixes:
    - Fixing bug introduced in `0.4.0` when using
     `ApplySingleLobeHenyeyGreensteinProduct()` for VMM-based 
     representations

## Open PGL 0.4.0

-   Performance:
    - Optimized KNN lookup of guiding caches (x3 speed-up).
    - Optimized Cosine product for VMM based representations.

-   Dependencies:
      - Removed the Embree library dependency for KNN lookups in favour
        of the header-only library nanoflann.

-   Adding ARM Neon support (e.g., Apple M1).

-   Fixing memory alignment bug for higher SIMD widths.

-   `PathSegmentStorage`:
    - Fixing bug when multiple refracted/reflected events hit a
      distant source (i.e., environment map) by clamping to a max
      distance.
    - Adding `GetMaxDistance` and `SetMaxDistance` methods.
    - Adding `GetNumSegments` and `GetNumSamples` methods.

-   `Field`:
    - Stopped tracing a total number of spp statistic since it is not really
      useful.
        - Removed the `GetTotalSPP` function.
        - Removed the `numPerPixelSamples` parameter from the `Update` function.

## Open PGL 0.3.1

-   `Field`:
    - Added `Reset()` function to reset a guiding field (e.g., when the lighting or the scene 
        geometry changed)

-   `PathSegmentStorage`:
    - Fixed bug when using `AddSample()`

## Open PGL 0.3.0

-   Added CMake Superbuild script to build Open PGL, including all its dependencies.      
    The dependencies (e.g., TBB and Embree) are downloaded, built, and installed automatically. 

-   Added support for different SIMD optimizations (SSE, AVX2, AVX-512).
        The optimization type can be chosen when initializing the `Device`.

-   Added support for directional quadtrees for the directional representation.

-   `PathSegmentStorage`:
    -   Added debug function `CalculatePixelEstimate` to validate if the stored
        path segment information represents the sampling behavior of the render
        (i.e., the resulting RGB value should match the pixel value the renderer
        adds to the framebuffer)

-   `SurfaceSamplingDistribution`:
    -   Added support for guiding based on the product of a normal-oriented
        cosine lobe and the incident radiance distribution:
        `(ApplyCosineProduct)`
        This feature is only supported for VMM-based directional distributions.
        Support can be checked with `SupportsApplyCosineProduct()`.

-   `VolumeSamplingDistribution`:
    -   Added support for guiding based on the product of a single lobe
        HG phase function and the incident radiance distribution:
        `ApplySingleLobeHenyeyGreensteinProduct()`
        This feature is only supported for VMM-based directional distributions.
        Support can be checked with `SupportsApplySingleLobeHenyeyGreensteinProduct()`.


## Open PGL 0.1.0

-   Initial release of Open PGL
    Features:
    -   Incremental learning/updating of a 5D spatio-directional radiance field
        from radiance samples (see `Field`).
    -   Directional representation based on (parallax-aware) von Mises-Fisher mixtures.
    -   `PathSegmentStorage` is a utility class to help keep track of all path segment 
        information and generate radiance samples when a path/random walk is finished/terminated.
    -   Support for guided importance sampling of directions on surfaces (see `SurfaceSamplingDistribution`)
        and inside volumes (see `VolumeSamplingDistribution`)

-   Added C-API and C++-API headers
    -   C-API: `#include <openpgl/openpgl.h>`
    -   C++-API: `#include <openpgl/cpp/OpenPGL.h>` and the namespace `openpgl::cpp::`

    