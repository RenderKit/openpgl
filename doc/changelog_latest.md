## Open PGL 0.8.0

- New (**Experimental**) Feature:
    - Volume Scatter Probability Guiding (VSPG):
        - This feature allows guiding the optimal volume scattering probability (VSP) and is based on Xu et al. work "Volume Scatter Probability Guiding".
        This **experimental** feature can be enabled by setting the CMake variable `OPENPGL_EF_VSP_GUIDING=ON`.
        The volume scattering probability for a given direction can be queried using the `VolumeScatterProbability` function of the `SurfaceSamplingDistribution` and `VolumeSamplingDistribution` classes.
- API changes:
    - `SampleData`:
    - New enum `ENextEventVolume` flag that identifies if the radiance stored in this sample comes from a volume or surface scatting event (e.g., if the next event is inside a volume or on a surface).
- API changes (`OPENPGL_EF_VSP_GUIDING=ON`):
    - `FieldConfig`:
        - `SetVarianceBasedVSP` when set to `true` the VSP value is calculated based on the `variance` and not the `contribution` of the nested volume and surface estimators. The default is `false` (i.e., `contribution`).
    - `VolumeScatterProbability` and `SurfaceSamplingDistribution`:
        - `VolumeScatterProbability` this function returns the optimal VSP probability for a given direction. Based on the type the VSP value is either calculated based on the `contribution` or the `variance` of the nested (surface and volume) estimators.
- API changes (`OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER=ON`):
    - `ImageSpaceGuidingBuffer`:
    Moving to a config-based initialization system and adding support to query the VSP for each pixel (i.e., primary ray) of the image-space guiding buffer.
    The `ImageSpaceGuidingBuffer` constructor now takes a `ImageSpaceGuidingBuffer::Config` instead of a `Point2i` parameter.
    - The function to query the estimate of a pixel's contribution got renamed from `ContributionEstimate` to `GetContributionEstimate`.
    
    - `ImageSpaceGuidingBuffer::Config`: Class for setting up the `ImageSpaceGuidingBuffer` (e.g., resolution, enabling contribution, or VSP buffers).
        - The `Config` constructor initializes the config class. It takes a `Point2i` defining the resolution of the desired `ImageSpaceGuidingBuffer`.
        - The resolution of the desired `ImageSpaceGuidingBuffer` can be queried using `GetResolution`.
        - Estimating the image contribution can be enabled or disables using `EnableContributionEstimate`.
        - If the estimation of the image contribution is enabled can be checked using `ContributionEstimate`.
        - The type of the estimated image contribution is defined via `SetContributionType`. The type is defined via the `PGLContributionTypes` enum and can be based on the contribution (`EContribContribution`) or variance (`EContribVariance`).
        - The type of the image contribution can be queried using `GetContributionType`.

- API changes (`OPENPGL_EF_IMAGE_SPACE_GUIDING_BUFFER=ON` and `OPENPGL_EF_VSP_GUIDING=ON`):
    - `ImageSpaceGuidingBuffer`: Adding the possibility to query the VSP value.
        - The VSP for a given pixel (i.e., primary ray) can be queried using the `GetVolumeScatterProbabilityEstimate` function.
    - `ImageSpaceGuidingBuffer::Config`:
        - Estimating the image space VSP values can be activated and deactivated using `EnableVolumeScatterProbabilityEstimate)`.
        - If estimating of the image-space VSP values is enabled can be checked using `VolumeScatterProbabilityEstimate`.
        - The type of the estimated image-space VSP values is defined via `SetVolumeScatterProbabilityType`. The type is defined via the `PGLVSPTypes` enum and can be based on the contribution (`EVSPContribution`) or variance (`EVSPVariance`).
        - The type of the image-space VSP values can be queried using `GetVolumeScatterProbabilityType`.