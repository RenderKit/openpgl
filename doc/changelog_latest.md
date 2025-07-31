## Open PGL 0.8.0

- Bugfixes:
    - Fixing invalidation of the guiding field on initial creation if a cell contains no samples [#23](https://github.com/RenderKit/openpgl/issues/23).
    - Fixing noisy stdout printouts [#19](https://github.com/RenderKit/openpgl/issues/19).
    - Improving robustness of the integer arithmetric used during the deterministic multi-threaded building of the spatial subdivision structure.
    - Improving numerical stability of fitting process of the VMM-based guiding models.
