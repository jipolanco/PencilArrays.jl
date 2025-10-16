# Changelog

The format is based on [Keep a Changelog] and [Common Changelog].

## [0.19.8] - 2025-10-16

### Fixed

- Fix deprecation warnings on Julia 1.12.

(Versions v0.19.1 to v0.19.7 missing...)

## [0.19.0] - 2023-07-14

### Changed

-   **Breaking:** change behaviour of `similar(u::PencilArray, [T], dims)` ([#83])

    When the `dims` argument is passed, we now try to return a new `PencilArray` instead of another (non-distributed) array type. Since this is only possible when `dims` matches the array size, an error is now thrown if that is not the case. This allows things to play nicely with other packages such as [StructArrays.jl](https://github.com/JuliaArrays/StructArrays.jl), which in some cases end up calling `similar` with the `dims` argument.

  [Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
  [Common Changelog]: https://common-changelog.org/
  [#83]: https://github.com/jipolanco/PencilArrays.jl/pull/83
