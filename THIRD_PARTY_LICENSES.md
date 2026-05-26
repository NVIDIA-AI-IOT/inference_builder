# Third-Party Licenses

This document lists third-party open source software source code that is
included in this repository and may be redistributed with it.

| Component | Location | Upstream | License |
| --- | --- | --- | --- |
| Triton Inference Server common | `dependencies/triton-inference-server/common` | <https://github.com/triton-inference-server/common> | BSD-3-Clause; see [`dependencies/triton-inference-server/common/LICENSE`](dependencies/triton-inference-server/common/LICENSE) |

## Dependency Notes

Python packages listed in `requirements.txt` are not vendored in this source
repository; they are resolved by the package manager during install or build.
If a release artifact, binary package, or container image redistributes those
packages, include the applicable third-party license notices with that artifact.
