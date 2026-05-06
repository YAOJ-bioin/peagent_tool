# Third-Party Notices

## scBasset

`peagent_tool` vendors a minimal TensorFlow model-construction subset adapted
from Calico's scBasset project so existing PEAgent `.h5` model weights can be
loaded without copying the full scBasset checkout to deployment servers.

- Upstream: https://github.com/calico/scBasset
- License: BSD 3-Clause License
- Copyright: Calico Life Sciences LLC and scBasset contributors

The vendored code is limited to the inference model architecture used by
`scbasset.utils.make_model`.
