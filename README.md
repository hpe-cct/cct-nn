# cct-nn

Deep learning library for the HPE Cognitive Computing Toolkit (CCT). The NN library layers
a `DifferentiableField` type on top of regular CCT fields. A `DifferentiableField` behaves
like a normal field in most contexts, but includes additional state that allow it to carry
gradients. Operators on `DifferentiableField` types include both forward and backward
functions. The NN toolkit can propagate gradients from a loss function back through a
directed acyclic graph of differentiable fields and operators.

If you're new to CCT, start with the [tutorial](https://github.com/hpe-cct/cct-tutorial).
