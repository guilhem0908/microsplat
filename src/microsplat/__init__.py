"""3D Gaussian Splatting written from the equations and checked numerically.

``reference`` is a NumPy rasteriser that serves as the specification, ``torch_renderer`` is its
differentiable tile-based twin, ``fit`` optimises a scene against posed images, and ``verify``
holds the cross-checks between them.
"""

__version__ = "0.1.0"
