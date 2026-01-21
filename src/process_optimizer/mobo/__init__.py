"""Multi-objective Bayesian optimization (MOBO).

Currently supports EHVI-style MOBO via Ax/BoTorch when optional deps are installed.
"""

from .ax_ehvi import suggest_mobo_ax  # noqa: F401
