"""BSplineFlow: Learned monotonic sigma schedule via local B-spline basis functions.

Compared to BézierFlow's global Bernstein basis, B-splines offer local support —
perturbing one coefficient only affects nearby segments. This enables independent
tuning of early (structure) vs late (detail) denoising phases without global ripple.

Same interface as BezierScheduler for drop-in comparison.
"""

from ltx_trainer.bsplineflow.scheduler import BSplineScheduler

__all__ = ["BSplineScheduler"]
