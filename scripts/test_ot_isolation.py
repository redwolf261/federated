import numpy as np
from flex_persona.similarity.robust_wasserstein_distance import RobustWassersteinDistanceCalculator

# Synthetic test cases for OT

def print_ot_result(a, b, M, label):
    ot_calc = RobustWassersteinDistanceCalculator()
    try:
        dist = ot_calc._wasserstein_with_sinkhorn(a, b, M, reg=0.08)
        print(f"{label} | Sinkhorn: {dist}")
    except Exception as e:
        print(f"{label} | Sinkhorn failed: {e}")
    try:
        dist = ot_calc._wasserstein_with_pot(a, b, M)
        print(f"{label} | EMD: {dist}")
    except Exception as e:
        print(f"{label} | EMD failed: {e}")
    try:
        dist = ot_calc._wasserstein_with_linprog(a, b, M)
        print(f"{label} | LP: {dist}")
    except Exception as e:
        print(f"{label} | LP failed: {e}")

# Overlapping support
a1 = np.array([0.5, 0.5, 0, 0])
b1 = np.array([0.25, 0.25, 0.25, 0.25])
M1 = np.array([
    [0, 1, 2, 3],
    [1, 0, 1, 2],
    [2, 1, 0, 1],
    [3, 2, 1, 0],
], dtype=np.float64)
print_ot_result(a1, b1, M1, "Overlap")

# Disjoint support
a2 = np.array([1, 0, 0, 0])
b2 = np.array([0, 0, 0, 1])
M2 = M1.copy()
print_ot_result(a2, b2, M2, "Disjoint")

# Nearly disjoint (small epsilon)
a3 = np.array([0.999, 0.001, 0, 0])
b3 = np.array([0, 0, 0.001, 0.999])
print_ot_result(a3, b3, M1, "Nearly Disjoint")

# Uniform support
a4 = np.array([0.25, 0.25, 0.25, 0.25])
b4 = np.array([0.25, 0.25, 0.25, 0.25])
print_ot_result(a4, b4, M1, "Uniform")
