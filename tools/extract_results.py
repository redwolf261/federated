import json

with open('debug_output.json', 'r', encoding='utf-16') as f:
    data = json.load(f)

print("\n=== MOON DEBUG ===")
moon = data.get('moon_debug', {})
agg = moon.get('aggregate', {})
print(f"Feature normalization present: {moon.get('feature_normalization_present', 'N/A')}")
print(f"μ = {moon.get('mu')}, τ = {moon.get('temperature')}")
print(f"CE grad norm mean: {agg.get('ce_grad_norm_mean', 'N/A'):.6f}")
print(f"Contrastive grad norm mean: {agg.get('contrastive_grad_norm_mean', 'N/A'):.6f}")
print(f"Contrastive/CE ratio mean: {agg.get('contrastive_to_ce_ratio_mean', 'N/A'):.6f}")
print(f"Contrastive/CE ratio max: {agg.get('contrastive_to_ce_ratio_max', 'N/A'):.6f}")

print("\n=== SCAFFOLD DEBUG ===")
scaffold = data.get('scaffold_debug', {})
agg = scaffold.get('aggregate', {})
print(f"Equation: {scaffold.get('implemented_equation', 'N/A')}")
print(f"Grad norm mean: {agg.get('grad_norm_mean', 'N/A'):.6f}")
print(f"Control norm mean: {agg.get('control_norm_mean', 'N/A'):.6f}")
print(f"Control/Grad ratio mean: {agg.get('control_to_grad_ratio_mean', 'N/A'):.6f}")
print(f"Control/Grad ratio max: {agg.get('control_to_grad_ratio_max', 'N/A'):.6f}")
print(f"c_global norm final: {agg.get('c_global_norm_final', 'N/A'):.6f}")

print("\n✓ Analysis complete")
