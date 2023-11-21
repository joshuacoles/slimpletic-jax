from ggl import compute_reduced_quadrature_scheme
import json


def tc(r: int) -> dict:
    x, w, d = compute_reduced_quadrature_scheme(r)

    return {
        "x": x.tolist(),
        "w": w.tolist(),
        "d": d.tolist(),
    }


data = [tc(r) for r in range(1, 20)]

output_file = f"ggl-data-neu.json"
with open(output_file, 'w') as f:
    json.dump(data, f)
