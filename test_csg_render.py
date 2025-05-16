"""Quick test of CSG rendering with OpenSCAD"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from util.sdf import sdf_sphere, sdf_box
from util.sdf import sdf_render_csg
from util.types import CSGRenderConfig

# Create output directory
output_dir = Path("reports/test_csg")
output_dir.mkdir(parents=True, exist_ok=True)

# Test sphere rendering
def sphere_sdf(points):
    return sdf_sphere(points, center=np.array([0.0, 0.0, 0.0]), radius=0.5)

config = CSGRenderConfig(
    grid_size=80,
    bounds=(-1.0, 1.0),
    save_path=str(output_dir / "sphere_test.png"),
    image_size=(800, 800),
    camera_rotation=(45, 20, 0),
    colorscheme="Cornfield",
    show_edges=True
)

print("Rendering sphere...")
sphere_path = sdf_render_csg(sphere_sdf, config)
print(f"Sphere rendered: {sphere_path}")

# Test box rendering
def box_sdf(points):
    return sdf_box(points, center=np.array([0.0, 0.0, 0.0]), dims=np.array([0.8, 0.6, 0.4]))

config.save_path = str(output_dir / "box_test.png")
print("Rendering box...")
box_path = sdf_render_csg(box_sdf, config)
print(f"Box rendered: {box_path}")

print("CSG rendering test completed!")