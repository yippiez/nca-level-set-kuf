# %% [markdown]
# # Visualization of All SDF Shapes
# This notebook visualizes all available SDF shapes using a grey and white color scheme.

# %%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from util.sdf import (
    sdf_sphere, sdf_pill, sdf_box, sdf_torus,
    sdf_cylinder, sdf_cone, sdf_octahedron, sdf_pyramid,
    sdf_hexagonal_prism, sdf_ellipsoid, sdf_rounded_box,
    sdf_link, sdf_star,
    sdf_render_csg, sdf_render_csg_animation
)
from util.types import CSGRenderConfig
from util.paths import get_reports_dir
from PIL import Image, ImageDraw, ImageFont

# Create output directory
OUTPUT_DIR = get_reports_dir("sdf_all_visualize")

# %% [markdown]
# ## Define Common Render Configuration

# %%
# Grey and white color scheme as default
DEFAULT_CONFIG = CSGRenderConfig(
    grid_size=60,
    bounds=(-1.5, 1.5),
    image_size=(400, 400),
    camera_rotation=(30, 45, 0),
    colorscheme="Metallic"  # Grey/silver metallic color scheme
)

ANIMATION_CONFIG = CSGRenderConfig(
    grid_size=50,
    bounds=(-1.5, 1.5),
    image_size=(400, 400),
    camera_rotation=(30, 45, 0),
    n_frames=36,
    fps=10,
    colorscheme="Metallic"  # Grey/silver metallic color scheme
)

# %% [markdown]
# ## Basic Shapes

# %%
# 1. Sphere
def sphere_sdf(points):
    return sdf_sphere(points, center=np.array([0, 0, 0]), radius=0.5)

config_sphere = DEFAULT_CONFIG.model_copy()
config_sphere.save_path = str(OUTPUT_DIR / "sphere.png")
sphere_path = sdf_render_csg(sphere_sdf, config_sphere)
print(f"Sphere rendered: {sphere_path}")

# %%
# 2. Box/Cube
def box_sdf(points):
    return sdf_box(points, center=np.array([0, 0, 0]), dims=np.array([0.8, 0.8, 0.8]))

config_box = DEFAULT_CONFIG.model_copy()
config_box.save_path = str(OUTPUT_DIR / "box.png")
box_path = sdf_render_csg(box_sdf, config_box)
print(f"Box rendered: {box_path}")

# %%
# 3. Cylinder
def cylinder_sdf(points):
    return sdf_cylinder(points, center=np.array([0, 0, 0]), radius=0.4, height=1.0)

config_cylinder = DEFAULT_CONFIG.model_copy()
config_cylinder.save_path = str(OUTPUT_DIR / "cylinder.png")
cylinder_path = sdf_render_csg(cylinder_sdf, config_cylinder)
print(f"Cylinder rendered: {cylinder_path}")

# %%
# 4. Pill/Capsule
def pill_sdf(points):
    return sdf_pill(points, p1=np.array([-0.5, 0, 0]), p2=np.array([0.5, 0, 0]), radius=0.3)

config_pill = DEFAULT_CONFIG.model_copy()
config_pill.save_path = str(OUTPUT_DIR / "pill.png")
pill_path = sdf_render_csg(pill_sdf, config_pill)
print(f"Pill rendered: {pill_path}")

# %%
# 5. Torus
def torus_sdf(points):
    return sdf_torus(points, center=np.array([0, 0, 0]), r_major=0.5, r_minor=0.2)

config_torus = DEFAULT_CONFIG.model_copy()
config_torus.save_path = str(OUTPUT_DIR / "torus.png")
torus_path = sdf_render_csg(torus_sdf, config_torus)
print(f"Torus rendered: {torus_path}")

# %% [markdown]
# ## Advanced Shapes

# %%
# 6. Cone
def cone_sdf(points):
    return sdf_cone(points, tip=np.array([0, 0.5, 0]), base_center=np.array([0, -0.5, 0]), radius=0.5)

config_cone = DEFAULT_CONFIG.model_copy()
config_cone.save_path = str(OUTPUT_DIR / "cone.png")
cone_path = sdf_render_csg(cone_sdf, config_cone)
print(f"Cone rendered: {cone_path}")

# %%
# 7. Octahedron
def octahedron_sdf(points):
    return sdf_octahedron(points, center=np.array([0, 0, 0]), radius=0.7)

config_octahedron = DEFAULT_CONFIG.model_copy()
config_octahedron.save_path = str(OUTPUT_DIR / "octahedron.png")
octahedron_path = sdf_render_csg(octahedron_sdf, config_octahedron)
print(f"Octahedron rendered: {octahedron_path}")

# %%
# 8. Pyramid
def pyramid_sdf(points):
    return sdf_pyramid(points, tip=np.array([0, 0.5, 0]), base_center=np.array([0, -0.5, 0]), base_size=0.8)

config_pyramid = DEFAULT_CONFIG.model_copy()
config_pyramid.save_path = str(OUTPUT_DIR / "pyramid.png")
pyramid_path = sdf_render_csg(pyramid_sdf, config_pyramid)
print(f"Pyramid rendered: {pyramid_path}")

# %%
# 9. Hexagonal Prism
def hex_prism_sdf(points):
    return sdf_hexagonal_prism(points, center=np.array([0, 0, 0]), radius=0.5, height=0.8)

config_hex_prism = DEFAULT_CONFIG.model_copy()
config_hex_prism.save_path = str(OUTPUT_DIR / "hexagonal_prism.png")
hex_prism_path = sdf_render_csg(hex_prism_sdf, config_hex_prism)
print(f"Hexagonal prism rendered: {hex_prism_path}")

# %%
# 10. Ellipsoid
def ellipsoid_sdf(points):
    return sdf_ellipsoid(points, center=np.array([0, 0, 0]), radii=np.array([0.6, 0.4, 0.5]))

config_ellipsoid = DEFAULT_CONFIG.model_copy()
config_ellipsoid.save_path = str(OUTPUT_DIR / "ellipsoid.png")
ellipsoid_path = sdf_render_csg(ellipsoid_sdf, config_ellipsoid)
print(f"Ellipsoid rendered: {ellipsoid_path}")

# %% [markdown]
# ## Special Shapes

# %%
# 11. Rounded Box
def rounded_box_sdf(points):
    return sdf_rounded_box(points, center=np.array([0, 0, 0]), dims=np.array([0.8, 0.8, 0.8]), radius=0.15)

config_rounded_box = DEFAULT_CONFIG.model_copy()
config_rounded_box.save_path = str(OUTPUT_DIR / "rounded_box.png")
rounded_box_path = sdf_render_csg(rounded_box_sdf, config_rounded_box)
print(f"Rounded box rendered: {rounded_box_path}")

# %%
# 12. Chain Link
def link_sdf(points):
    return sdf_link(points, center=np.array([0, 0, 0]), r_major=0.3, r_minor=0.1, length=0.6)

config_link = DEFAULT_CONFIG.model_copy()
config_link.save_path = str(OUTPUT_DIR / "link.png")
link_path = sdf_render_csg(link_sdf, config_link)
print(f"Link rendered: {link_path}")

# %%
# 13. Star
def star_sdf(points):
    return sdf_star(points, center=np.array([0, 0, 0]), n_points=5, r_outer=0.6, r_inner=0.3, thickness=0.2)

config_star = DEFAULT_CONFIG.model_copy()
config_star.save_path = str(OUTPUT_DIR / "star.png")
star_path = sdf_render_csg(star_sdf, config_star)
print(f"Star rendered: {star_path}")

# %% [markdown]
# ## Create Grid of All Shapes

# %%
# Create a grid image showing all shapes
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# List of all shapes
shapes = [
    ("Sphere", sphere_path),
    ("Box", box_path),
    ("Cylinder", cylinder_path),
    ("Pill", pill_path),
    ("Torus", torus_path),
    ("Cone", cone_path),
    ("Octahedron", octahedron_path),
    ("Pyramid", pyramid_path),
    ("Hex Prism", hex_prism_path),
    ("Ellipsoid", ellipsoid_path),
    ("Rounded Box", rounded_box_path),
    ("Link", link_path),
    ("Star", star_path)
]

# Create grid layout
grid_cols = 4
grid_rows = (len(shapes) + grid_cols - 1) // grid_cols

fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(16, 12))
axes = axes.flatten()

for i, (name, path) in enumerate(shapes):
    img = mpimg.imread(path)
    axes[i].imshow(img)
    axes[i].set_title(name, fontsize=12)
    axes[i].axis('off')

# Hide unused subplots
for i in range(len(shapes), len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "all_shapes_grid.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Grid saved to: {OUTPUT_DIR}/all_shapes_grid.png")

# %% [markdown]
# ## Create Animations for Selected Shapes

# %%
# Rotating torus animation
config_torus_anim = ANIMATION_CONFIG.model_copy()
config_torus_anim.save_path = str(OUTPUT_DIR / "torus_rotation.gif")
torus_anim_path = sdf_render_csg_animation(torus_sdf, config_torus_anim)
print(f"Torus animation saved: {torus_anim_path}")

# %%
# Rotating star animation
config_star_anim = ANIMATION_CONFIG.model_copy()
config_star_anim.save_path = str(OUTPUT_DIR / "star_rotation.gif")
star_anim_path = sdf_render_csg_animation(star_sdf, config_star_anim)
print(f"Star animation saved: {star_anim_path}")

# %%
# Rotating link animation
config_link_anim = ANIMATION_CONFIG.model_copy()
config_link_anim.save_path = str(OUTPUT_DIR / "link_rotation.gif")
link_anim_path = sdf_render_csg_animation(link_sdf, config_link_anim)
print(f"Link animation saved: {link_anim_path}")

# %% [markdown]
# ## Create Extended Transition Animation
# Morphing between different shapes using 4D SDFs

# %%
def multi_shape_4d_sdf(points):
    """4D SDF that morphs between multiple shapes based on 4th dimension."""
    if points.shape[1] != 4:
        raise ValueError(f"Expected 4D points, got {points.shape[1]}D")
    
    spatial_points = points[:, :3]
    shape_param = points[:, 3]
    
    # Define shape transitions
    # 0-1: Sphere to Box
    # 1-2: Box to Cylinder
    # 2-3: Cylinder to Torus
    # 3-4: Torus to Star
    # 4-5: Star to Octahedron
    # 5-6: Octahedron back to Sphere
    
    sdf_values = np.zeros(len(points))
    
    for i in range(len(points)):
        t = shape_param[i]
        p = spatial_points[i]  # Single point
        
        if t <= 1:
            # Sphere to Box
            alpha = t
            sphere_val = sdf_sphere(p, center=np.array([0, 0, 0]), radius=0.5)
            box_val = sdf_box(p, center=np.array([0, 0, 0]), dims=np.array([0.8, 0.8, 0.8]))
            sdf_values[i] = (1 - alpha) * float(sphere_val) + alpha * float(box_val)
        elif t <= 2:
            # Box to Cylinder
            alpha = t - 1
            box_val = sdf_box(p, center=np.array([0, 0, 0]), dims=np.array([0.8, 0.8, 0.8]))
            cyl_val = sdf_cylinder(p, center=np.array([0, 0, 0]), radius=0.4, height=1.0)
            sdf_values[i] = (1 - alpha) * float(box_val) + alpha * float(cyl_val)
        elif t <= 3:
            # Cylinder to Torus
            alpha = t - 2
            cyl_val = sdf_cylinder(p, center=np.array([0, 0, 0]), radius=0.4, height=1.0)
            torus_val = sdf_torus(p, center=np.array([0, 0, 0]), r_major=0.5, r_minor=0.2)
            sdf_values[i] = (1 - alpha) * float(cyl_val) + alpha * float(torus_val)
        elif t <= 4:
            # Torus to Star
            alpha = t - 3
            torus_val = sdf_torus(p, center=np.array([0, 0, 0]), r_major=0.5, r_minor=0.2)
            star_val = sdf_star(p, center=np.array([0, 0, 0]), n_points=5, r_outer=0.6, r_inner=0.3, thickness=0.2)
            sdf_values[i] = (1 - alpha) * float(torus_val) + alpha * float(star_val)
        elif t <= 5:
            # Star to Octahedron
            alpha = t - 4
            star_val = sdf_star(p, center=np.array([0, 0, 0]), n_points=5, r_outer=0.6, r_inner=0.3, thickness=0.2)
            oct_val = sdf_octahedron(p, center=np.array([0, 0, 0]), radius=0.7)
            sdf_values[i] = (1 - alpha) * float(star_val) + alpha * float(oct_val)
        else:
            # Octahedron to Sphere
            alpha = t - 5
            oct_val = sdf_octahedron(p, center=np.array([0, 0, 0]), radius=0.7)
            sphere_val = sdf_sphere(p, center=np.array([0, 0, 0]), radius=0.5)
            sdf_values[i] = (1 - alpha) * float(oct_val) + alpha * float(sphere_val)
    
    return sdf_values.flatten()

# %%
# Create extended transition animation with error handling
import tempfile
from PIL import Image, ImageDraw, ImageFont
from util.sdf import sdf_render_level_set

def safe_render_extended_transition(sdf_func, config, shape_values):
    """Render with error handling for marching cubes failures."""
    frames = []
    original_save_path = config.save_path
    
    print(f"Generating {len(shape_values)} frames for extended transition...")
    
    for i, shape_val in enumerate(shape_values):
        # Create SDF function with current shape value
        def current_sdf(points):
            points_4d = np.concatenate([points, np.full((points.shape[0], 1), shape_val)], axis=1)
            return sdf_func(points_4d)
        
        # Create temporary frame
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            config.save_path = tmp.name
            
            try:
                frame_path = sdf_render_csg(current_sdf, config)
                img = Image.open(frame_path)
            except ValueError as e:
                if "Surface level must be within volume data range" in str(e):
                    # Create error placeholder
                    img = Image.new('RGB', config.image_size, color='white')
                    draw = ImageDraw.Draw(img)
                    text = "Transition Error"
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
                    except:
                        font = ImageFont.load_default()
                    
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    position = ((config.image_size[0] - text_width) // 2, 
                               (config.image_size[1] - text_height) // 2)
                    draw.text(position, text, fill='black', font=font)
                    print(f"Frame {i+1}/{len(shape_values)} - Transition error")
                else:
                    raise
            
            # Add shape parameter annotation
            draw = ImageDraw.Draw(img)
            text = f"t={shape_val:.2f}"
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            position = (config.image_size[0] - text_width - 10, 10)
            draw.text(position, text, fill='black', font=font)
            
            img.save(tmp.name)
            frames.append(Image.open(tmp.name))
            
            if (i+1) % 10 == 0:
                print(f"Rendered frame {i+1}/{len(shape_values)}")
    
    # Create GIF
    print("Creating animated GIF...")
    output_path = original_save_path
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000//config.fps,
        loop=0
    )
    
    print(f"Animation saved to: {output_path}")
    return output_path

# Create extended animation
extended_config = CSGRenderConfig(
    grid_size=50,
    bounds=(-1.2, 1.2),
    save_path=str(OUTPUT_DIR / "extended_shape_transition.gif"),
    image_size=(500, 500),
    camera_rotation=(30, 45, 0),
    n_frames=120,  # 12 seconds at 10 fps
    fps=10,
    colorscheme="Metallic"  # Grey/metallic theme
)

shape_values = np.linspace(0, 6, extended_config.n_frames)
extended_path = safe_render_extended_transition(multi_shape_4d_sdf, extended_config, shape_values)
print(f"Extended transition saved: {extended_path}")

# %% [markdown]
# ## Summary
# 
# All SDF shapes have been visualized with a grey/white color scheme:
# - Basic shapes: sphere, box, cylinder, pill, torus
# - Advanced shapes: cone, octahedron, pyramid, hexagonal prism, ellipsoid  
# - Special shapes: rounded box, chain link, star
# - Grid layout of all shapes
# - Rotating animations for selected shapes
# - Extended transition animation morphing between shapes
# 
# Check the `reports/sdf_all_visualize/` directory for all generated images and animations.