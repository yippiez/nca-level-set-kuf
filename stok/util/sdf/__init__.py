"""SDF module with shape definitions and rendering functions."""

# Import all SDF shape functions
from .definitions import (
    sdf_sphere,
    sdf_pill,
    sdf_box,
    sdf_torus,
    sdf_cylinder,
    sdf_cone,
    sdf_octahedron,
    sdf_pyramid,
    sdf_hexagonal_prism,
    sdf_ellipsoid,
    sdf_rounded_box,
    sdf_link,
    sdf_star
)

# Import CSG rendering functions
from .render import (
    sdf_render_csg,
    sdf_render_csg_animation,
    sdf_render_level_set,
    sdf_render_level_set_grid
)

# Import similarity functions
from .similarity import sdf_get_sampled_boolean_similarity

__all__ = [
    # Shape functions
    'sdf_sphere',
    'sdf_pill',
    'sdf_box',
    'sdf_torus',
    'sdf_cylinder',
    'sdf_cone',
    'sdf_octahedron',
    'sdf_pyramid',
    'sdf_hexagonal_prism',
    'sdf_ellipsoid',
    'sdf_rounded_box',
    'sdf_link',
    'sdf_star',
    # Rendering functions
    'sdf_render_csg',
    'sdf_render_csg_animation',
    'sdf_render_level_set',
    'sdf_render_level_set_grid',
    # Similarity functions
    'sdf_get_sampled_boolean_similarity'
]