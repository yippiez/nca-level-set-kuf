import numpy as np
from typing import Union

def sdf_sphere(point: np.ndarray, center: np.ndarray, radius: float) -> Union[float, np.ndarray]:
    """Calculate the signed distance from a point to a sphere."""
    point = np.asarray(point)
    center = np.asarray(center)
    dist = np.linalg.norm(point - center, axis=-1)
    return dist - radius

def sdf_pill(point: np.ndarray, p1: np.ndarray, p2: np.ndarray, radius: float) -> Union[float, np.ndarray]:
    """Calculate the signed distance from a point to a pill shape."""
    point = np.asarray(point)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    
    axis = p2 - p1
    axis_len = np.linalg.norm(axis)
    axis_norm = axis / axis_len if axis_len > 0 else np.array([0, 0, 1])
    
    if point.ndim > 1:
        v = point - p1
        proj = np.sum(v * axis_norm, axis=-1)
        proj = np.clip(proj, 0, axis_len)
        closest = p1 + np.outer(proj, axis_norm)
        dist = np.linalg.norm(point - closest, axis=-1)
        return dist - radius
    else:
        v = point - p1
        proj = np.dot(v, axis_norm)
        proj = np.clip(proj, 0, axis_len)
        closest = p1 + proj * axis_norm
        dist = np.linalg.norm(point - closest)
        return dist - radius # type: ignore

def sdf_box(point: np.ndarray, center: np.ndarray, dims: np.ndarray) -> Union[float, np.ndarray]:
    """Calculate the signed distance from a point to an axis-aligned box."""
    point = np.asarray(point)
    center = np.asarray(center)
    dims = np.asarray(dims)
    
    half = dims / 2
    local = np.abs(point - center)
    d = local - half
    
    inside = np.min(half - local, axis=-1)
    inside = np.where(np.all(d < 0, axis=-1), -inside, 0)
    outside = np.sqrt(np.sum(np.maximum(d, 0)**2, axis=-1))
    
    return outside + inside

def sdf_torus(point: np.ndarray, center: np.ndarray, r_major: float, r_minor: float) -> Union[float, np.ndarray]:
    """Calculate the signed distance from a point to a torus aligned with the xz-plane."""
    point = np.asarray(point)
    center = np.asarray(center)
    
    local = point - center
    xz_dist = np.sqrt(local[..., 0]**2 + local[..., 2]**2) - r_major
    dist = np.sqrt(xz_dist**2 + local[..., 1]**2) - r_minor
    
    return dist

def sdf_cylinder(point: np.ndarray, center: np.ndarray, radius: float, height: float) -> Union[float, np.ndarray]:
    """Calculate the signed distance from a point to a cylinder aligned with the y-axis."""
    point = np.asarray(point)
    center = np.asarray(center)
    
    local = point - center
    radial_dist = np.sqrt(local[..., 0]**2 + local[..., 2]**2) - radius
    axial_dist = np.abs(local[..., 1]) - height/2
    
    # Interior distance
    interior_dist = np.maximum(radial_dist, axial_dist)
    
    # Exterior distance
    exterior_dist = np.sqrt(np.maximum(radial_dist, 0)**2 + np.maximum(axial_dist, 0)**2)
    
    return np.where((radial_dist < 0) & (axial_dist < 0), interior_dist, exterior_dist)

def sdf_cone(point: np.ndarray, tip: np.ndarray, base_center: np.ndarray, radius: float) -> Union[float, np.ndarray]:
    """Calculate the signed distance from a point to a cone."""
    point = np.asarray(point)
    tip = np.asarray(tip)
    base_center = np.asarray(base_center)
    
    axis = base_center - tip
    height = np.linalg.norm(axis)
    axis_norm = axis / height if height > 0 else np.array([0, 1, 0])
    
    # Project point onto axis
    v = point - tip
    proj_len = np.dot(v, axis_norm)
    
    # Handle cases
    if point.ndim == 1:
        if proj_len < 0:
            # Below tip
            return np.linalg.norm(v) # type: ignore
        elif proj_len > height:
            # Above base
            return np.linalg.norm(point - base_center) # type: ignore
        else:
            # Along cone surface
            proj_point = tip + proj_len * axis_norm
            lateral_vec = point - proj_point
            lateral_dist = np.linalg.norm(lateral_vec)
            cone_radius = radius * proj_len / height
            return lateral_dist - cone_radius
    else:
        dist = np.zeros(point.shape[0])
        for i in range(point.shape[0]):
            dist[i] = sdf_cone(point[i], tip, base_center, radius)
        return dist

def sdf_octahedron(point: np.ndarray, center: np.ndarray, radius: float) -> Union[float, np.ndarray]:
    """Calculate the signed distance from a point to an octahedron."""
    point = np.asarray(point)
    center = np.asarray(center)
    
    local = np.abs(point - center)
    return (np.sum(local, axis=-1) - radius) * 0.57735027  # 1/sqrt(3)

def sdf_pyramid(point: np.ndarray, tip: np.ndarray, base_center: np.ndarray, base_size: float) -> Union[float, np.ndarray]:
    """Calculate the signed distance from a point to a square pyramid."""
    point = np.asarray(point)
    tip = np.asarray(tip)
    base_center = np.asarray(base_center)
    
    # Simple approximation using cone and box combination
    height = np.linalg.norm(tip - base_center)
    cone_dist = sdf_cone(point, tip, base_center, base_size * 0.7071)  # sqrt(2)/2
    box_dist = sdf_box(point, base_center - np.array([0, height/2, 0]), np.array([base_size, height, base_size]))
    
    return np.maximum(cone_dist, -box_dist)

def sdf_hexagonal_prism(point: np.ndarray, center: np.ndarray, radius: float, height: float) -> Union[float, np.ndarray]:
    """Calculate the signed distance from a point to a hexagonal prism."""
    point = np.asarray(point)
    center = np.asarray(center)
    
    local = point - center
    
    # Hexagon in XZ plane
    abs_x = np.abs(local[..., 0])
    abs_z = np.abs(local[..., 2])
    
    # Use max of three planes to approximate hexagon
    hex_dist = np.maximum.reduce([
        abs_x - radius,
        0.5 * abs_x + 0.866 * abs_z - radius,
        0.866 * abs_x + 0.5 * abs_z - radius
    ])
    
    # Height constraint
    height_dist = np.abs(local[..., 1]) - height/2
    
    # Combine
    interior_dist = np.maximum(hex_dist, height_dist)
    exterior_dist = np.sqrt(np.maximum(hex_dist, 0)**2 + np.maximum(height_dist, 0)**2)
    
    return np.where((hex_dist < 0) & (height_dist < 0), interior_dist, exterior_dist)

def sdf_ellipsoid(point: np.ndarray, center: np.ndarray, radii: np.ndarray) -> Union[float, np.ndarray]:
    """Calculate the signed distance from a point to an ellipsoid."""
    point = np.asarray(point)
    center = np.asarray(center)
    radii = np.asarray(radii)
    
    local = point - center
    # Normalize to unit sphere
    normalized = local / radii
    dist_normalized = np.linalg.norm(normalized, axis=-1)
    
    # Approximate gradient magnitude for correct distance
    gradient = normalized / radii
    gradient_mag = np.linalg.norm(gradient, axis=-1)
    
    return (dist_normalized - 1.0) / gradient_mag

def sdf_rounded_box(point: np.ndarray, center: np.ndarray, dims: np.ndarray, radius: float) -> Union[float, np.ndarray]:
    """Calculate the signed distance from a point to a rounded box."""
    point = np.asarray(point)
    center = np.asarray(center)
    dims = np.asarray(dims)
    
    # Reduce box dimensions by rounding radius
    inner_dims = dims - 2 * radius
    inner_dims = np.maximum(inner_dims, 0)  # Ensure non-negative
    
    # Distance to inner box
    inner_dist = sdf_box(point, center, inner_dims)
    
    # Apply rounding
    return inner_dist - radius

def sdf_link(point: np.ndarray, center: np.ndarray, r_major: float, r_minor: float, length: float) -> Union[float, np.ndarray]:
    """Calculate the signed distance from a point to a chain link shape."""
    point = np.asarray(point)
    center = np.asarray(center)
    
    # Two tori connected by cylinders
    torus1_center = center + np.array([length/2, 0, 0])
    torus2_center = center - np.array([length/2, 0, 0])
    
    dist1 = sdf_torus(point, torus1_center, r_major, r_minor)
    dist2 = sdf_torus(point, torus2_center, r_major, r_minor)
    
    # Connecting cylinders
    cyl1_dist = sdf_cylinder(point, center + np.array([0, r_major, 0]), r_minor, length)
    cyl2_dist = sdf_cylinder(point, center - np.array([0, r_major, 0]), r_minor, length)
    
    # Union of all parts
    return np.minimum.reduce([dist1, dist2, cyl1_dist, cyl2_dist])

def sdf_star(point: np.ndarray, center: np.ndarray, n_points: int, r_outer: float, r_inner: float, thickness: float) -> Union[float, np.ndarray]:
    """Calculate the signed distance from a point to a star shape in XZ plane."""
    point = np.asarray(point)
    center = np.asarray(center)
    
    local = point - center
    
    # Convert to polar coordinates in XZ plane
    angle = np.arctan2(local[..., 2], local[..., 0])
    radius_xz = np.sqrt(local[..., 0]**2 + local[..., 2]**2)
    
    # Star shape radius at given angle
    segment_angle = 2 * np.pi / n_points
    angle_in_segment = np.mod(angle, segment_angle)
    
    # Linear interpolation between inner and outer radius
    t = np.abs(angle_in_segment - segment_angle/2) / (segment_angle/2)
    star_radius = r_inner + (r_outer - r_inner) * (1 - t)
    
    # Distance in XZ plane
    radial_dist = radius_xz - star_radius
    
    # Thickness in Y direction
    y_dist = np.abs(local[..., 1]) - thickness/2
    
    # Combine distances
    interior_dist = np.maximum(radial_dist, y_dist)
    exterior_dist = np.sqrt(np.maximum(radial_dist, 0)**2 + np.maximum(y_dist, 0)**2)
    
    return np.where((radial_dist < 0) & (y_dist < 0), interior_dist, exterior_dist)
