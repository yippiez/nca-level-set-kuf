import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import functools
from skimage.measure import marching_cubes

def sdf_sphere(point, center, radius):
    """
    Calculate the signed distance from a point to a sphere.
    
    Args:
        point: Array-like of shape (..., 3) containing (x, y, z) coordinates
        center: Array-like of shape (3,) for the center of the sphere
        radius: Positive float, the radius of the sphere
        
    Returns:
        Float or array of floats, the signed distance from the point(s) to the sphere
    """
    point = np.asarray(point)
    center = np.asarray(center)
    
    # Calculate the distance from the point to the center of the sphere
    dist_to_center = np.linalg.norm(point - center, axis=-1)
    
    # The signed distance is the distance to the center minus the radius
    # Positive outside, zero on the surface, negative inside
    return dist_to_center - radius

def sdf_pill(point, p1, p2, radius):
    """
    Calculate the signed distance from a point to a pill shape.
    
    Args:
        point: Array-like of shape (..., 3) containing (x, y, z) coordinates
        p1: Array-like of shape (3,) for one end of the pill's axis
        p2: Array-like of shape (3,) for the other end of the pill's axis
        radius: Positive float, the radius of the pill
        
    Returns:
        Float or array of floats, the signed distance from the point(s) to the pill
    """
    point = np.asarray(point)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    
    # Calculate the axis of the pill
    axis = p2 - p1
    axis_length = np.linalg.norm(axis)
    axis_normalized = axis / axis_length if axis_length > 0 else np.array([0, 0, 1])
    
    # For vectorized computation, we need to handle the shape correctly
    if point.ndim > 1:
        # Vectorized computation for multiple points
        # Calculate vector from p1 to each point
        p1_to_point = point - p1
        
        # Project this vector onto the normalized axis
        projection = np.sum(p1_to_point * axis_normalized, axis=-1)
        
        # Clip the projection to the segment length
        projection_clipped = np.clip(projection, 0, axis_length)
        
        # Calculate the closest point on the axis
        closest_point = p1 + np.outer(projection_clipped, axis_normalized)
        
        # Calculate the perpendicular distance from point to axis
        dist_to_axis = np.linalg.norm(point - closest_point, axis=-1)
        
        # The signed distance is the perpendicular distance minus the radius
        return dist_to_axis - radius
    else:
        # Single point calculation
        p1_to_point = point - p1
        projection = np.dot(p1_to_point, axis_normalized)
        projection_clipped = np.clip(projection, 0, axis_length)
        closest_point = p1 + projection_clipped * axis_normalized
        dist_to_axis = np.linalg.norm(point - closest_point)
        return dist_to_axis - radius

def sdf_box(point, center, dimensions):
    """
    Calculate the signed distance from a point to an axis-aligned box.
    
    Args:
        point: Array-like of shape (..., 3) containing (x, y, z) coordinates
        center: Array-like of shape (3,) for the center of the box
        dimensions: Array-like of shape (3,) for the dimensions of the box (width, height, depth)
        
    Returns:
        Float or array of floats, the signed distance from the point(s) to the box
    """
    point = np.asarray(point)
    center = np.asarray(center)
    dimensions = np.asarray(dimensions)
    
    # Calculate the half-dimensions
    half_dims = dimensions / 2
    
    # Calculate the local coordinates relative to the center
    local_point = np.abs(point - center)
    
    # Calculate the distance to the closest face for each axis
    distance_to_face = local_point - half_dims
    
    # For points inside the box, calculate the negative distance to the closest face
    inside_distance = np.min(half_dims - local_point, axis=-1)
    inside_distance = np.where(np.all(distance_to_face < 0, axis=-1), -inside_distance, 0)
    
    # For points outside the box, calculate the positive distance to the closest face
    outside_distance = np.sqrt(np.sum(np.maximum(distance_to_face, 0)**2, axis=-1))
    
    # Combine inside and outside distances
    return outside_distance + inside_distance

def sdf_torus(point, center, major_radius, minor_radius):
    """
    Calculate the signed distance from a point to a torus aligned with the xz-plane.
    
    Args:
        point: Array-like of shape (..., 3) containing (x, y, z) coordinates
        center: Array-like of shape (3,) for the center of the torus
        major_radius: Positive float, the radius from the center of the torus to the center of the tube
        minor_radius: Positive float, the radius of the tube
        
    Returns:
        Float or array of floats, the signed distance from the point(s) to the torus
    """
    point = np.asarray(point)
    center = np.asarray(center)
    
    # Translate the point to the local coordinate system
    local_point = point - center
    
    # Calculate the distance in the xz-plane from the center
    xz_dist = np.sqrt(local_point[..., 0]**2 + local_point[..., 2]**2) - major_radius
    
    # Calculate the combined distance
    dist = np.sqrt(xz_dist**2 + local_point[..., 1]**2) - minor_radius
    
    return dist

def sdf_render(sdf_func, grid_size=50, bounds=(-1, 1), threshold=0.0, n_frames=36, save_path=None, figsize=(10, 10), dpi=100, fps=15):
    """
    Render a 3D signed distance function as a rotating animation.
    
    Args:
        sdf_func: A function that takes point(s) of shape (..., 3) and returns signed distance(s)
        grid_size: Number of points along each dimension in the grid
        bounds: Tuple (min, max) for the bounds of the grid in all dimensions
        threshold: The isosurface threshold value (usually 0 for the surface)
        n_frames: Number of frames in the rotation animation
        save_path: Path to save the resulting animation (if None, just displays it)
        figsize: Figure size for the plot
        dpi: DPI for the rendered animation
        fps: Frames per second for the animation
        
    Returns:
        The created animation object
    """
    # Create a 3D grid of points
    x = np.linspace(bounds[0], bounds[1], grid_size)
    y = np.linspace(bounds[0], bounds[1], grid_size)
    z = np.linspace(bounds[0], bounds[1], grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Reshape the grid for vectorized evaluation
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
    
    # Evaluate the SDF at all grid points
    distances = sdf_func(points).reshape(X.shape)
    
    # Create the figure for the visualization
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate the spacing between grid points
    spacing = (
        (bounds[1] - bounds[0]) / (grid_size - 1),
        (bounds[1] - bounds[0]) / (grid_size - 1),
        (bounds[1] - bounds[0]) / (grid_size - 1)
    )
    
    # Use scikit-image's marching cubes to create the isosurface
    verts, faces, normals, _ = marching_cubes(
        distances, 
        level=threshold,
        spacing=spacing
    )
    
    # Shift from grid coordinates to world coordinates
    verts = verts + np.array([bounds[0], bounds[0], bounds[0]])
    
    # Create the initial mesh plot
    mesh = ax.plot_trisurf(verts[:, 0], verts[:, 1], triangles=faces, Z=verts[:, 2], 
                           cmap='viridis', edgecolor='none', alpha=0.8)
    
    # Set nice equal aspect ratio and view angle
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_zlim(bounds)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Function to update the plot for each animation frame
    def update(frame, mesh, verts, faces, ax):
        # Clear the axes instead of removing the specific mesh
        ax.clear()
        
        # Set view angle for this frame
        ax.view_init(elev=30, azim=frame * (360 / n_frames))
        
        # Recreate the mesh with the new view angle
        mesh = ax.plot_trisurf(verts[:, 0], verts[:, 1], triangles=faces, Z=verts[:, 2], 
                               cmap='viridis', edgecolor='none', alpha=0.8)
        
        # Reset the axis properties
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_zlim(bounds)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        return [mesh]
    
    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, fargs=(mesh, verts, faces, ax),
        interval=1000/fps, blit=False
    )
    
    # Save the animation if a path is provided
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps, dpi=dpi)
    
    plt.close(fig)
    
    return anim