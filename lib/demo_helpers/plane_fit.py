#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import numpy as np


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

# .....................................................................................................................

def get_xyz_samples(depth_data, num_x_samples = 16, num_y_samples = 16, jitter_scale = 0.75):
    
    '''
    Helper used to grab samples from a depth image used for finding a plane of best fit.
    Note that the 'plane of best fit' is meant to represent the 'floor' of the scene, assuming
    the floor is always the furthest thing away from the camera. This could fail if large parts of
    the scene have walls/no floor visible!
    
    Inputs:
        depth_data: should be a 1 channel image of depth values (e.g. output from MiDaS model)
        num_x_samples, num_y_samples: the number of sampling points to use on the depth map
        jitter_scale: amount of randomness to apply to sampling points
    
    Outputs:
        xyz_samples, xyz_mean
    
    Note: The total number of samples will be the product of: num_x_samples * num_y_samples
          (Unless there aren't enough rows/columns in the provided depth data!)
    '''
    
    # For clarity
    x_step = 1.0 / num_x_samples
    y_step = 1.0 / num_y_samples
    data_h, data_w = depth_data.shape[0:2]
    
    # Force 'safe' input values
    num_x_samples = min(num_x_samples, data_w)
    num_y_samples = min(num_y_samples, data_h)
    jitter_scale = np.clip(jitter_scale, 0.0, 1.0)
    
    # Make sampling grid points Nx-by-Ny, with gap around edges
    x_samples_vector = x_step * (0.5 + np.arange(num_x_samples, dtype=np.float32))
    y_samples_vector = y_step * (0.5 + np.arange(num_y_samples, dtype=np.float32))
    xgrid, ygrid = np.meshgrid(x_samples_vector, y_samples_vector)
    
    # Add noise to slightly jitter sample points so they aren't all on a perfect grid
    xgrid += np.clip(np.random.randn(*xgrid.shape), -1, 1) * (x_step / 2.0) * jitter_scale
    ygrid += np.clip(np.random.randn(*ygrid.shape), -1, 1) * (y_step / 2.0) * jitter_scale

    # Convert samples from normalized (0-to-1) units to pixels for sampling depth 'image'
    norm_to_px_scale = np.float32((data_w - 1, data_h - 1))
    xy_samples_norm = np.dstack((xgrid,ygrid)).reshape(-1,2)
    xy_samples_px = np.int32(np.round(xy_samples_norm * norm_to_px_scale))
    
    # Sample depth data for z at corresponding xy grid values
    z_samples = depth_data[xy_samples_px[:,1], xy_samples_px[:,0]]
    
    # Record the exact mean of x,y (which we know in advance) and sample mean of the z data
    xyz_samples = np.hstack((xy_samples_px, np.expand_dims(z_samples, 1)))
    x_mean = (data_w - 1) * 0.5
    y_mean = (data_h - 1) * 0.5
    z_mean = np.mean(xyz_samples[:,2])
    xyz_mean = np.array([x_mean, y_mean, z_mean])
    
    return xyz_samples, xyz_mean

# .....................................................................................................................

def find_plane_normal(xyz_samples, xyz_mean = None):
    
    '''
    Finds the normal vector of the 'plane-of-best-fit' to the given xyz sample data
    (Uses singular-value-decomposition, so should be numerically stable...?)
    
    Input:
        xyz_samples: should be a N x 3 numpy array, where N is number of samples (at least 3!)
                     The 3 columns should be [x-coord, y-coord, depth]
    
        xyz_mean: Optional, represents average xyz value of data. If not given, the mean will
                  be calculated from the sample values (this may give an overly biased result!).
                  Should be given as a numpy array of length 3, or None.
        
    Returns:
        plane_normal_vector
    '''
    
    # Make sure we get 2 dimensions -> columns of x,y,z with each row representing a single xyz sample
    shape = xyz_samples.shape
    bad_shape = (len(shape) != 2)
    if bad_shape:
        raise TypeError("Expecting samples of shape: [num samples, 3] Got: {}".format(shape))
    not_xyz = (shape[1] != 3)
    if not_xyz:
        raise TypeError("Expecting columns of 'x,y,z', so shape should be [num samples, 3]. Got: {}".format(shape))
    
    # Find the mean if needed
    no_mean_data = xyz_mean is None
    if no_mean_data:
        xyz_mean = np.mean(xyz_samples, axis = 0)
        
    # Find singular-value-decomposition on sample data
    svd = np.linalg.svd(xyz_samples - xyz_mean)
    left_vectors, singular_values, right_vectors = svd
    
    # Assuming data has shape: N x 3
    # -> SVD gives matrices:
    #       U : N x N  |  Sigma : N x 3  |  V.T : 3 X 3
    #       (numpy returns only the diagonal of sigma, an array of 3 values)
    # The normal of the plane-of-best-fit corresponds to the column vector of V (not transposed)
    # associated with the smallest singular value (from sigma), because of mathemagical reasons
    # -> Since we get V transposed, we instead want the *row* vector corresponding to the smallest singular value
    idx_of_lowest_singular_value = np.argmin(singular_values)
    plane_normal_vec = right_vectors[idx_of_lowest_singular_value, :]
    
    return plane_normal_vec

# .....................................................................................................................

def generate_image_from_plane_normal(output_shape_hw, plane_normal, xyz_mean = None):
    
    '''
    Function which generates a 'plane image' from a plane normal & xyz mean
    
    Equation of a plane is given by:
        ax + by + cz + d = 0
        
    Where [a, b, c] represents the normal of the plane
    so: d = -(ax' + by' + cz'), where (x', y', z') is some point that is known to be on the plane
    -> We can use the plane mean for (x', y', z') if provided, otherwise assume the center of the plane has z = 0
    
    We can generate a plane 'image' by calculating the plane z at all xy's using:
        z = -(d + ax + by) / c
    
    See: http://www.songho.ca/math/plane/plane.html
    
    
    Inputs:
        output_shape_hw: a tuple containing desired (height, width) of result
        plane_normal: array of [nx, ny, nz] representing plane-of-best-fit
        xyz_mean: array of [mx, my, mz] representing mean offset used for finding plane-of-best-fit
    
    Returns:
        plane_as_2D_image
    '''
    
    # For convenience
    num_rows, num_cols = output_shape_hw.shape[0:2]
    nx, ny, nz = plane_normal
    mx, my, mz = xyz_mean if xyz_mean is not None else ((num_cols-1)/2, (num_rows-1)/2, 0)
    
    # Find the mean offset/zeroing term
    d = -1*(nx*mx + ny*my + nz*mz)
    
    # Build x/y indexing sampling matrices (with mean removal)
    xidx = np.arange(0, num_cols)
    yidx = np.arange(0, num_rows)
    xmesh, ymesh = np.meshgrid(xidx, yidx)
    
    # Calculate plane z and subtract it from the input depth data
    plane_image = -(d + nx*xmesh + ny*ymesh) / nz
    
    return plane_image

# .....................................................................................................................

def estimate_plane_of_best_fit(depth_data, samples_per_side = 16):
    
    '''
    Helper which simply performs all plane removal steps
    Returns:
        plane_of_best_fit_image
    '''
    
    xyz_samples, xyz_mean = get_xyz_samples(depth_data, samples_per_side, samples_per_side)
    plane_normal_vector = find_plane_normal(xyz_samples, xyz_mean)
    plane_image = generate_image_from_plane_normal(depth_data, plane_normal_vector, xyz_mean)
    
    return plane_image
