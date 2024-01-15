#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 12:04:32 2023

@author: declanbracken
"""

import numpy as np
import matplotlib.pyplot as plt
# from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres, lgmres, cg, bicgstab, cgs, spilu
from scipy.sparse import diags
import time
import pandas as pd
import os
from tqdm import tqdm

# Set plot resolution
plt.rcParams['figure.dpi'] = 600
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 16,
    "axes.labelsize": 16,  # set default size for axis labels
    "axes.titlesize": 16,  # set default size for titles
    "xtick.labelsize": 16,  # set default size for x-tick labels
    "ytick.labelsize": 16,  # set default size for y-tick labels
    "legend.fontsize": 16,  # set default size for legends
    "axes.linewidth": 0.25,  # set default linewidth for axes
    "xtick.major.size": 5,  # set default size for major x-ticks
    "xtick.major.width": 0.5,  # set default linewidth for major x-ticks
    "ytick.major.size": 5,  # set default size for major y-ticks
    "ytick.major.width": 0.5,  # set default linewidth for major y-ticks
    "xtick.minor.size": 3,  # set default size for minor x-ticks (if using)
    "xtick.minor.width": 1,  # set default linewidth for minor x-ticks (if using)
    "ytick.minor.size": 3,  # set default size for minor y-ticks (if using)
    "ytick.minor.width": 1  # set default linewidth for minor y-ticks (if using)
})
#%% Ploting functions

def contour_heatplot(x, y, phi, title='Velocity Contour Plot', label='velocity', 
                     num_levels=40, cmap='coolwarm', fig_size=(10, 6), save_path=None):

    # Plot with specified figure size
    fig, ax = plt.subplots(figsize=fig_size)
    fs = 16  # You can also set this as a parameter if needed
    
    # Set aspect of the plot to be equal
    ax.set_aspect('equal', adjustable='box')
    
    # Using contourf for filled contours
    contour = ax.contourf(x, y, phi, levels=num_levels, cmap=cmap)
    cbar = plt.colorbar(contour)
    cbar.set_label(label, fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)
    plt.tick_params(axis='both', labelsize=fs)
    ax.set_xlabel('X Coordinate', fontsize=fs)
    ax.set_ylabel('Y Coordinate', fontsize=fs)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.show()

def streamlines(x, y, u, v, fs=16, cmap='viridis', save_path=None, den = 3.5):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    # Set aspect of the plot to be equal
    ax.set_aspect('equal', adjustable='box')

    # Calculate the speed (magnitude of velocity)
    speed = np.sqrt(u**2 + v**2)

    # Apply a modified logarithmic scale to the speed for color and linewidth
    # This helps in handling a wide range of values
    epsilon = 1e-6  # small constant to avoid log(0)
    scaled_speed = np.log(speed + epsilon)

    # Normalize the scaled speed for color and linewidth
    norm_speed = (scaled_speed - scaled_speed.min()) / (scaled_speed.max() - scaled_speed.min())

    # Create the streamline plot
    strm = plt.streamplot(x, y, u, v, color=norm_speed, linewidth=1.5,
                          cmap=cmap, density=den, arrowstyle='->', arrowsize=1.5)

    plt.colorbar(strm.lines, label= r'Log-Scaled Speed')
    plt.xlabel(r'X Coordinate', fontsize=fs)
    plt.ylabel(r'Y Coordinate', fontsize=fs)
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.show()
    
def plot_horizontal_velocity_profile(x, y, u, fs=16, compare_Ghia = False, save_path = None):
    
    # Find the index of the horizontal center line
    # center_index = np.where(y == np.median(y))[0][0]
    center_index = len(y[:,0])//2

    # Extract the u-velocity along the horizontal center line
    u_center = u[center_index, :]
    x_center = x[center_index, :]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_center, u_center, label='V-Velocity Profile')
    
    if compare_Ghia:
        # v values at horitzontal cross section retrieved from Ghia et al.
        Ghia_u = [0.00000, -0.05906,-0.07391, -0.08864, -0.10313, -0.16914, 
                  -0.22445, -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 
                  0.12317, 0.10890, 0.10091, 0.09233, 0.00000]
        
        # Corresponding x coordinates for Ghia et al. v velocities
        Ghia_x = [1.0, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                  0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 
                  0.0000]
        
        plt.plot(Ghia_x, Ghia_u, label='Ghia et al.',marker = 'o',linewidth = 0)
    
    plt.xlabel('X Coordinate', fontsize=fs)
    plt.ylabel('V-Velocity', fontsize=fs)
    plt.legend(fontsize=fs)
    plt.grid(False)
    plt.tick_params(axis='both', labelsize=fs)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()
    
def plot_vertical_velocity_profile(x, y, u, fs=16, compare_Ghia = False, save_path = None):
    
    # Find the index of the horizontal center line
    center_index = len(x[0,:])//2

    # Extract the u-velocity along the horizontal center line
    u_center = u[:, center_index]
    y_center = y[:, center_index]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_center, u_center, label='U-Velocity Profile')
    
    if compare_Ghia:
        # v values at horitzontal cross section retrieved from Ghia et al.
        Ghia_y = [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 
                  0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547,
                  0.0000 ]
        
        # Corresponding x coordinates for Ghia et al. v velocities
        Ghia_u = [1.0000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332,
                  -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434,
                  -0.04775, -0.04192, -0.03717, 0.00000]
        
        plt.plot(Ghia_y, Ghia_u, label='Ghia et al.',marker = 'o',linewidth = 0)
    
    plt.xlabel('Y Coordinate', fontsize=fs)
    plt.ylabel('U-Velocity', fontsize=fs)
    plt.legend(fontsize=fs)
    plt.grid(False)
    plt.tick_params(axis='both', labelsize=fs)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()

def plot_residuals(iterations, residuals, fs=16):
    """Plot the residuals from the solver."""
    plt.figure(figsize=(8,6))
    for res in residuals:
        plt.semilogy(iterations, res, linestyle='-')
    plt.title('Solver Convergence', fontsize=fs)
    plt.xlabel('Iteration Number', fontsize=fs)
    plt.ylabel('Residual (L2-Norm)', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(True, which="both", ls="--")
    plt.legend(labels =["Pressure", "U-Velocity", "V-Velocity"])
    plt.tight_layout()
    plt.show()


def Linear_Solve(A, b, x0, solver='bicgstab', use_preconditioner=True, 
                 precon_tol=1e-4, solver_tol=1e-5, rest=50, 
                 calculate_residuals=False):
    """
    ARGS:
        A:  Dependency matrix
        b:  Source terms array
        x0: Initial Guess
        solver: choose what solver you want to use
        use_preconditioner: boolean to choose preconditioning with ilu factorization (leave on)
        calculate_residuals: boolean to choose whether to calculate residuals
    """
    # Optional: List to store the residuals
    residuals = None
    callback_func = None

    # Define the callback function only if residuals need to be calculated
    if calculate_residuals:
        residuals = []

        def callback(xk):
            r = np.linalg.norm(b - A.dot(xk))
            residuals.append(r)
        
        callback_func = callback

    # List of solver methods
    solver_list = ['gmres', 'lgmres', 'cg', 'bicgstab', 'cgs']
    
    # Raise error if an unspecified solver is used
    if solver not in solver_list:
        raise ValueError(f"Invalid solver. Supported options are {solver_list}.")

    # Construct preconditioner separately from solving
    M_x = None
    if use_preconditioner:
        # Apply incomplete LU decomposition to approximate the inverse of A, with optional tolerance
        preconditioned_A = spilu(A, drop_tol=precon_tol)
        M_x = LinearOperator(A.shape, preconditioned_A.solve)
    
    # Select the solver method
    solvers = {
        'lgmres': lgmres,
        'cg': cg,
        'bicgstab': bicgstab,
        'cgs': cgs,
        'gmres': gmres
    }
    solver_func = solvers[solver]

    # Execute the solver
    x, info = solver_func(A, b, x0=x0, M=M_x, atol=solver_tol, callback=callback_func)

    # Check for successful convergence
    if info != 0:
        print(f"Warning: {solver} did not converge (info={info}).")

    return x, residuals


def calculate_link_coeffs(uface, vface, u_star, v_star, P, Ap, Ae, Aw, An, As, Sx, Sy, alpha_uv, lid_vel, Re, ra, ar, dx, dy, nx, ny):
    """
    Calculate the link coefficients for the link matrix:
    1.  Calculate interior coefficients by looping through rows & columns, 
        getting face velocities, multiplying by density, and then calculating
        coefficients according to the equations (aswell as source terms).
    2.  Calculate wall coefficients through the same process, each seperately
    3.  Calcilate corner coefficients the same way.
    """
    
    # Get diffusive terms:
    De = ra / Re
    Dw = ra / Re
    Dn = ar / Re
    Ds = ar / Re
    
    # interior cells
    i = slice(2, ny)
    j = slice(2, nx)
    i_min = slice(1, ny-1) #interior + boundary
    j_min = slice(1, nx-1) #interior + boundary
    i_plus = slice(3, ny+1) #interior + boundary
    j_plus = slice(3, nx+1) #interior + boundary
    
    # Get face velocities
    Fe = dy * uface[i, j]
    Fw = dy * uface[i, j_min]
    Fn = dx * vface[i, j]
    Fs = dx * vface[i_min, j]
    
    # Create coefficients:
    Ae[i, j] = De + np.maximum(0, -Fe)
    Aw[i, j] = Dw + np.maximum(0, Fw)
    An[i, j] = Dn + np.maximum(0, -Fn)
    As[i, j] = Ds + np.maximum(0, Fs)
   
    # Central Coefficient is the sum of the others (but positive)
    Ap[i, j] =  (De + Dw + Dn + Ds + np.maximum(0, Fe) + np.maximum(0, -Fw) + 
                      np.maximum(0, Fn) + np.maximum(0, -Fs))
    Pe = P[i, j_plus]
    Pw = P[i, j_min]
    Pn = P[i_plus, j]
    Ps = P[i_min, j]
    
    Sx[1:-1, 1:-1] = (Pw - Pe) * dy / 2 + (1-alpha_uv) * Ap[i, j] * u_star[i, j]
    Sy[1:-1, 1:-1] = (Ps - Pn) * dx / 2 + (1-alpha_uv) * Ap[i, j] * v_star[i, j]
            
    # Boundary cells
#-------------------------------------------------------------------------
    # Eastern Wall
    j = nx
    j_min = nx-1
    j_plus = nx + 1
    i = slice(2, ny)
    i_min = slice(1, ny-1)
    i_plus = slice(3, ny+1)
    
    #calculate advective coeffs
    Fe = dy * uface[i, j] # right face initialized to 0.
    Fw = dy * uface[i, j_min]
    Fn = dx * vface[i, j]
    Fs = dx * vface[i_min, j]
    
    #calculate coeffs
    Ae[i,j] = 0
    Aw[i,j] = Dw + np.maximum(0, Fw)
    An[i,j] = Dn + np.maximum(0, -Fn)
    As[i,j] = Ds + np.maximum(0, Fs)
    # calculate central coeff
    Ap[i, j] = (2*De + Dw + Dn + Ds + np.maximum(0, Fe) + np.maximum(0, -Fw) + 
                      np.maximum(0, Fn) + np.maximum(0, -Fs))
    # get pressure slices
    Pe = P[i, j]
    Pw = P[i, j_min]
    Pn = P[i_plus, j]
    Ps = P[i_min, j]
    # calculate sources
    Sx[1:-1,-1] = (Pw - Pe) * dy / 2 + (1-alpha_uv) * Ap[i,j] * u_star[i,j]
    Sy[1:-1,-1] = (Ps - Pn) * dx / 2 + (1-alpha_uv) * Ap[i,j] * v_star[i,j]
    
#-------------------------------------------------------------------------
    # Western Wall
    j = 1
    j_min = 0
    j_plus = 2
    i = slice(2,ny)
    i_min = slice(1, ny-1)
    i_plus = slice(3, ny+1)
    
    #calculate advective coeffs
    Fe = dy * uface[i, j]
    Fw = dy * uface[i, j_min] # left face initialized to 0.
    Fn = dx * vface[i, j]
    Fs = dx * vface[i_min, j]
    
    #calculate coeffs
    Ae[i,j] = De + np.maximum(0, -Fe)
    Aw[i,j] = 0
    An[i,j] = Dn + np.maximum(0, -Fn)
    As[i,j] = Ds + np.maximum(0, Fs)
    # calculate central coeff
    Ap[i, j] = (De + 2*Dw + Dn + Ds + np.maximum(0, Fe) + np.maximum(0, -Fw) + 
                      np.maximum(0, Fn) + np.maximum(0, -Fs))
    # get pressure slices
    Pe = P[i, j_plus]
    Pw = P[i, j]
    Pn = P[i_plus, j]
    Ps = P[i_min, j]
    # calculate sources
    Sx[1:-1,0] = (Pw - Pe) * dy / 2 + (1-alpha_uv) * Ap[i,j] * u_star[i,j]
    Sy[1:-1,0] = (Ps - Pn) * dx / 2 + (1-alpha_uv) * Ap[i,j] * v_star[i,j]

#-------------------------------------------------------------------------
    # Northern Wall
    i = ny
    i_min = ny-1
    i_plus = ny+1
    j = slice(2,nx)
    j_min = slice(1, nx-1)
    j_plus = slice(3, nx+1)
    
    #calculate advective coeffs
    Fe = dy * uface[i, j]
    Fw = dy * uface[i, j_min] 
    Fn = dx * vface[i, j] # north face initialized to 0.
    Fs = dx * vface[i_min, j]
    
    #calculate coeffs
    Ae[i,j] = De + np.maximum(0, -Fe)
    Aw[i,j] = Dw + np.maximum(0, Fw)
    An[i,j] = 0
    As[i,j] = Ds + np.maximum(0, Fs)
    # calculate central coeff
    Ap[i, j] = (De + Dw + 2*Dn + Ds + np.maximum(0, Fe) + np.maximum(0, -Fw) + 
                      np.maximum(0, Fn) + np.maximum(0, -Fs))
    # get pressure slices
    Pe = P[i, j_plus]
    Pw = P[i, j_min]
    Pn = P[i, j]
    Ps = P[i_min, j]
    
    # calculate sources
    Sx[-1,1:-1] = ((Pw - Pe) * dy / 2 + (1-alpha_uv) * Ap[i,j] * u_star[i,j] +
                   alpha_uv * lid_vel * (2*Dn + np.maximum(0,-Fn)) )
    
    Sy[-1,1:-1] = (Ps - Pn) * dx / 2 + (1-alpha_uv) * Ap[i,j] * v_star[i,j]
#-------------------------------------------------------------------------    
    # Southern Wall
    i = 1
    i_min = 0
    i_plus = 2
    j = slice(2,nx)
    j_min = slice(1, nx-1)
    j_plus = slice(3, nx+1)
    
    #calculate advective coeffs
    Fe = dy * uface[i, j]
    Fw = dy * uface[i, j_min] 
    Fn = dx * vface[i, j] # north face initialized to 0.
    Fs = dx * vface[i_min, j]
    
    #calculate coeffs
    Ae[i,j] = De + np.maximum(0, -Fe)
    Aw[i,j] = Dw + np.maximum(0, Fw)
    An[i,j] = Dn + np.maximum(0, -Fn)
    As[i,j] = 0
    # calculate central coeff
    Ap[i, j] = (De + Dw + Dn + 2*Ds + np.maximum(0, Fe) + np.maximum(0, -Fw) + 
                      np.maximum(0, Fn) + np.maximum(0, -Fs))
    # get pressure slices
    Pe = P[i, j_plus]
    Pw = P[i, j_min]
    Pn = P[i_plus, j]
    Ps = P[i, j]
    # calculate sources
    Sx[0, 1:-1] = (Pw - Pe) * dy / 2 + (1-alpha_uv) * Ap[i,j] * u_star[i,j]
    Sy[0, 1:-1] = (Ps - Pn) * dx / 2 + (1-alpha_uv) * Ap[i,j] * v_star[i,j]
        
    # Corner cells
    #-------------------------------------------------------------------------
    # South - West Corner
    i = 1
    j = 1
    i_min = 0
    j_min = 0
    i_plus = 2
    j_plus = 2
    
    #calculate advective coeffs
    Fe = dy * uface[i, j]
    Fw = dy * uface[i, j_min] 
    Fn = dx * vface[i, j]
    Fs = dx * vface[i_min, j]
    
    #calculate coeffs
    Ae[i,j] = De + np.maximum(0, -Fe)
    Aw[i,j] = 0
    An[i,j] = Dn + np.maximum(0, -Fn)
    As[i,j] = 0
    # calculate central coeff
    Ap[i, j] = (De + 2*Dw + Dn + 2*Ds + np.maximum(0, Fe) + np.maximum(0, -Fw) + 
                      np.maximum(0, Fn) + np.maximum(0, -Fs))
    # get pressure slices
    Pe = P[i, j_plus]
    Pw = P[i, j]
    Pn = P[i_plus, j]
    Ps = P[i, j]
    # calculate sources
    Sx[0, 0] = (Pw - Pe) * dy / 2 + (1-alpha_uv) * Ap[i,j] * u_star[i,j]
    Sy[0, 0] = (Ps - Pn) * dx / 2 + (1-alpha_uv) * Ap[i,j] * v_star[i,j]
    
    #-------------------------------------------------------------------------
    # South - East
    i = 1
    i_min = 0
    i_plus = 2
    j = nx
    j_min = nx-1
    j_plus = nx+1
    
    #calculate advective coeffs
    Fe = dy * uface[i, j]
    Fw = dy * uface[i, j_min] 
    Fn = dx * vface[i, j]
    Fs = dx * vface[i_min, j]
    
    #calculate coeffs
    Ae[i,j] = 0
    Aw[i,j] = Dw + np.maximum(0, Fw)
    An[i,j] = Dn + np.maximum(0, -Fn)
    As[i,j] = 0
    # calculate central coeff
    Ap[i, j] = (2*De + Dw + Dn + 2*Ds + np.maximum(0, Fe) + np.maximum(0, -Fw) + 
                      np.maximum(0, Fn) + np.maximum(0, -Fs))
    # get pressure slices
    Pe = P[i, j]
    Pw = P[i, j_min]
    Pn = P[i_plus, j]
    Ps = P[i, j]
    # calculate sources
    Sx[0, -1] = (Pw - Pe) * dy / 2 + (1-alpha_uv) * Ap[i,j] * u_star[i,j]
    Sy[0, -1] = (Ps - Pn) * dx / 2 + (1-alpha_uv) * Ap[i,j] * v_star[i,j]
    
    #-------------------------------------------------------------------------
    # North - East
    i = ny
    j = nx
    i_min = ny-1
    j_min = nx-1
    i_plus = ny+1
    j_plus = nx+1
    
    #calculate advective coeffs
    Fe = dy * uface[i, j]
    Fw = dy * uface[i, j_min] 
    Fn = dx * vface[i, j]
    Fs = dx * vface[i_min, j]
    
    #calculate coeffs
    Ae[i,j] = 0
    Aw[i,j] = Dw + np.maximum(0, Fw)
    An[i,j] = 0
    As[i,j] = Ds + np.maximum(0, Fs)
    # calculate central coeff
    Ap[i, j] = (2*De + Dw + 2*Dn + Ds + np.maximum(0, Fe) + np.maximum(0, -Fw) + 
                      np.maximum(0, Fn) + np.maximum(0, -Fs))
    # get pressure slices
    Pe = P[i, j]
    Pw = P[i, j_min]
    Pn = P[i, j]
    Ps = P[i_min, j]
    # calculate sources
    Sx[-1, -1] = ((Pw - Pe) * dy / 2 + (1-alpha_uv) * Ap[i,j] * u_star[i,j] + 
                  alpha_uv * lid_vel * (2*Dn + np.maximum(0, -Fn)) )
    
    Sy[-1, -1] = (Ps - Pn) * dx / 2 + (1-alpha_uv) * Ap[i,j] * v_star[i,j]
    
    #-------------------------------------------------------------------------
    # North - West
    i = ny
    i_min = ny-1
    i_plus = ny+1
    j = 1
    j_min = 0
    j_plus = 2
    
    #calculate advective coeffs
    Fe = dy * uface[i, j]
    Fw = dy * uface[i, j_min] 
    Fn = dx * vface[i, j]
    Fs = dx * vface[i_min, j]
    
    #calculate coeffs
    Ae[i,j] = De + np.maximum(0, -Fe)
    Aw[i,j] = 0
    An[i,j] = 0
    As[i,j] = Ds + np.maximum(0, Fs)
    # calculate central coeff
    Ap[i, j] = (De + 2*Dw + 2*Dn + Ds + np.maximum(0, Fe) + np.maximum(0, -Fw) + 
                      np.maximum(0, Fn) + np.maximum(0, -Fs))
    # get pressure slices
    Pe = P[i, j_plus]
    Pw = P[i, j]
    Pn = P[i, j]
    Ps = P[i_min, j]
    # calculate sources
    Sx[-1, 0] = ((Pw - Pe) * dy / 2 + (1-alpha_uv) * Ap[i,j] * u_star[i,j] + 
                  alpha_uv * lid_vel * (2*Dn + np.maximum(0, -Fn)) )
    
    Sy[-1, 0] = (Ps - Pn) * dx / 2 + (1-alpha_uv) * Ap[i,j] * v_star[i,j]
    #-------------------------------------------------------------------------
    
    return -alpha_uv * Ae, -alpha_uv * Aw, -alpha_uv * An, -alpha_uv * As, Ap, Sx, Sy


def unpad_arrays(arrays):
    """
    Takes a list of 2D arrays, each potentially padded with an outer layer of zeros, 
    and returns a list of the arrays with the padding removed.

    Args:
    arrays (list of np.array): List of 2D numpy arrays.

    Returns:
    list of np.array: List of unpadded 2D arrays.
    """
    unpadded_arrays = []

    for array in arrays:
        if array.ndim != 2:
            raise ValueError("All arrays must be 2D.")
        
        # Unpad the array
        unpadded_array = array[1:-1, 1:-1]
        unpadded_arrays.append(unpadded_array)

    return unpadded_arrays

def Create_link_matrix(Ae, Aw, An, As, Ap, nx, ny):
    """
    To create the dependency matrix:
        1.  Unravel all the coefficients so that each is of the size nx*ny
        2.  Use the central coefficients (Ap) to populate the main diagonal of the matrix
        3.  Use Ae and Aw to populate the off diagonals, remove either's last/first entry
        4.  Use An and As to populate the diagonal nx or ny to the right/left of the center
    """
    
    # Unravel Coefficients
    Ap_1d = np.ravel(Ap)
    Ae_1d = np.ravel(Ae)[:-1]
    Aw_1d = np.ravel(Aw)[1:]
    An_1d = np.ravel(An)[:-nx] #north point is exactly one row right
    As_1d = np.ravel(As)[nx:] #south point is exactly one row left
    
    # Position of Ap, Ae, Aw, An, As in the link matrix
    diagonal_pos = [0, 1, -1, nx, -nx]
    diagonals = [Ap_1d, Ae_1d, Aw_1d, An_1d, As_1d]
    # create csc
    link_matrix = diags(diagonals, diagonal_pos, format = "csc")
    
    return link_matrix

def solve_matrix(link_matrix, Source, state, nx, ny, method = 'gmres', solver_tol = 1e-5, precon_tol = 1e-5, reform = True):
    # Get state in 1d form
    state = np.ravel(state)
    # Get source in 1d form
    Source = np.ravel(Source)
    
    # Solve Matrix
    state1, state1_residuals = Linear_Solve(link_matrix, Source, state, 
                                solver = method, use_preconditioner = True, 
                                precon_tol = precon_tol, solver_tol = solver_tol, rest = 50)
    if reform:
        state1 = state1.reshape(ny,nx)
    
    return state1, state1_residuals

def face_velocity_calculation(uface, vface, u, v, P, Ap, alpha_uv, dy, dx, nx, ny):
    """
    Calculate the velocities at the cell faces using pressure weighted interpolation
    method, but not for k = 0, or we should use only distance weighted interpolation.
    args:
        u: x-direction velocity field for cell centers, type = 2d numpy array
        v: y-direction velocty field for cell centers, type = 2d numpy array
        P: pressure field at cell centers, type = 2d numpy array
        Ap: Central Coefficients for link matrix, type = 2d numpy array
        dx, dy: cell sizes, type = float
    Outs:
        u_face x-direction velocity field at the cell faces, type = 2d numpy array
        v_face: y-direction velocity field at the cell faces, type = 2d numpy array
    Formula for an east face velocity:
        uface_e= 0.5 * (uhat_P + uhat_E) + 1/4 * [(PE - PW)/APP + ((PEE - PP)/APE)]* dy - [1/APE + 1/APP] * (PE + PP) * dy/2
    Formula for a north face velocity:
        vface_n= 0.5 * (vhat_P + vhat_N) + 1/4 * [(PE - PW)/APP + ((PEE - PP)/APE)]* dy - [1/APE + 1/APP] * (PE + PP) * dy/2
    """

    # might need to pad ghost pressure with the boundary pressure here
    P[:,-1] = P[:,-2]   # East ghost pressure
    P[:,0] = P[:,1]     # West ghost pressure
    P[-1,:] = P[-2,:]   # North ghost pressure
    P[0,:] = P[1,:]     # South ghost pressure
    
    # East/West velocities: We need EE & WW cells, so 2 cells on either side of solving face, so index interior from 2:-2
    i = slice(1, ny+1)
    
    j = slice(1, nx)
    j_plus = slice(2, nx+1)
    j_plus2 = slice(3, nx+2)
    j_minus = slice(0, nx-1)
    
    #velocities
    uW = u[i,j]
    uE = u[i,j_plus]
    #central corffs
    APw = Ap[i,j]
    APe = Ap[i,j_plus]
    #pressures
    Pw = P[i,j]
    Pww = P[i,j_minus]
    Pe = P[i,j_plus]
    Pee = P[i,j_plus2]

    # face velocities
    uface[i,j] = (0.5*(uW + uE) + 0.25 * alpha_uv * dy * ( (Pe - Pww)/APw + (Pee - Pw)/APw )
                  - alpha_uv * 0.5 * (1 / APe + 1 / APw) * (Pe - Pw) * dy)
    
    # North/South velocities
    i = slice(1, ny)
    i_plus = slice(2, ny+1)
    i_plus2 = slice(3, ny+2)
    i_minus = slice(0, ny-1)
    
    j = slice(1, nx+1)
    
    #velocities
    vS = v[i,j]
    vN = v[i_plus,j]
    #central coeffs
    APs = Ap[i,j]
    APn = Ap[i_plus,j]
    #pressures
    Ps = P[i,j]
    Pss = P[i_minus,j]
    Pn = P[i_plus,j]
    Pnn = P[i_plus2,j]
    
    vface[i,j] = (0.5 * (vS + vN) + 0.25 * alpha_uv * dx * ( (Pn - Pss)/APs + (Pnn - Ps)/APn ) 
                  - alpha_uv * 0.5 * (1/APn + 1/APs) * (Pn - Ps) * dx)
    
    return uface, vface

def pressure_correction_coeffs(Ap, uface, vface, alpha_uv, dy, dx, nx, ny):
    """
    Function which calculates and returns the coefficients to solve the pressure
    correction.
    Args:
        Ap: Central coefficients for momentum balance, type = 2d numpy array
        uface: x-direction cell face velocities AFTER pressure weighted interpolation, type = 2d numpy array
        vface: y-direction cell face velocities AFTER pressure weighted interpolation, type = 2d numpy array
        dy, dx: mesh size, floats
    Outs:
        Ap, Ae, Aw, An, As: Coefficients to solve pressure correction in linked matrix form
        Sp: Source terms representing the mass imbalance, also used for the linked matrix function
        
    """
    # pad the outside of the central array with 1's so we can slice and avoid
    # boundary conditions
    Ap[0,:] = 1
    Ap[-1,:] = 1
    Ap[:,0] = 1
    Ap[:,-1] = 1
    
    # The 'keepX' arrays represent which of the coefficients to keep, and which
    # to set to 0 during slicing when the boundaries are reached.
    keepE = np.ones((ny,nx))
    keepE[:,-1] = 0
    keepW = np.ones((ny,nx))
    keepW[:,0] = 0
    keepN = np.ones((ny,nx))
    keepN[-1,:] = 0
    keepS = np.ones((ny,nx))
    keepS[0,:] = 0
    
    i = slice(1, ny+1)
    i_min = slice(0, ny)
    i_plus = slice(2, ny+2)
    j = slice(1,nx+1)
    j_min = slice(0, nx)
    j_plus = slice(2, nx+2)
    
    #Original Coefficients from Ap
    APP = Ap[i, j]
    APE = Ap[i, j_plus]
    APW = Ap[i, j_min]
    APN = Ap[i_plus, j]
    APS = Ap[i_min, j]
    
    # Neighbor Cells
    Ae = dy**2/2 * alpha_uv * (1/APE + 1/APP) * keepE
    Aw = dy**2/2 * alpha_uv * (1/APW + 1/APP) * keepW
    An = dx**2/2 * alpha_uv * (1/APN + 1/APP) * keepN
    As = dx**2/2 * alpha_uv * (1/APS + 1/APP) * keepS
    
    # Central Coefficient
    AP = Ae + Aw + An + As
    
    # Create source term with face velocities:
    ue = uface[i, j]
    uw = uface[i, j_min]
    vn = vface[i,j]
    vs = vface[i_min,j]
    
    #Source term:
    Sp = -((ue - uw)*dy + (vn - vs)*dx)
    
    return AP, -Ae, -Aw, -An, -As, Sp


def correct_pressure(P, correction, relaxation = 1):
    return P + relaxation * correction

def correct_velocities(uv, correction, relaxation = 1):
    uv_new = uv + correction
    return relaxation * uv_new + (1 - relaxation) * uv  
 
def velocity_correction(u_center_prime, v_center_prime, Ap, P, dy, dx):
    """
    Parameters
    ----------
    Ap : 2d numpy array
        Central coefficients from momentum balance link matrix
    P : 2d numpy array
        Pressure correction term at cell centers (P')
    dy : float
        Vertical cell length
    dx : float
        Horizontal cell length

    Returns
    -------
    u_center_prime : 2d numpy array
        velocity correction terms for central cells in the x direction
    v_center_prime : 2d numpy array
        velocity correction terms for central cells in the y direction
    """
    # P = np.pad(P, pad_width = 1, mode = 'constant', constant_values = 0)
    # Set the pressure of the ghost cells equal to the pressure of the boundary
    # cell acording to the boundary layer approximation
    P[:,-1] = P[:,-2]   # East ghost pressure
    P[:,0] = P[:,1]     # West ghost pressure
    P[-1,:] = P[-2,:]   # North ghost pressure
    P[0,:] = P[1,:]     # South ghost pressure
    
    # Central Cells:
    # create indices
    i = slice(1, ny+1)
    i_plus = slice(2, ny+2)
    i_min = slice(0, ny)
    j = slice(1, nx+1)
    j_plus = slice(2, nx+2)
    j_min = slice(0, nx)
    
    #Get east, west, north, south pressures
    Pe = P[i, j_plus]
    Pw = P[i, j_min]
    Pn = P[i_plus, j]
    Ps = P[i_min, j]
    
    #calculate velocity correction terms:
    u_center_prime[i,j] = (Pw - Pe) * dy / (2 * Ap[i,j])
    v_center_prime[i,j] = (Ps - Pn) * dx / (2 * Ap[i,j])

    return u_center_prime, v_center_prime

def face_velocity_correction(Ap, P, uface, vface, dy, dx):
    """

    Parameters
    ----------
    Ap : TYPE numpy 2d array
        DESCRIPTION.Central coefficients from momentum balance link matrix
    P : TYPE numpy 2d array
        DESCRIPTION. Pressure correction term at cell centers
    uface : TYPE numpy 2d array
        DESCRIPTION. u velocities at the cell faces
    vface : TYPE numpy 2d array
        DESCRIPTION. v velocities at the cell faces
    dy : TYPE float
        DESCRIPTION. Vertical cell length
    dx : TYPE float
        DESCRIPTION. Horizontal cell length

    Returns
    -------
    uface : TYPE numpy 2d array
        DESCRIPTION. u face velocity correction terms
    vface : TYPE numpy 2d array
        DESCRIPTION. v face velocity correction terms

    """
    
    # create indices
    i = slice(1, -1)
    i_plus = slice(2, -1)
    i_min = slice(1, -2)
    j = slice(1, -1)
    j_plus = slice(2, -1)
    j_min = slice(1, -2)
    
    #Get east, west, north, south pressures
    # 10x10
    Pe = P[i,j_plus]
    Pw = P[i,j_min]
    Pn = P[i_plus,j]
    Ps = P[i_min,j]
    
    # get east west north south central coefficients
    Ape = Ap[i,j_plus]
    Apw = Ap[i,j_min]
    Apn = Ap[i_plus,j]
    Aps = Ap[i_min,j]
    
    uface_interior = dy * (1/Ape + 1/Apw) * (Pw - Pe) / 2
    
    vface_interior = dx * (1/Aps + 1/Apn) * (Ps - Pn) / 2
    
    # Replace interior cell face velocities with calculated ones, leaving the 
    # edge faces at 0
    uface[i,1:-1] = uface_interior
    vface[1:-1,j] = vface_interior
    
    return uface, vface

def check_convergence(u, u_new, tolerance):
    norm = np.linalg.norm(u_new - u)
    if norm < tolerance:
        return True, norm
    else:
        return False, norm

def save_arrays_to_csv(arrays, filenames, path):
    """
    Save multiple numpy arrays to CSV files in a specified path.
    
    Parameters:
    arrays (list of np.array): List of numpy arrays to save.
    filenames (list of str): List of filenames for the CSV files.
    path (str): Directory path where the files will be saved.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    for array, filename in zip(arrays, filenames):
        full_path = os.path.join(path, filename)
        df = pd.DataFrame(array)
        df.to_csv(full_path, index=False)

        
def read_arrays_from_csv(filenames, directory=''):
    """
    Read multiple CSV files into numpy arrays from a specified directory.
    
    Parameters:
    filenames (list of str): List of filenames to read.
    directory (str): Directory path where the files are located.

    Returns:
    list of np.array: List of numpy arrays read from the CSV files.
    """
    arrays = []
    for filename in filenames:
        full_path = os.path.join(directory, filename)
        df = pd.read_csv(full_path)
        arrays.append(df.values)
    return arrays


# These are 2 custom functions I made to try and play with adaptive relaxation.
# I wouldn't use them.
def is_stagnant(history, threshold):
    # Returns True if the history is within a threshold range
    max_val = max(history)
    min_val = min(history)
    return (max_val - min_val) / max_val < threshold

def adaptive_relaxation(alpha_uv, alpha_p, u_norm, v_norm, p_norm, memory=10, decrease_factor=0.95, stagnation_threshold=0.01):
    # Check for stagnation in pressure residuals
    if len(p_norm) >= memory and is_stagnant(p_norm[-memory:], stagnation_threshold):
        alpha_p = max(0.00001, alpha_p * decrease_factor)
    
    # Check for stagnation in velocity residuals
    if len(u_norm) >= memory and is_stagnant(u_norm[-memory:], stagnation_threshold):
        alpha_uv = max(0.1, alpha_uv * decrease_factor)
    
    if len(v_norm) >= memory and is_stagnant(v_norm[-memory:], stagnation_threshold):
        alpha_uv = max(0.1, alpha_uv * decrease_factor)
    
    return alpha_uv, alpha_p
#%%
# Parameters
nx, ny = 40, 40 # Number of grid points in x and y directions
lx, ly = 1, 1  # Length of the cavity in x and y directions
dx, dy = lx / nx, ly / ny  # Grid spacing in x and y directions
x = np.linspace(-lx/2,lx/2, nx+2)
y = np.linspace(-ly/2,ly/2, ny+2)
X,Y = np.meshgrid(x,y)
ar = dx/dy
ra = dy/dx
rho = 1.0  # Density
characteristic_length = lx
Re = 1000 # Reynolds Number
u_lid = 1.0  # Lid velocity
method = 'bicgstab' #Specify solver method


# calculate dynamic viscosity from reynolds number, flow speed, and length
mu = rho * u_lid * characteristic_length / Re

# Initialize fields
u = np.zeros((ny+2, nx+2)) # initial x cell velocity
v = np.zeros((ny+2, nx+2)) # initial y cell velocity
u_star = np.zeros((ny+2, nx+2)) # momentum corrected velocity
v_star = np.zeros((ny+2, nx+2)) # momentum corrected velocity
uface = np.zeros((ny+2, nx+1))    # x-velocity at faces
vface = np.zeros((ny+1, nx+2))    # y-velocity at faces
uface_star = np.zeros((ny+2, nx+1)) # u face 
vface_star = np.zeros((ny+1, nx+2))
uface_prime = np.zeros((ny+2, nx+1))
vface_prime = np.zeros((ny+1, nx+2))
u_center_prime = np.zeros((ny+2, nx+2)) 
v_center_prime = np.zeros((ny+2, nx+2)) 

# Initialize momentum coefficients with ghost cells:
Ap = np.zeros((ny+2, nx+2))
Ae = np.zeros((ny+2, nx+2))
Aw = np.zeros((ny+2, nx+2))
An = np.zeros((ny+2, nx+2))
As = np.zeros((ny+2, nx+2))
# Source terms
Sx = np.zeros((ny, nx))
Sy = np.zeros((ny, nx))

# Initialize pressure coefficients with ghost cells:
AP_p = np.zeros((ny+2, nx+2))
Ae_p = np.zeros((ny+2, nx+2))
Aw_p = np.zeros((ny+2, nx+2))
An_p = np.zeros((ny+2, nx+2))
As_p = np.zeros((ny+2, nx+2))
# Source terms
Sp = np.zeros((ny, nx))

# Set ghost cell velocity
u[-1,:] = u_lid
u_star[-1,:] = u_lid
uface[-1,:] = u_lid

P = np.zeros((ny+2, nx+2))          # Pressure
P_prime = np.zeros((ny+2, nx+2))    # Corrected Pressure

#lists to hold residuals
iterations = []
P_res = []
u_res = []
v_res = []
#%%
# Discretization parameters (these might need adjustment)
alpha_uv, alpha_p = 0.3, 0.001  # Under-relaxation factors
use_adaptive_relax = False
mem = 10

# Iteration parameters
max_iter = 4000
tolerance = 1e-4
solver_tol = 1e-10
precon_tol = 1e-9

# Start Timer
T0 = time.time()
# Run in loop:
for k in tqdm(range(max_iter), desc=f"Solving SIMPLE for {max_iter} max iterations"):
    # Calculate coefficients
    Ae, Aw, An, As, Ap, Sx, Sy = calculate_link_coeffs(uface, vface, u, v, P, Ap, Ae, Aw, An, As, Sx, Sy, alpha_uv, u_lid, Re, ra, ar, dx, dy, nx, ny)
    
    # Unpad coefficient arrays
    Ap_center, Ae_center, Aw_center, An_center, As_center = unpad_arrays([Ap, Ae, Aw, An, As])
    
    # Create link matrix
    link_matrix = Create_link_matrix(Ae_center, Aw_center, An_center, As_center, Ap_center, nx, ny)
    # link_matrix_dense = link_matrix.toarray() # for troubleshooting
    
    # Unpad the velocity arrays to solve momentum:
    u_unpadded, v_unpadded = unpad_arrays([u, v])
    
    # Solve x and y momentum equations
    u_star[1:-1,1:-1], _ = solve_matrix(link_matrix, Sx, u_unpadded, nx, ny, solver_tol = solver_tol, precon_tol = precon_tol, method = method)
    v_star[1:-1,1:-1], _ = solve_matrix(link_matrix, Sy, v_unpadded, nx, ny, solver_tol = solver_tol, precon_tol = precon_tol, method = method)
    
    # Get face velocities
    uface_star, vface_star = face_velocity_calculation(uface, vface, u_star, v_star, P, Ap, alpha_uv, dy, dx, nx, ny)
    
    # Get pressure correction coefficients
    AP_p, Ae_p, Aw_p, An_p, As_p, Sp = pressure_correction_coeffs(Ap, uface_star, vface_star, alpha_uv, dy, dx, nx, ny)
    
    #Create link matrix:
    link_matrix_p = Create_link_matrix(Ae_p, Aw_p, An_p, As_p, AP_p, nx, ny)
    # link_matrix_p_dense = link_matrix_p.toarray() # for troubleshooting
    
    # Unpad Pressure grid
    P_unpadded = unpad_arrays([P])
    
    # Solve pressure correction equations:
    P_prime[1:-1,1:-1], _ = solve_matrix(link_matrix_p, Sp, P_unpadded, nx, ny, solver_tol = solver_tol, precon_tol = precon_tol, method = method)
    
    # Calculate central cell velocity corrections:
    u_center_prime, v_center_prime = velocity_correction(u_center_prime, v_center_prime, Ap, P_prime, dy, dx)
    
    uface_prime, vface_prime = face_velocity_correction(Ap, P_prime, uface_prime, vface_prime, dy, dx)
    
    # Correct Pressure
    P_corrected = correct_pressure(P, P_prime, alpha_p)
    
    # Correct U velocity
    u_center_corrected = correct_velocities(u_star, u_center_prime, alpha_uv)
    uface_corrected = correct_velocities(uface_star, uface_prime, alpha_uv)
    
    # Correct V velocities
    v_center_corrected = correct_velocities(v_star, v_center_prime, alpha_uv)
    vface_corrected = correct_velocities(vface_star, vface_prime, alpha_uv)
    
    # Calculate L2 Norm and check for convergence
    converged_u, u_norm = check_convergence(u, u_center_corrected, tolerance)
    converged_v, v_norm = check_convergence(v, v_center_corrected, tolerance)
    converged_p, p_norm = check_convergence(P, P_corrected, tolerance)
    
    if k%mem == 0: # add to residuals every 10th iteration
        iterations.append(k)
        P_res.append(p_norm)
        u_res.append(u_norm)
        v_res.append(v_norm)
        if use_adaptive_relax:
            alpha_uv, alpha_p = adaptive_relaxation(
            alpha_uv, alpha_p, u_res, v_res, P_res,
            mem)
    
    # Check for convergence
    if converged_u & converged_v & converged_p:
        print(f"\nConverged after {k} iterations")
        print(f"U velocity L2 norm: {u_norm}")
        print(f"V velocity L2 norm: {v_norm}")
        print(f"Pressure L2 norm: {p_norm}")
        break

    # Re-initialize parameters for the next iteration
    u = u_center_corrected
    v = v_center_corrected
    uface = uface_corrected
    vface = vface_corrected
    P = P_corrected

# End Timer
T1 = time.time()
print(f"\nRun time: {T1-T0:.2f} seconds")

# Plot convergence
plot_residuals(iterations, np.array([P_res, u_res, v_res]))

#%% Save results to csv

mesh = '40' # name of mesh shape saved
# Specify path to save files:
directory_path = '/Users/declanbracken/Documents/U_of_T/MIE_1210/A4/Results_CSVs'
# Saving arrays to CSV
arrays = [X, Y, u, v, P]
filenames = [f'X{mesh}.csv', f'Y{mesh}.csv', f'u{mesh}.csv', f'v{mesh}.csv', f'P{mesh}.csv']
save_arrays_to_csv(arrays, filenames, directory_path)

#%% Optionally Load in results for post processing

mesh = '257' # name of mesh shape saved
# Optionally, load in a saved csv for post-processing
filenames = [f'X{mesh}.csv', f'Y{mesh}.csv', f'u{mesh}.csv', f'v{mesh}.csv', f'P{mesh}.csv']
read_directory_path = f'/Users/declanbracken/Documents/U_of_T/MIE_1210/A4/Results_CSVs/{mesh}mesh'
X, Y, u, v, P = read_arrays_from_csv(filenames, read_directory_path)

#%% Post processing and figure Saving

# plotting options
num_levels = 80
colorway = 'bwr'
# create folder path based in mesh shape name
mesh = '40' # name of mesh shape saved
mesh_path = f'/Users/declanbracken/Documents/U_of_T/MIE_1210/A4/Results_CSVs/{mesh}mesh'

# Create path + image names (png)
save_path_u = os.path.join(mesh_path, 'u_profile')
save_path_v = os.path.join(mesh_path, 'v_profile')
save_path_p = os.path.join(mesh_path, 'p_profile')
save_path_stream = os.path.join(mesh_path, 'streamlines')
save_path_Ghia = os.path.join(mesh_path, 'centerline')
save_path_Ghia_vert = os.path.join(mesh_path, 'centerline_vert')
save_path_eddy = os.path.join(mesh_path, 'Eddy')

# Remove Ghost values:
X_noborder, Y_noborder, u_noborder, v_noborder, P_noborder = (
    unpad_arrays([X, Y, u, v, P]))

# get bottom right corner for eddy visualization:
cutoff = lx//6
X_bot = X_noborder[:cutoff,:cutoff]
Y_bot = Y_noborder[:cutoff,:cutoff]
u_bot = u_noborder[:cutoff,:cutoff]
v_bot = v_noborder[:cutoff,:cutoff]

# streamlines
streamlines(X_noborder, Y_noborder, u_noborder, v_noborder, cmap=colorway, den =3.5)#, save_path = save_path_stream) 

# Get streamlines from corner Eddy:
# streamlines(X_bot, Y_bot, u_bot, v_bot, cmap=colorway, den = 2)#, save_path = save_path_eddy)

# U velocity plot
contour_heatplot(X_noborder, Y_noborder, u_noborder, label = "Velocity", 
                  num_levels = num_levels, cmap = colorway)#, save_path = save_path_u)
# V velocity plot
contour_heatplot(X_noborder, Y_noborder, v_noborder, label = "Velocity", 
                  num_levels = num_levels, cmap = colorway)#, save_path = save_path_v)
# P velocity plot
contour_heatplot(X_noborder, Y_noborder, P_noborder, label = "Pressure", 
                  num_levels = num_levels, cmap = colorway)#, save_path = save_path_p)

# # Horizontal cross-section velocity profile
plot_horizontal_velocity_profile(X+lx/2, Y+ly/2, v, compare_Ghia=True, fs = 20)#, save_path = save_path_Ghia) #, save_path = save_path_Ghia

# Vertical Cross-section velocity profile
plot_vertical_velocity_profile(X+lx/2, Y+ly/2, u, compare_Ghia = True, fs = 20)#, save_path = save_path_Ghia_vert) #, save_path = save_path_Ghia_vert

#%%

# Solving SIMPLE for 35000 max iterations: 100%|██████████| 35000/35000 [13:43:16<00:00,  1.41s/it]    
# Run time: 49396.23 seconds