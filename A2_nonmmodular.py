#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:21:22 2023

@author: declanbracken
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres, lgmres, cg, bicgstab, cgs, spilu
import time
from scipy.interpolate import RegularGridInterpolator

# Set plot resolution
plt.rcParams['figure.dpi'] = 600
plt.rcParams['text.usetex'] = False
#%%
"""
Functions cell
"""


def create_grid(width, height, nx, ny, r, default_value = 10):
    """
    Initialize a 2d array of dictionaries which hold index, neighboor, and
    boundary information for each cell, to be looped through later.
    """
    # Initial temperature values
    scalar_array = default_value * np.ones(nx*ny)
    
    #arrays to hold spacing vals
    x_spacing = np.zeros(nx+1)
    y_spacing = np.zeros(ny+1)
    
    # If r = 1 make even spacing, else use inflation factor
    if r == 1:
        x_spacing = width/nx * np.ones(nx+1)
        y_spacing = height/ny * np.ones(ny+1)
    else:
        #initalize
        x_spacing[0] = (1-r)/(1-r**(nx/2))*width/2
        y_spacing[0] = (1-r)/(1-r**(ny/2))*height/2
        
        #calculate half space of the domain for inflation condition
        half_width = width/2
        half_height = height/2
        
        #calculate successive spacing in x direction
        for i in range(1,nx+1):
            #if the sum of the current spacing is less than halfway
            if np.sum(x_spacing[:i]) <= half_width:
                x_spacing[i] = r*x_spacing[i-1]
            else:
                x_spacing[i] = x_spacing[i-1]/r
                
        #calculate successive spacing in y direction
        for j in range(1,ny+1):
            if np.sum(y_spacing[:j]) <= half_height:
                y_spacing[j] = r*y_spacing[j-1]
            else:
                y_spacing[j] = y_spacing[j-1]/r
        
    #initialize grid object
    grid = np.zeros((nx, ny), dtype=object)
    
    #Initialize properties for all cells
    for i in range(nx):
        for j in range(ny):
            
            #Initialize properties for a single cell
            cell = {

                "neighbors": [], #list of neighbor indices
                "neighbor_spacing": [], #list of the spacing between neighbors
                "neighbor_surface_area": [], #the area for the flux to pass through
                "boundaries": [], #list of boundaries
                "bound_spacing": [], #spacing from boundary
                "boundary_surface_area": [], #boundary area
            }
            
            #add cell dictionary to the grid
            grid[i, j] = cell
    
    
    # Connect neighboring cells, add their spacing, and list cells with boundaries
    for i in range(nx):
        for j in range(ny):
            
            #If not at the north wall
            if i > 0:
                grid[i, j]["neighbors"].append([i-1,j]) #get neighbor index
                grid[i, j]["neighbor_spacing"].append(y_spacing[i]) #get distance to neighbor
                grid[i, j]["neighbor_surface_area"].append((x_spacing[i] + x_spacing[i+1])/2) #calculate cell area
                
            else: #Add north boundary
                grid[i, j]["boundaries"].append('north') #set boundary orientation
                grid[i, j]["bound_spacing"].append(y_spacing[0]/2) #set distance to wall
                grid[i, j]["boundary_surface_area"].append((x_spacing[i] + x_spacing[i+1])/2) #calculate wall area

            #If not at the south wall
            if i < nx - 1:
                grid[i, j]["neighbors"].append([i + 1, j])
                grid[i, j]["neighbor_spacing"].append(y_spacing[i+1])
                grid[i, j]["neighbor_surface_area"].append((x_spacing[i] + x_spacing[i+1])/2)

            else:
                grid[i, j]["boundaries"].append('south') 
                grid[i, j]["bound_spacing"].append(y_spacing[-1]/2)
                grid[i, j]["boundary_surface_area"].append((x_spacing[i] + x_spacing[i+1])/2)

            #If not on the west wall    
            if j > 0:
                grid[i, j]["neighbors"].append([i, j - 1])
                grid[i, j]["neighbor_spacing"].append(x_spacing[j])
                grid[i, j]["neighbor_surface_area"].append((y_spacing[j] + y_spacing[j+1])/2)

            else:
                grid[i, j]["boundaries"].append('west')
                grid[i, j]["bound_spacing"].append(x_spacing[0]/2)
                grid[i, j]["boundary_surface_area"].append((y_spacing[j] + y_spacing[j+1])/2)

            #If not on the east wall    
            if j < ny - 1:
                grid[i, j]["neighbors"].append([i, j + 1])
                grid[i, j]["neighbor_spacing"].append(x_spacing[j+1])
                grid[i, j]["neighbor_surface_area"].append((y_spacing[j] + y_spacing[j+1])/2)

            else:
                grid[i, j]["boundaries"].append('east')
                grid[i, j]["bound_spacing"].append(x_spacing[-1]/2)
                grid[i, j]["boundary_surface_area"].append((y_spacing[j] + y_spacing[j+1])/2)

    return grid, x_spacing, y_spacing, scalar_array

def diffusion_coeffs(transfer_coeff, Tao, area, spacing):
    conduction_coeff = Tao * area / spacing
    return conduction_coeff #, vertical_conduction_coeff

def fixed_temp_bound_coeff(transfer_coeff, Tao, area, spacing):
    return Tao * area / spacing

def robin_bound_coeff(transfer_coeff, Tao, area, spacing):
    return area / (spacing / Tao + 1 / transfer_coeff) #*d

def insulated_bound(transfer_coeff, Tao, area, spacing):
    return 0

def create_dependency_matrix(grid, nx, ny, dx, dy, permeability, bound_info, ext_temps):
    """
    ARGS:
        grid:           2d numpy array of dictionaries where each dictionary
                        holds information about that cell. Specifically the
                        indices of it's neighbors, the neighbor spacing and
                        area, and whether or not that cell is on a boundary.
                        Also has boundary spacing and area.
        
        bound_coeffs:   dictionary of boundary coefficients for north, south, east
                        west.
        bound_info:     dictionary of the boundary function, the permeability coefficient,
                        and the transfer coefficient for constant temp and mixed bounds.
        ext_temps:      dictionary of boundary temperatures for north, south, east, west
    
    To create the dependency matrix:
        1.  Loop through all cells of the grid.
        2.  For the kth cell, calculte it's position in the csr (k,k). (where k = i * ny + j)
        3.  For the kth cell, loop through it's neighbors and populate the kth
            row with the neighbor coefficients given the neighbor indices.
        4.  For the kth cell, add up the coefficients of it's neighbors for
            it's own coefficient.
        5.  Check if it has boundary conditions, if it does, add a source term 
            to the cell's coefficient, and to the source terms array.
        6.  Finally, populate the (k,k) (diagonal) position with the total 
            of the neighbor + source coefficients (if source exists).
        7.  Return the CSR matrix, which (for each row) should have at least
            3 and at most 5 non zero values (3 if it's a corner in the grid,
            5 if it's an interior node).
            
    """
    
    num_cells = nx * ny #total number of cells
    
    rows = [] #arrays to hold csr matrix vals
    cols = []
    coeffs = []
    source_terms = np.zeros(num_cells) #array to hold source terms

    # Loop through all grid points
    for i in range(nx):
        for j in range(ny):
            cell = grid[i, j]
            k = i * ny + j
            ap = 0
            #Get neighbor info
            neighbors = cell["neighbors"]
            neighbor_spacings = cell["neighbor_spacing"]
            neighbor_areas = cell["neighbor_surface_area"]
            
            #Loop through Neighbors
            for n, neighbor_idx in enumerate(neighbors):
                neighbor_spacing = neighbor_spacings[n]
                neighbor_area = neighbor_areas[n]
                dependency_coefficient = diffusion_coeffs(0, permeability, 
                                                          neighbor_area, 
                                                          neighbor_spacing)
                rows.append(k)
                col_idx = neighbor_idx[0] * ny + neighbor_idx[1]
                cols.append(col_idx)
                coeffs.append(dependency_coefficient)
                
                # add to central coeff
                ap -= dependency_coefficient
            
            #Loop through boundaries if applicable
            boundaries = cell["boundaries"]
            
            if boundaries is not None:
                boundary_areas = cell["boundary_surface_area"]
                boundary_spacings = cell["bound_spacing"]
                for z, boundary in enumerate(boundaries):
                    
                    #get boundary area & spacing
                    boundary_area = boundary_areas[z]
                    boundary_spacing = boundary_spacings[z]
                    
                    #Get Boundary Info
                    boundary_func, Tao, transfer_coeff = bound_info[boundary]
                    
                    #Calculate coefficient using boundary_func
                    bound_coeff = boundary_func(transfer_coeff, Tao, boundary_area, boundary_spacing)
                    
                    #Update central coeff
                    ap -= bound_coeff
                    
                    #Update source array {b}
                    source_terms[k] -= bound_coeff * ext_temps[boundary]

            #Append diagonal element
            rows.append(k)
            cols.append(k)
            coeffs.append(ap)

    dependency_matrix_csr = csr_matrix((coeffs, (rows, cols)), shape=(num_cells, num_cells))

    return dependency_matrix_csr, source_terms


def Linear_Solve(A, b, x0, solver = 'gmres', use_preconditioner = True, 
                 precon_tol = 1e-4, solver_tol = 1e-5, rest = 50, callback = None):
    """
    ARGS:
        A:  Dependancy matrix
        b:  Source terms array
        x0: Initial Guess
        solver: choose what solver you want to use
        use_preconditioner: boolean to choose preconditioning with ilu factorization (leave on)
    """
    # List to store the residuals
    residuals = []
    
    # Callback function to capture the residuals
    def callback(xk):
        r = np.linalg.norm(b - A @ xk)
        residuals.append(r)
    
    #List of solver methods
    solver_list = ['gmres', 'lgmres','cg', 'bicgstab', 'cgs']
    
    #raise error if unspecified solver is used
    if solver not in (solver_list):
        raise ValueError("Invalid solver. Supported options are 'gmres', 'lgmres', 'cg', 'bicg', or 'cgs'.")
    
    #Construct preconditioner seperately from solving
    if use_preconditioner:
        #apply incomplete LU decomposition to approximate the inverse of A, with optional tolerance
        preconditioned_A = spilu(A, drop_tol = precon_tol)
        M_x = LinearOperator(A.shape, preconditioned_A.solve)
        
    else:
        M_x = None
    
    if solver == 'lgmres':
        x, info = lgmres(A, b, x0 = x0, M=M_x, atol=solver_tol, callback=callback)
    elif solver == 'cg':
        x, info = cg(A, b, x0 = x0,M=M_x, atol=solver_tol, callback=callback)
    elif solver == 'bicgstab':
        x, info = bicgstab(A, b, x0 = x0, M=M_x,atol=solver_tol, callback=callback)
    elif solver == 'cgs':
        x, info = cgs(A, b, x0 = x0, M=M_x,atol=solver_tol, callback=callback)
    else:
        x, info = gmres(A, b, x0 = x0, M=M_x,atol=solver_tol, restart = rest, callback=callback)
     
    #check for successful convergeance
    if info != 0:
        print("Warning: {} did not converge (info={}).".format(solver,info))
    
    return x, residuals

#Plotting Methods:
    
def heatplot(x_spacing,y_spacing, width, height, nx, ny, phi):
    
    # Reshape to 2d array
    phi_out_mat = phi.reshape((nx,ny), order='C')
    
    x_spacing_func = x_spacing
    y_spacing_func = y_spacing

    # Adjust for distance from boundaries
    x_spacing_func[0], x_spacing_func[-1] = x_spacing_func[0]/2, x_spacing_func[-1]/2
    y_spacing_func[0], y_spacing_func[-1] = y_spacing_func[0]/2, y_spacing_func[-1]/2

    x_coords = np.cumsum(x_spacing_func[:-1]) - 0.5*width
    y_coords = np.cumsum(y_spacing_func[:-1]) - 0.5*height
    
    # Create a meshgrid 
    x, y = np.meshgrid(x_coords, y_coords)

    #Pplot
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(x,y,np.flip(phi_out_mat,0),cmap = 'hot') #coolwarm
    cbar = plt.colorbar(pcm, label='Temperature')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Temperature Distribution')

    plt.show()
    return x_coords, y_coords


def contour_heatplot(x_spacing, y_spacing, width, height, nx, ny, phi):
    
    # Reshape to 2d array
    phi_out_mat = phi.reshape((nx,ny), order='C')

    x_spacing_func = x_spacing
    y_spacing_func = y_spacing

    # Adjust for distance from boundaries
    x_spacing_func[0], x_spacing_func[-1] = x_spacing_func[0]/2, x_spacing_func[-1]/2
    y_spacing_func[0], y_spacing_func[-1] = y_spacing_func[0]/2, y_spacing_func[-1]/2
    
    # Get node positions
    x_coordinates = np.cumsum(x_spacing_func[:-1]) - 0.5*width
    y_coordinates = np.cumsum(y_spacing_func[:-1]) - 0.5*height
    
    # Create a meshgrid 
    x, y = np.meshgrid(x_coordinates, y_coordinates)

    # Plot
    fig, ax = plt.subplots()
    fs = 16
    
    # Using contourf for filled contours
    contour = ax.contourf(x, y, np.flip(phi_out_mat, 0), levels = 60, cmap='coolwarm') #coolwarm
    cbar = plt.colorbar(contour)
    cbar.set_label("Temperature", fontsize=fs)
    
    cbar.ax.tick_params(labelsize=fs)
    plt.tick_params(axis='both', labelsize=fs)
    
    ax.set_xlabel('X Coordinate', fontsize=fs)
    ax.set_ylabel('Y Coordinate', fontsize=fs)
    ax.set_title('Temperature Distribution Contour', fontsize=fs)

    plt.show()
    return x_coordinates, y_coordinates


def gridplot(x_spacing, y_spacing, width, height):
    fs = 16
    
    x_spacing_func = x_spacing
    y_spacing_func = y_spacing

    # Adjust for distance from boundaries
    x_spacing_func[0], x_spacing_func[-1] = x_spacing_func[0]/2, x_spacing_func[-1]/2
    y_spacing_func[0], y_spacing_func[-1] = y_spacing_func[0]/2, y_spacing_func[-1]/2
    
    # Get node positions
    x_coords = np.cumsum(x_spacing_func[:-1]) - 0.5*width
    y_coords = np.cumsum(y_spacing_func[:-1]) - 0.5*height
    
    # Create a meshgrid 
    x, y = np.meshgrid(x_coords, y_coords)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b')
    plt.plot(x.T, y.T, 'b')
    plt.title('Finite Element Distribution', fontsize = fs)
    plt.xlabel('X-coordinate', fontsize = fs)
    plt.ylabel('Y-coordinate', fontsize = fs)
    plt.grid()
    # Setting tick label font size
    plt.tick_params(axis='both', labelsize=fs)
    plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio
    plt.show()
    
    return

def plot_residuals(residuals, fs=16):
    """Plot the residuals from theolver."""
    plt.figure(figsize=(8,6))
    plt.semilogy(residuals, marker='o', linestyle='-')
    plt.title('Solver Convergence', fontsize=fs)
    plt.xlabel('Iteration Number', fontsize=fs)
    plt.ylabel('Residual (L2-Norm)', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()


# Norm calculation

def L2norm(xf_coords, yf_coords, xc_coords, yc_coords, phi_fine, phi_course):
    """
    xf_coords, yf_coords: x and y points for phi_fine
    xc_coords, yc_coords: x and y points for phi_course
    phi_fine: The output for the solved system using a fine mesh
    phi_course: The output for the solved system using a coarse mesh
    """

    # Reshape arrays to 2d
    phi_fine_mat = phi_fine.reshape((len(xf_coords),len(yf_coords)), order='C')
    phi_course_mat = phi_course.reshape((len(xc_coords),len(yc_coords)), order='C')
    
    # Generate mesh grid from fine grid points
    Xf, Yf = np.meshgrid(xf_coords, yf_coords)
    
    # Points for which we want the interpolated values
    points = np.vstack((Xf.ravel(), Yf.ravel())).T
    
    # Create interpolation function that allows for extrapolation
    interpolating_function = RegularGridInterpolator((xc_coords, yc_coords), phi_course_mat, bounds_error=False, fill_value=None)
    
    # Get interpolated values
    phi_course_interpolated = interpolating_function(points)
    
    # Reshape to the shape of your fine grid
    phi_course_interpolated = phi_course_interpolated.reshape(Xf.shape)
    
    error = np.sqrt(np.mean((phi_fine_mat - phi_course_interpolated)**2))
    
    return error

#%% SIMULATE

"""
Set variables here and run this cell if you want to test different problems.
"""

#Initialize variables
Tao = 20
hf = 10
T_west = 10
T_east = 100
T_ext = 300
width = 1
height = 1
nx = 80
ny = 80
r = 1.05 #keep within range of 1.0 - 1.05

dx, dy = width/nx, height/ny

# Boundary temperature dictionary
bound_info = {"north": [robin_bound_coeff, Tao, hf], #mixed
              "south": [insulated_bound, 0, 0],  #insulated
              "east": [fixed_temp_bound_coeff, Tao, 0], #fixed
              "west": [fixed_temp_bound_coeff, Tao, 0]  #fixed
              }

ext_temps = {"north": T_ext,
            "south": 0,
            "east": T_east,
            "west": T_west 
            }

t0 = time.time()

#create grid
grid, x_spacing, y_spacing, init_temp = create_grid(width, height, 
                                                    nx, ny, r) 
#create dependency matrix
dependency_mat, source_arr = create_dependency_matrix(grid, nx, ny, #Create Dependancy Mat
                                                      dx, dy, Tao, 
                                                      bound_info, ext_temps)
dependency_mat_thing = dependency_mat.reshape((nx*ny,ny*ny), order='C')
t1 = time.time()
print("dependency matrix setup time: {} s".format(round(t1-t0,2)))


t0 = time.time()
phi_out, residuals = Linear_Solve(dependency_mat, source_arr, init_temp, 
                            solver = 'bicgstab', use_preconditioner = True, 
                            precon_tol = 1e-4, solver_tol = 1e-5, rest = 50)
t1 = time.time()
print("Solver time: {} s".format(round(t1-t0,2)))

#Cell size
print("Min Cell Dim: {} \nMin Cell Area: {}\nMax Cell Dim: {}\nMax Cell Area: {}".format(
    [x_spacing.min(), y_spacing.min()], x_spacing.min()*y_spacing.min(),
    [x_spacing.max(),y_spacing.max()], x_spacing.max()*y_spacing.max()))


# Plotting

#heat plot
x_coords, y_coords = heatplot(x_spacing, y_spacing, width, height, nx, ny, phi_out)
#contour plot
contour_heatplot(x_spacing, y_spacing, width, height, nx, ny, phi_out)
#Convergence plot
plot_residuals(residuals)
#mesh plot
gridplot(x_spacing, y_spacing, width, height)


#%%
"""
Different resolution plots for estimating order of convergence with variable
inflation factor r
"""
r = 1.01
#------------------------------------------------------------------------------
#Fine
nx = 320
ny = 320

grid, x_spacing, y_spacing, init_temp = create_grid(width, height, 
                                                    nx, ny, r) #create grid

dependency_mat, source_arr = create_dependency_matrix(grid, nx, ny, #Create Dependancy Mat
                                                      dx, dy, Tao, 
                                                      bound_info, ext_temps)

phi_fine, residuals_fine = Linear_Solve(dependency_mat, source_arr, init_temp, 
                            solver = 'bicgstab', use_preconditioner = True, 
                            precon_tol = 1e-4, solver_tol = 1e-5)

xf_coords, yf_coords = contour_heatplot(x_spacing, y_spacing, width, height, nx, ny, phi_fine)
plot_residuals(residuals_fine)
# xf_coords, yf_coords = heatplot(x_spacing,y_spacing, width, height, nx, ny, phi_fine)
# gridplot(x_spacing, y_spacing, width, height)

#------------------------------------------------------------------------------
#Medium
nx = 160
ny = 160

grid, x_spacing, y_spacing, init_temp = create_grid(width, height, 
                                                    nx, ny, r) #create grid

dependency_mat, source_arr = create_dependency_matrix(grid, nx, ny, #Create Dependancy Mat
                                                      dx, dy, Tao, 
                                                      bound_info, ext_temps)

phi_med, residuals_med = Linear_Solve(dependency_mat, source_arr, init_temp, 
                            solver = 'bicgstab', use_preconditioner = True, 
                            precon_tol = 1e-4, solver_tol = 1e-5)
h_med = np.mean(x_spacing)

xm_coords, ym_coords = contour_heatplot(x_spacing, y_spacing, width, height, nx, ny, phi_med)

plot_residuals(residuals_med)
# xm_coords, ym_coords = heatplot(x_spacing,y_spacing, width, height, nx, ny, phi_med)
# gridplot(x_spacing, y_spacing, width, height)
#------------------------------------------------------------------------------
#Course
nx = 80
ny = 80

grid, x_spacing, y_spacing, init_temp = create_grid(width, height, 
                                                    nx, ny, r) #create grid

dependency_mat, source_arr = create_dependency_matrix(grid, nx, ny, #Create Dependancy Mat
                                                      dx, dy, Tao, 
                                                      bound_info, ext_temps)

phi_course, residuals_course = Linear_Solve(dependency_mat, source_arr, init_temp, 
                            solver = 'bicgstab', use_preconditioner = True, 
                            precon_tol = 1e-4, solver_tol = 1e-5)

h_course = np.mean(x_spacing)

xc_coords, yc_coords = contour_heatplot(x_spacing, y_spacing, width, height, nx, ny, phi_course)
plot_residuals(residuals_course)
# xc_coords, yc_coords = heatplot(x_spacing,y_spacing, width, height, nx, ny, phi_course)
# gridplot(x_spacing, y_spacing, width, height)
#%%
"""
Run to calculate error norms and order of convergence given results from the previous cell
"""
#error norms
medium_error = L2norm(xf_coords, yf_coords, xm_coords, ym_coords, phi_fine, phi_med)
course_error = L2norm(xf_coords, yf_coords, xc_coords, yc_coords, phi_fine, phi_course)

order_of_convergence = np.log(abs(course_error)/abs(medium_error))/(np.log(abs(h_course)/abs(h_med)))
print("medium error: {}".format(medium_error))
print("course error: {}".format(course_error))
print("Order of Convergence: ",order_of_convergence)

