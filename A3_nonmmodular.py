
"""
Created on Mon Nov 10 10:21:22 2023

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
# New for A3
# Calculate the x and y velocity fields using the node positions for a circular
# velocity pattern
def circular_vel(x_coord, y_coord, u, v):
    # Use meshgrid to create 2D arrays of x and y coordinates
    X, Y = np.meshgrid(x_coord, y_coord)
    
    # Calculate the radius and theta for each point in the grid
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    # Calculate the horizontal and vertical components of the velocity
    horizontal_vel = u * -r * np.sin(theta)
    vertical_vel = v * r * np.cos(theta)
    horizontal_vel = np.flip(horizontal_vel, axis = 0)
    
    # Return the 2D arrays for horizontal and vertical velocities
    return horizontal_vel, vertical_vel

#create constant velocity meshgrid with speed u
def constant_vel(x_coord, y_coord, u, v):
    horizontal_vel = u * np.ones((len(x_coord),len(y_coord)))
    vertical_vel = v * np.ones((len(x_coord),len(y_coord)))
    return horizontal_vel, vertical_vel

def get_density(x_coord, y_coord, p):
    return p * np.ones((len(x_coord),len(y_coord)))

def create_grid(width, height, nx, ny, r, velocity_func, u = 2, v = 2, den = 1, default_value = 10):
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
        
    #compensate for boundary spacing
    x_spacing[0], x_spacing[-1] = x_spacing[0]/2, x_spacing[-1]/2
    y_spacing[0], y_spacing[-1] = y_spacing[0]/2, y_spacing[-1]/2
    
    #calculate node position (centered at 0)
    x_coord = np.cumsum(x_spacing[:-1]) - 0.5*width
    y_coord = np.cumsum(y_spacing[:-1]) - 0.5*height
    
    # Call velocity calculation:
    x_vel, y_vel = velocity_func(x_coord, y_coord, u, v)
    
    # Call density calculation:
    density = get_density(x_coord,y_coord, den)
        
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
                
                #New for A3:
                "neighbors_pos": [], #get neighbor position in grid
                "neighbor_velocity": [], #get presecribed velocity
                "neighbor_density": [],
                "boundary_velocity": [],
                "boundary_density": [],
            }
            
            #add cell dictionary to the grid
            grid[i, j] = cell
    
    
    # Connect neighboring cells, add their spacing, and list cells with boundaries
    for i in range(nx):
        for j in range(ny):
            
            #If not at the north wall
            if i > 0:
                grid[i, j]["neighbors"].append([i-1,j]) #get neighbor index
                grid[i, j]["neighbors_pos"].append([x_coord[i-1],y_coord[j]]) #get neighbor position
                grid[i, j]["neighbor_spacing"].append(y_spacing[i]) #get distance to neighbor
                grid[i, j]["neighbor_surface_area"].append((x_spacing[i] + x_spacing[i+1])/2) #calculate cell area
                #new for A3
                # for neighbor velocity, only include the EITHER vertical or horizontal speed depending on orientation
                #northern velocity is negative
                grid[i, j]["neighbor_velocity"].append(-y_vel[i-1,j]) #calculate neighbor velocity
                grid[i, j]["neighbor_density"].append(density[i-1,j])
                
            else: #Add north boundary
                grid[i, j]["boundaries"].append('north') #set boundary orientation
                grid[i, j]["bound_spacing"].append(y_spacing[0]) #set distance to wall
                grid[i, j]["boundary_surface_area"].append((x_spacing[i] + x_spacing[i+1])/2) #calculate wall area
                grid[i, j]["boundary_velocity"].append(-y_vel[0,j])
                grid[i, j]["boundary_density"].append(density[0,j])
                
            #If not at the south wall
            if i < nx - 1:
                grid[i, j]["neighbors"].append([i + 1, j])
                grid[i, j]["neighbors_pos"].append([x_coord[i+1],y_coord[j]]) #get neighbor position
                grid[i, j]["neighbor_spacing"].append(y_spacing[i+1])
                grid[i, j]["neighbor_surface_area"].append((x_spacing[i] + x_spacing[i+1])/2)
                #southward velocity is positive
                grid[i, j]["neighbor_velocity"].append(y_vel[i+1,j])
                grid[i, j]["neighbor_density"].append(density[i+1,j])

            else:
                grid[i, j]["boundaries"].append('south') 
                grid[i, j]["bound_spacing"].append(y_spacing[-1])
                grid[i, j]["boundary_surface_area"].append((x_spacing[i] + x_spacing[i+1])/2)
                grid[i, j]["boundary_velocity"].append(y_vel[-1,j])
                grid[i, j]["boundary_density"].append(density[-1,j])

            #If not on the west wall    
            if j > 0:
                grid[i, j]["neighbors"].append([i, j - 1])
                grid[i, j]["neighbors_pos"].append([x_coord[i],y_coord[j-1]]) #get neighbor position
                grid[i, j]["neighbor_spacing"].append(x_spacing[j])
                grid[i, j]["neighbor_surface_area"].append((y_spacing[j] + y_spacing[j+1])/2)
                #westward velocity is positive
                grid[i, j]["neighbor_velocity"].append(x_vel[i,j-1])
                grid[i, j]["neighbor_density"].append(density[i,j-1])

            else:
                grid[i, j]["boundaries"].append('west')
                grid[i, j]["bound_spacing"].append(x_spacing[0])
                grid[i, j]["boundary_surface_area"].append((y_spacing[j] + y_spacing[j+1])/2)
                grid[i, j]["boundary_velocity"].append(x_vel[i,0])
                grid[i, j]["boundary_density"].append(density[i,0])

            #If not on the east wall    
            if j < ny - 1:
                grid[i, j]["neighbors"].append([i, j + 1])
                grid[i, j]["neighbors_pos"].append([x_coord[i],y_coord[j + 1]]) #get neighbor position
                grid[i, j]["neighbor_spacing"].append(x_spacing[j+1])
                grid[i, j]["neighbor_surface_area"].append((y_spacing[j] + y_spacing[j+1])/2)
                # eastwards velocity is negative
                grid[i, j]["neighbor_velocity"].append(-x_vel[i,j+1])
                grid[i, j]["neighbor_density"].append(density[i,j+1])

            else:
                grid[i, j]["boundaries"].append('east')
                grid[i, j]["bound_spacing"].append(x_spacing[-1])
                grid[i, j]["boundary_surface_area"].append((y_spacing[j] + y_spacing[j+1])/2)
                grid[i, j]["boundary_velocity"].append(-x_vel[i,-1])
                grid[i, j]["boundary_density"].append(density[i,-1])

    return grid, x_coord, y_coord, scalar_array, x_vel, y_vel

def diffusion_coeffs(Gamma, area, spacing):
    conduction_coeff = Gamma / spacing * area
    return conduction_coeff

# New for A3
def convection_coeffs(area, velocity, density):
    F = density * velocity * area
    return F

def fixed_temp_bound_coeff(transfer_coeff, Gamma, area, spacing):
    return Gamma * area / spacing

def robin_bound_coeff(transfer_coeff, Gamma, area, spacing):
    return area / (spacing / Gamma + 1 / transfer_coeff) #*d

def insulated_bound(transfer_coeff, Gamma, area, spacing):
    return 0

#New for A3
# ADVECTORS
def central_difference(velocity_coeff):
    return velocity_coeff/2

def upwind_first_order(velocity_coeff):
    return max(0,velocity_coeff)


def create_dependency_matrix(grid, nx, ny, dx, dy, permeability, bound_info, ext_temps, advector):
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
        
        advector:       Function describing advection scheme (Eg. Central Diff,
                                                              1st order upwind, etc.)
    
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
            neighbor_velocities = cell["neighbor_velocity"] #new for A3
            neighbor_densities = cell["neighbor_density"] #new for A3
            
            #Loop through Neighbors
            for n, neighbor_idx in enumerate(neighbors):
                
                #get neighbor parameters
                neighbor_spacing = neighbor_spacings[n]
                neighbor_area = neighbor_areas[n]
                neighbor_velocity = neighbor_velocities[n]
                neighbor_density = neighbor_densities[n]
                
                # Calculate 'D'
                diffusion_coefficient = diffusion_coeffs(permeability, 
                                                          neighbor_area, 
                                                          neighbor_spacing)
                # Calculate 'F'
                velocity_coefficient = convection_coeffs(neighbor_area,
                                                       neighbor_velocity,
                                                       neighbor_density)
                
                # Calculate dependency coefficient based on the passed advector function (central diff or upwind)
                dependency_coefficient = diffusion_coefficient + advector(velocity_coefficient)
                
                #Append dependency_coefficient
                rows.append(k)
                col_idx = neighbor_idx[0] * ny + neighbor_idx[1]
                cols.append(col_idx)
                coeffs.append(dependency_coefficient)

                # add to central coeff
                ap -= dependency_coefficient # add aw, ae, an, as
                ap -= velocity_coefficient # add Fe, -Fw, Fs, -Fn
                
            #Loop through boundaries if applicable
            boundaries = cell["boundaries"]
            
            if boundaries is not None:
                boundary_areas = cell["boundary_surface_area"]
                boundary_spacings = cell["bound_spacing"]
                boundary_velocities = cell["boundary_velocity"]
                boundary_densities = cell["boundary_density"]
                
                for z, boundary in enumerate(boundaries):

                    #get boundary area & spacing
                    boundary_area = boundary_areas[z]
                    boundary_spacing = boundary_spacings[z]
                    boundary_velocity = boundary_velocities[z]
                    boundary_density = boundary_densities[z]
                    
                    #Get Boundary Info
                    diffusion_boundary_func, Tao, transfer_coeff = bound_info[boundary]

                    #Calculate diffusive coefficient using diffusion_boundary_func
                    diffusion_bound_coeff = diffusion_boundary_func(transfer_coeff, 
                                                                    Tao, 
                                                                    boundary_area, 
                                                                    boundary_spacing)
                    
                    #Calculate velocity coefficient using convection function:
                    velocity_bound_coeff = convection_coeffs(boundary_area,
                                                             boundary_velocity,
                                                             boundary_density)

                    #calculate overall coeff using advector
                    bound_coeff = diffusion_bound_coeff + advector(velocity_bound_coeff)
                    # Add Sp to ap
                    ap -= bound_coeff
                    ap -= velocity_bound_coeff 
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
    # def callback(xk):
    #     r = np.linalg.norm(b - A @ xk)
    #     residuals.append(r)
    def callback(xk):
        r = np.linalg.norm(b - A.dot(xk))  # Use .dot() for compatibility with sparse matrices
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
    
def heatplot(x_coordinates, y_coordinates, nx, ny, phi):
    
    # Reshape to 2d array
    phi_out_mat = phi.reshape((nx,ny), order='C')
    
    # Create a meshgrid 
    x, y = np.meshgrid(x_coordinates, y_coordinates)

    #Pplot
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(x,y,np.flip(phi_out_mat,0),cmap = 'hot') #coolwarm
    plt.colorbar(pcm, label='Temperature')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Temperature Distribution')

    plt.show()
    return phi_out_mat


def contour_heatplot(x_coordinates, y_coordinates, nx, ny, phi, num_levels = 20):
    
    # Reshape to 2d array
    phi_out_mat = phi.reshape((nx,ny), order='C')
    
    # Create a meshgrid 
    x, y = np.meshgrid(x_coordinates, y_coordinates)

    # Plot
    fig, ax = plt.subplots()
    fs = 16
    
    # Using contourf for filled contours
    contour = ax.contourf(x, y, np.flip(phi_out_mat, 0), levels = num_levels, cmap='coolwarm') #coolwarm
    cbar = plt.colorbar(contour)
    cbar.set_label("Temperature", fontsize=fs)
    
    cbar.ax.tick_params(labelsize=fs)
    plt.tick_params(axis='both', labelsize=fs)
    
    ax.set_xlabel('X Coordinate', fontsize=fs)
    ax.set_ylabel('Y Coordinate', fontsize=fs)
    ax.set_title('Temperature Distribution Contour', fontsize=fs)

    plt.show()


def gridplot(x_coordinates, y_coordinates):
    fs = 16
    
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

def plot_velocity_field(x_coords, y_coords, x_vel, y_vel, subsample=1):
    """
    Plots a 2D velocity field.

    Parameters:
    - x_coords: 2D numpy array of x coordinates
    - y_coords: 2D numpy array of y coordinates
    - x_vel: 2D numpy array of x components of velocity
    - y_vel: 2D numpy array of y components of velocity
    - subsample: Subsampling rate for the points (default 1, no subsampling)
    """
    fs = 16
    
    if subsample > 1:
        # Subsampling the arrays
        x_coords = x_coords[subsample::subsample, subsample::subsample]
        y_coords = y_coords[subsample::subsample, subsample::subsample]
        x_vel = x_vel[subsample::subsample, subsample::subsample]
        y_vel = y_vel[subsample::subsample, subsample::subsample]

    plt.figure(figsize=(10, 8))
    plt.quiver(x_coords, y_coords, x_vel, y_vel, pivot='middle')
    plt.xlabel('X Coordinate', fontsize = fs)
    plt.ylabel('Y Coordinate', fontsize = fs)
    plt.title('2D Velocity Field', fontsize = fs)
    plt.show()
    
def plot_diagonal(fig, diagonal):
    x = np.linspace(-np.sqrt((width/2)**2 + (height/2)**2), np.sqrt((width/2)**2 + (height/2)**2), len(diagonal))
    fig.plot(x, diagonal)
    
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
def calculate_link_coeffs(uface, vface, P, rho, mu, ra, ar, dx, dy, nx, ny):
    """
    Calculate the link coefficients for the link matrix:
    1.  Calculate interior coefficients by looping through rows & columns, 
        getting face velocities, multiplying by density, and then calculating
        coefficients according to the equations (aswell as source terms).
    2.  Calculate wall coefficients through the same process, each seperately
    3.  Calcilate corner coefficients the same way.
    """
    
    # Initialize coefficients:
    Ap = np.zeros((ny, nx))
    Ae = np.zeros((ny, nx))
    Aw = np.zeros((ny, nx))
    An = np.zeros((ny, nx))
    As = np.zeros((ny, nx))
    # Source terms
    Sx = np.zeros((ny, nx))
    Sy = np.zeros((ny, nx))
    
    # interior cells
    # We begin and end indexing at the second and second last points
    for i in range(1,ny-1): # Iterate through rows. 
        for j in range(1, nx-1): # Iterate through columns.
            # Get face velocities and multiply by density
            ue = rho * uface[i,j]
            uw = rho * uface[i,j+1]
            un = rho * vface[i+1,j]
            us = rho * vface[i,j]
            
            # Create coefficients:
            Ae[i,j] = -0.5 * (np.abs(ue) - ue) * dy - mu*ra
            Aw[i,j] = -0.5 * (np.abs(uw) + uw) * dy - mu*ra
            An[i,j] = -0.5 * (np.abs(un) - un) * dx - mu*ar
            As[i,j] = -0.5 * (np.abs(us) + us) * dx - mu*ar
            # Central Coefficient is the sum of the others (but positive)
            Ap[i,j] =  (0.5 * (np.abs(ue) + ue) * dy + mu*ra +
                        0.5 * (np.abs(uw) - uw) * dy + mu*ra + 
                        0.5 * (np.abs(un) + un) * dx + mu*ar + 
                        0.5 * (np.abs(us) - us) * dx + mu*ar)
            
            Sx[i,j] = (P[i,j-1] - P[i,j+1]) / 2 * dy # (Pw - Pe)*dy/2
            Sy[i,j] = (P[i+1,j] - P[i-1,j]) / 2 * dx # (Ps - Pn)*dx/2
            
    # Boundary cells
    #-------------------------------------------------------------------------
    # Eastern Wall
    j = ny-1
    for i in range(1, nx-1): # Iterate through rows
        uw = rho * uface[i,j+1]
        un = rho * vface[i+1,j]
        us = rho * vface[i,j]

        Ae[i,j] = 0 #-2*mu*ra
        Aw[i,j] = -0.5 * (np.abs(uw) + uw) * dy - mu*ra * (4 / 3)
        An[i,j] = -0.5 * (np.abs(un) - un) * dx - mu*ar
        As[i,j] = -0.5 * (np.abs(us) + us) * dx - mu*ar

        Ap[i,j] =   (0.5 * (np.abs(uw) - uw) * dy + mu*ra * (4) + 
                      0.5 * (np.abs(un) + un) * dx + mu*ar + 
                      #2*mu*ra +
                      0.5 * (np.abs(us) - us) * dx + mu*ar)
        
        Sx[i,j] = (P[i,j-1] - P[i,j]) * dx / 2# Assume Pe ~ Pp by extrapolation
        Sy[i,j] = (P[i+1,j] - P[i-1,j]) / 2 * dy
        
    
    # Western Wall
    j = 0
    for i in range(1, nx-1): # Iterate through rows
        ue = rho * uface[i,j]
        un = rho * vface[i+1,j]
        us = rho * vface[i,j]

        Ae[i,j] = -0.5 * (np.abs(ue) - ue) * dy - mu*ra * (4 / 3)
        Aw[i,j] = 0 #-2*mu*ra
        An[i,j] = -0.5 * (np.abs(un) - un) * dx - mu*ar
        As[i,j] = -0.5 * (np.abs(us) + us) * dx - mu*ar

        Ap[i,j] =   (0.5 * (np.abs(ue) + ue) * dy + mu*ra * (4) + 
                      0.5 * (np.abs(un) + un) * dx + mu*ar + 
                      #2*mu*ra +
                      0.5 * (np.abs(us) - us) * dx + mu*ar)
        
        Sx[i,j] = (P[i,j] - P[i,j+1]) * dx / 2 # Assume Pw ~ Pp by extrapolation
        Sy[i,j] = (P[i+1,j] - P[i-1,j]) / 2 * dy
    # Northern Wall
    i = nx-1
    for j in range(1, ny-1): # Iterate through columns
        u_bound = rho * uu_t[j]
        ue = rho * uface[i,j]
        uw = rho * uface[i,j+1]
        us = rho * vface[i,j]
        

        Ae[i,j] = -0.5 * (np.abs(ue) - ue) * dy - mu*ra
        Aw[i,j] = -0.5 * (np.abs(uw) + uw) * dy - mu*ra
        An[i,j] = 0#-2*mu*ar
        As[i,j] = -0.5 * (np.abs(us) + us) * dx - mu*ar * (4 / 3)

        Ap[i,j] =   (0.5 * (np.abs(ue) + ue) * dy + mu*ra + 
                      0.5 * (np.abs(uw) - uw) * dy + mu*ra +
                      #2*mu*ar +
                      0.5 * (np.abs(us) - us) * dx + mu*ar * (4) +
                      2 * mu * u_bound * ra)
        
        Sy[i,j] = (P[i-1,j] - P[i,j]) * dx / 2 # Assume Pn ~ Pp by extrapolation
        Sx[i,j] = (P[i,j-1] - P[i,j+1]) / 2 * dy + 2 * mu * u_bound * ra
        
    # Southern Wall
    i = 0
    for j in range(1, ny-1): # Iterate through columns
        ue = rho * uface[i,j]
        uw = rho * uface[i,j+1]
        un = rho * vface[i+1,j]

        Ae[i,j] = -0.5 * (np.abs(ue) - ue) * dy - mu*ra
        Aw[i,j] = -0.5 * (np.abs(uw) + uw) * dy - mu*ra
        An[i,j] = -0.5 * (np.abs(un) - un) * dx - mu*ar * (4 / 3)
        As[i,j] = 0#-2*mu*ar

        Ap[i,j] =   (0.5 * (np.abs(ue) + ue) * dy + mu*ra + 
                      0.5 * (np.abs(uw) - uw) * dy + mu*ra +
                      # 2*mu*ar +
                      0.5 * (np.abs(un) + un) * dx + mu*ar * (4))
        Sy[i,j] = (P[i,j] - P[i+1,j]) * dx / 2 # Assume Ps ~ Pp by extrapolation
        Sx[i,j] = (P[i,j-1] - P[i,j+1]) / 2 * dy
        
    # Corner cells
    #-------------------------------------------------------------------------
    # South - West Corner
    i = 0
    j = 0
    ue = rho * uface[i,j]
    un = rho * vface[i+1,j]

    Ae[i,j] = -0.5 * (np.abs(ue) - ue) * dy - mu*ra * (4 / 3)
    Aw[i,j] = 0#- 2*mu*ra
    An[i,j] = -0.5 * (np.abs(un) - un) * dx - mu*ar * (4 / 3)
    As[i,j] = 0#- 2*mu*ar
    # Central Coeff
    Ap[i,j] =   (0.5 * (np.abs(ue) + ue) * dy + mu*ra * (4) + 
                 # 2*mu*ra +
                 # 2*mu*ar +
                 0.5 * (np.abs(un) + un) * dx + mu*ar * (4))
    # Pressure Sources
    Sy[i,j] = (P[i,j] - P[i+1,j]) * dx / 2
    Sx[i,j] = (P[i,j] - P[i,j+1]) * dy / 2
    
    # South - East
    i = 0
    j = ny-1
    uw = rho * uface[i,j+1]
    un = rho * vface[i+1,j]
    
    Ae[i,j] = 0#- 2*mu*ra
    Aw[i,j] = -0.5 * (np.abs(uw) + uw) * dy - mu*ra * (4 / 3)
    An[i,j] = -0.5 * (np.abs(un) - un) * dx - mu*ar * (4 / 3)
    As[i,j] = 0#- 2*mu*ar
    # Central Coeff
    Ap[i,j] =  (0.5 * (np.abs(uw) - uw) * dy + mu*ra * (4) + 
                # 2*mu*ra +
                # 2*mu*ar +
                0.5 * (np.abs(un) + un) * dx + mu*ar * (4))
    # Pressure Sources
    Sy[i,j] = (P[i,j] - P[i+1,j]) * dx / 2
    Sx[i,j] = (P[i,j-1] - P[i,j]) * dy / 2
    
    # North - East
    i = nx-1
    j = ny-1
    uw = rho * uface[i,j+1]
    us = rho * vface[i,j]
    u_bound = rho * uu_t[j]

    Ae[i,j] = 0#- 2*mu*ra
    Aw[i,j] = -0.5 * (np.abs(uw) + uw) * dy - mu*ra * (4 / 3)
    An[i,j] = 0#- 2*mu*ar
    As[i,j] = -0.5 * (np.abs(us) + us) * dx - mu*ar * (4 / 3)
    # Central Coeff
    Ap[i,j] =  (0.5 * (np.abs(uw) - uw) * dy + mu*ra * (4) +  
                # 2*mu*ra +
                # 2*mu*ar +
                0.5 * (np.abs(us) - us) * dx + mu*ar * (4) +
                2 * mu * u_bound * ra)
    # Pressure Sources
    Sy[i,j] = (P[i-1,j] - P[i,j]) * dx / 2
    Sx[i,j] = (P[i,j-1] - P[i,j]) * dy / 2 + 2 * mu * u_bound * ra
    
    # North - West
    i = nx-1
    j = 0
    ue = rho * uface[i,j]
    us = rho * vface[i,j]
    u_bound = rho * (uu_t[j])
    
    Ae[i,j] = -0.5 * (np.abs(ue) - ue) * dy - mu*ra * (4 / 3)
    Aw[i,j] = 0#- 2*mu*ra
    An[i,j] = 0#- 2*mu*ar
    As[i,j] = -0.5 * (np.abs(us) + us) * dx - mu*ar * (4 / 3)
    # Central Coeff
    Ap[i,j] =  (0.5 * (np.abs(ue) + ue) * dy + mu*ra * (4) + 
                # 2*mu*ra +
                # 2*mu*ar +
                0.5 * (np.abs(us) - us) * dx + mu*ar * (4) +
                2 * mu * u_bound * ra)
    # Pressure Sources
    Sy[i,j] = (P[i-1,j] - P[i,j]) * dx / 2
    Sx[i,j] = (P[i,j] - P[i,j+1]) * dy / 2 + 2 * mu * u_bound * ra
    #-------------------------------------------------------------------------
    # unravel
    Sx, Sy = np.ravel(Sx), np.ravel(Sy)
    
    return Ae, Aw, An, As, Ap, Sx, Sy

#%% SIMULATE

"""
Set variables here and run this cell if you want to test different problems.
"""

#Initialize variables
Gamma = 0.001
u = 1
v = 1
T_west = 100
T_east = 0
T_north = 100
T_south = 0
width = 1
height = 1
nx = 10
ny = 10
r = 1.0 #keep within range of 1.0 - 1.05

dx, dy = width/nx, height/ny

# Boundary temperature dictionary
bound_info = {"north": [fixed_temp_bound_coeff, Gamma, 0], #mixed
              "south": [fixed_temp_bound_coeff, Gamma, 0],  #fixed
              "east": [fixed_temp_bound_coeff, Gamma, 0], #fixed
              "west": [fixed_temp_bound_coeff, Gamma, 0]  #fixed
              }

ext_temps = {"north": T_north,
            "south": 0,
            "east": 0,
            "west": T_west 
            }

# CHOOSE ADVECTION SCHEME
advector = upwind_first_order
# advector = central_difference

# CHOOSE VELOCITY FUNCTION
# velocity_func = constant_vel
velocity_func = circular_vel

t0 = time.time()
#create grid
grid, x_coords, y_coords, init_temp, x_vel, y_vel = create_grid(width, height, 
                                                    nx, ny, r, velocity_func, u = u, v = v) 
#create dependency matrix
dependency_mat, source_arr = create_dependency_matrix(grid, nx, ny, #Create Dependancy Mat
                                                      dx, dy, Gamma, 
                                                      bound_info, ext_temps,
                                                      advector)

t1 = time.time()

print("dependency matrix setup time: {} s".format(round(t1-t0,2)))


t0 = time.time()
phi_out, residuals = Linear_Solve(dependency_mat, source_arr, init_temp, 
                            solver = 'lgmres', use_preconditioner = True, 
                            precon_tol = 1e-10, solver_tol = 1e-7, rest = 50)
t1 = time.time()
print("Solver time: {} s".format(round(t1-t0,2)))

# ____________________________________________________________________________
# Plotting

# Apply Rounding If Necessary for Plotting:
phi_out = np.around(phi_out, decimals=5)
phi_mat_out = phi_out.reshape((nx,ny), order='C')
phi_out_mat = np.flip(phi_mat_out, 0)

#heat plot
heatplot(x_coords, y_coords, nx, ny, phi_out)
#contour plot
contour_heatplot(x_coords, y_coords, nx, ny, phi_out, num_levels = 60)
#Convergence plot
plot_residuals(residuals)
# Create a meshgrid 
x, y = np.meshgrid(x_coords, y_coords)
# Plot the velocity field
if velocity_func == circular_vel:
    plot_velocity_field(x, y, -x_vel, y_vel, subsample=10)
else:
    plot_velocity_field(x, y, x_vel, y_vel, subsample=10)

#mesh plot
gridplot(x_coords, y_coords)


#%%
"""
Different resolution plots for estimating order of convergence
"""


#------------------------------------------------------------------------------
#Fine
nx = 320
ny = 320

grid, xf_coords, yf_coords, init_temp, x_vel, y_vel = create_grid(width, height, 
                                                    nx, ny, r, velocity_func, 
                                                    u = u, v = v) #create grid

dependency_mat, source_arr = create_dependency_matrix(grid, nx, ny, #Create Dependancy Mat
                                                      dx, dy, Gamma, 
                                                      bound_info, ext_temps,
                                                      advector)

phi_fine, residuals_fine = Linear_Solve(dependency_mat, source_arr, init_temp, 
                            solver = 'lgmres', use_preconditioner = True, 
                            precon_tol = 1e-10, solver_tol = 1e-10)

heatplot(xf_coords, yf_coords, nx, ny, phi_fine)
phi_out_mat_fine = phi_fine.reshape((nx,ny), order='C')
diagonal_f = np.diag(phi_out_mat_fine)
# plot_residuals(residuals_fine)


#------------------------------------------------------------------------------
#Medium
nx = 160
ny = 160

grid, xm_coords, ym_coords, init_temp, x_vel, y_vel = create_grid(width, height, 
                                                    nx, ny, r, velocity_func, 
                                                    u = u, v = v) #create grid

dependency_mat, source_arr = create_dependency_matrix(grid, nx, ny, #Create Dependancy Mat
                                                      dx, dy, Gamma, 
                                                      bound_info, ext_temps,
                                                      advector)

phi_med, residuals_med = Linear_Solve(dependency_mat, source_arr, init_temp, 
                            solver = 'lgmres', use_preconditioner = True, 
                            precon_tol = 1e-10, solver_tol = 1e-10)
h_med = width/nx

heatplot(xm_coords, ym_coords, nx, ny, phi_med)
phi_out_mat_med = phi_med.reshape((nx,ny), order='C')
diagonal_m = np.diag(phi_out_mat_med)
# plot_residuals(residuals_med)

#------------------------------------------------------------------------------
#Course
nx = 80
ny = 80

grid, xc_coords, yc_coords, init_temp, x_vel, y_vel = create_grid(width, height, 
                                                    nx, ny, r, velocity_func, 
                                                    u = u, v = v) #create grid

dependency_mat, source_arr = create_dependency_matrix(grid, nx, ny, #Create Dependancy Mat
                                                      dx, dy, Gamma, 
                                                      bound_info, ext_temps,
                                                      advector)

phi_course, residuals_course = Linear_Solve(dependency_mat, source_arr, init_temp, 
                            solver = 'lgmres', use_preconditioner = True, 
                            precon_tol = 1e-10, solver_tol = 1e-10)

h_course = width/nx
heatplot(xc_coords, yc_coords, nx, ny, phi_course)
phi_out_mat_course = phi_course.reshape((nx,ny), order='C')
diagonal_c = np.diag(phi_out_mat_course)
# plot_residuals(residuals_course)
#%%
#Plot false diffusion patterns.
fig, ax = plt.subplots(figsize = (10,10))
plot_diagonal(ax, diagonal_c)
plot_diagonal(ax, diagonal_m)
plot_diagonal(ax, diagonal_f)
plt.xlabel("position on diagonal", fontsize = 16)
plt.ylabel("Temperature", fontsize = 16)
plt.title("Temperature distribution across Diagonal", fontsize = 16)
ax.legend(labels = ["Course Mesh (10x10)","Medium Mesh (50x50)", "Fine Mesh (100x100)"], fontsize = 16)
plt.show()
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

