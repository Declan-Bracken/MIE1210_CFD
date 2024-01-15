#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:30:29 2023

@author: declanbracken
"""

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, gmres, lgmres, cg, bicgstab, cgs, spilu
import numpy as np
import time
from matplotlib import pyplot as plt
#%%

def Linear_Solve(A, b,  solver = 'gmres', use_preconditioner = True, precon_tol = 1e-4, solver_tol = 1e-5, rest = 50):
    
    #List of solver methods
    solver_list = ['gmres', 'lgmres','cg', 'bicgstab', 'cgs']
    
    #raise error if unspecified solver is used
    if solver not in (solver_list):
        raise ValueError("Invalid solver. Supported options are 'gmres', 'lgmres', 'cg', 'bicg', or 'cgs'.")
    
    
    #convert A to csr format if not already
    if A.getformat() != 'csr':
        A = csr_matrix(A)
    
    #Construct preconditioner seperately from solving
    if use_preconditioner:
        #apply incomplete LU decomposition to approximate the inverse of A, with optional tolerance
        preconditioned_A = spilu(A, drop_tol = precon_tol)
        M_x = LinearOperator(A.shape, preconditioned_A.solve)
        
    else:
        M_x = None
    
    
    if solver == 'lgmres':
        x, info = lgmres(A, b, M=M_x, atol=solver_tol)
    elif solver == 'cg':
        x, info = cg(A, b, M=M_x,atol=solver_tol)
    elif solver == 'bicgstab':
        x, info = bicgstab(A, b, M=M_x,atol=solver_tol)
    elif solver == 'cgs':
        x, info = cgs(A, b, M=M_x,atol=solver_tol)
    else:
        x, info = gmres(A, b, M=M_x,atol=solver_tol, restart = rest) #atol=solver_tol
     
    #check for successful convergeance
    if info != 0:
        print("Warning: {} did not converge (info={}).".format(solver,info))
    
    return x
#%%
"""
Solve using a tridiagonal matrix of dimension N. Change "solver_method" to alternate solvers
"""

N = 10**6

# Create a sparse matrix with diagonals of 2's and sub-diagonals and super-diagonals of -1's
diagonals = [2 * np.ones(N), -1 * np.ones(N-1), -1 * np.ones(N-1)]
A = diags(diagonals, [0, -1, 1], shape=(N, N), format='csr')

# Create a random right-hand side vector
b = np.random.rand(N)

solver_method = 'bicgstab'

# Solve the linear system
t0 = time.time()
x = Linear_Solve(A, b, solver= solver_method, use_preconditioner=True)
t1 = time.time()


r = A*x - b
# Calculate the norm of the residual for error assessment
residual_norm = np.linalg.norm(r)

print("Solver {} took {} seconds to run with residual norm {}.".format(solver_method,t1-t0,residual_norm))

#%%
"""
Code to test different solver methods on different matrix sizes. The list N
denotes matrix size, while the list solver_methods is used to loop through 
solving types. precondition variable is a boolean to test with or without
preconditiong.

Note: If running the code without preconditiong, gmres will take minutes if
not hours to converge above N = 2,000.

"""

N = np.array([1e3, 1e4, 2e4, 1e5, 2e5, 1e6],dtype=int)
# N = np.array([100, 200, 500, 1000, 2000, 5000, 10000],dtype=int)

solver_methods = ['gmres','lgmres','cg', 'bicgstab','cgs'] 
# solver_methods = ['lgmres','cg', 'bicgstab','cgs'] 

precondition = False

#initialize array to store average time values
times = np.zeros([len(solver_methods),len(N)])

#loop to average
for n in range(10):
    print(n)
    times0 = np.zeros([len(solver_methods),len(N)])
    #loop through matrix size
    for i in range(len(N)):
        
        # Create a sparse matrix with diagonals of 2's and sub-diagonals and super-diagonals of -1's
        diagonals = [2 * np.ones(N[i]), -1 * np.ones(N[i]-1), -1 * np.ones(N[i]-1)]
        A = diags(diagonals, [0, -1, 1], shape=(N[i], N[i]), format='csr')
        # Create a random right-hand side vector
        b = np.random.rand(N[i])
        
        #loop through methods
        for j in range(len(solver_methods)):
        
            t0 = time.time()
            x = Linear_Solve(A, b, solver= solver_methods[j], use_preconditioner=precondition)
            t1 = time.time()
            
            times0[j,i] = t1-t0
    times += times0
times /= (n+1)
#%% Plotting

fig, axs = plt.subplots(1,1)

axs.plot(N,times.T)

axs.set_xlabel("Matrix size N")
axs.set_ylabel("Single trial solving time (s)")
axs.set_title("Average Solver Convergence Time")
axs.set_xscale('log')
axs.set_yscale('log')
axs.legend([f'{i}' for i in solver_methods])
fig.set_dpi(400)
fig.savefig('/Users/declanbracken/Documents/U of T/MIE 1210/nonpreconditioned_time1.png', dpi=400)

