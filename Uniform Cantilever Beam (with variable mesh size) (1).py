#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Only FEM
# FEM results for uniform cantilever beam with linearly varying mesh size

from dolfin import *
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from dolfin import plot

# Define the beam dimensions
L = 20.0
B = 0.5
H = 1

A0 = B*H
m = -0.01  ### Beam inclination slope

# Define the cross-sectional area function
def cross_section_area(x):
    # Define the cross-sectional area as a function of x
    return A0 + m * x  # Example linearly varying area

# Define the number of elements along the length
num_elements = 20

# Create the mesh
mesh = BoxMesh(Point(0, -B / 2, -H / 2), Point(L, B / 2, H / 2), num_elements, 1, 1)

# Modify the mesh vertices to match the cross-sectional area function
coordinates = mesh.coordinates()
for i, x in enumerate(coordinates[:, 0]):
    A = cross_section_area(x)
    coordinates[i, 1] *= sqrt(A / B)  # Scale the y-coordinate
    coordinates[i, 2] *= sqrt(A / H)  # Scale the z-coordinate
    
# mesh.write("MA_trial.vtk")


# from dolfin import *
import numpy as np

E, nu = Constant(1e5), Constant(0.)
rho = Constant(1e-3)

# Lame coefficient for constitutive relation
mu = E/2./(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)

def eps(v):
    return sym(grad(v))
def sigma(v):
    dim = v.geometric_dimension()
    return 2.0*mu*eps(v) + lmbda*tr(eps(v))*Identity(dim)

V = VectorFunctionSpace(mesh, 'Lagrange', degree=1)
u = TrialFunction(V)
du = TestFunction(V)


def left(x, on_boundary):
    return near(x[0],0.)

bc = DirichletBC(V, Constant((0.,0.,0.)), left)

k_form = inner(sigma(du),eps(u))*dx
l_form = Constant(1.)*u[0]*dx
K = PETScMatrix()
b = PETScVector()
assemble_system(k_form, l_form, bc, A_tensor=K, b_tensor=b)

m_form = rho*dot(du,u)*dx
M = PETScMatrix()
assemble(m_form, tensor=M)

eigensolver = SLEPcEigenSolver(K, M)
eigensolver.parameters['problem_type'] = 'gen_hermitian'
eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
eigensolver.parameters['spectral_shift'] = 0.

N_eig = 12   # number of eigenvalues
print("Computing {} first eigenvalues...".format(N_eig))
eigensolver.solve(N_eig)

# Exact solution computation
from scipy.optimize import root
from math import cos, cosh
falpha = lambda x: cos(x)*cosh(x)+1
alpha = lambda n: root(falpha, (2*n+1)*pi/2.)['x'][0]

# Set up file for exporting results
file_results = XDMFFile("MA_linear_area_change.xdmf")
# file_results = XDMFFile("MA_linear_area_change.vtk")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

eigenmodes = []
# Extraction
for i in range(N_eig):
    # Extract eigenpair
    r, c, rx, cx = eigensolver.get_eigenpair(i)

    # 3D eigenfrequency
    freq_3D = sqrt(r)/2/pi

    print("{:8.5f} Hz".format(freq_3D))

    # Initialize function and assign eigenvector
    eigenmode = Function(V,name="Eigenvector "+str(i))
    eigenmode.vector()[:] = rx

    eigenmodes.append(eigenmode)

# Write i-th eigenfunction to xdmf file
file_results.write(eigenmode, i)
    
# Load the mode shapes
mode_shapes = [Function(V, name=f"Eigenvector {i}") for i in range(len(eigenmodes))]
for mode, eigenmode in zip(mode_shapes, eigenmodes):
    mode.vector()[:] = eigenmode.vector()

# Plot mode shapes
# Plot mode shapes
# for i, mode in enumerate(mode_shapes):
#     plot(mode, mode="displacement", title=f"Mode {i+1}")
#     interactive()


# In[ ]:




