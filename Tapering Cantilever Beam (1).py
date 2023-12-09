#!/usr/bin/env python
# coding: utf-8

# In[1]:


import meshio
import numpy as np

# Read the .msh file
msh = meshio.read("Cantilever_taper_beam.msh")

# Find the index of the tetrahedral cell type
tetra_index = None
for i, cell_block in enumerate(msh.cells):
    if cell_block.type == "tetra":
        tetra_index = i
        break

# Extract the tetrahedral cell data
tetra_cells = msh.cells[tetra_index].data

# Create the cell block for tetrahedral cells
tetra_cell_block = (
    "tetra",
    tetra_cells,
)

# Create a new Mesh object with the desired cells
mesh = meshio.Mesh(points=msh.points, cells=[tetra_cell_block])

# Write the mesh to XDMF format
meshio.write("Cantilever_taper_beam.xdmf", mesh)

# Getting the xdmf file in the field
from dolfin import *
mesh = Mesh()
with XDMFFile("Cantilever_taper_beam.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)

####################################################################

### Starting Modal analysis ###
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

# Standard FunctionSpace is defined 
V = VectorFunctionSpace(mesh, 'Lagrange', degree=1)
u = TrialFunction(V)
du = TestFunction(V)

# Boundary conditions correspond to a fully clamped support at x=0
def left(x, on_boundary):
    return near(x[0],0.)

bc = DirichletBC(V, Constant((0.,0.,0.)), left)

# The system stiffness matrix [K] and mass matrix [M]
# are respectively obtained from assembling the corresponding variational forms
k_form = inner(sigma(du),eps(u))*dx
l_form = Constant(1.)*u[0]*dx
K = PETScMatrix()
b = PETScVector()
assemble_system(k_form, l_form, bc, A_tensor=K, b_tensor=b)

m_form = rho*dot(du,u)*dx
M = PETScMatrix()
assemble(m_form, tensor=M)

# Eigen-solver
# Various parameters of the instance 'eigensolver' are then set to control the solver
eigensolver = SLEPcEigenSolver(K, M)
eigensolver.parameters['problem_type'] = 'gen_hermitian'
eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
eigensolver.parameters['spectral_shift'] = 0.

# Number of eigenvalues
N_eig = 6   
print("Computing {} first eigenvalues...".format(N_eig))
eigensolver.solve(N_eig)

# Set up file for exporting results
file_results = XDMFFile("Ganpati_Bappa_Morya.xdmf")
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
    
# Contribution of each mode
u = Function(V, name="Unit displacement")
u.interpolate(Constant((0, 1, 0)))
combined_mass = 0
for i, xi in enumerate(eigenmodes):
    qi = assemble(action(action(m_form, u), xi))
    mi = assemble(action(action(m_form, xi), xi))
    meff_i = (qi / mi) ** 2
    total_mass = assemble(rho * dx(domain=mesh))

    print("-" * 50)
    print("Mode {}:".format(i + 1))
    print("  Modal participation factor: {:.2e}".format(qi))
    print("  Modal mass: {:.4f}".format(mi))
    print("  Effective mass: {:.2e}".format(meff_i))
    print("  Relative contribution: {:.2f} %".format(100 * meff_i / total_mass))

    combined_mass += meff_i

print(
    "\nTotal relative mass of the first {} modes: {:.2f} %".format(
        N_eig, 100 * combined_mass / total_mass
    )
)


# In[ ]:




