#!/usr/bin/env python
# coding: utf-8

# In[1]:


### FEM vs Beam Theory
### Participation factors also calculated

from dolfin import *
import numpy as np

L, H, B = 20., 1, 2.

Nx = 100
Ny = int(H/L*Nx)+1
Nz = int(B/L*Nx)+1

mesh = BoxMesh(Point(0.,0.,0.),Point(L,H,B), Nx, Ny, Nz)

E, nu = Constant(1e5), Constant(0.)
rho = Constant(1e-3)

##### Lame coefficient for constitutive relation
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

N_eig = 2   # number of eigenvalues
print("Computing {} first eigenvalues...".format(N_eig))
eigensolver.solve(N_eig)

##### Exact solution computation
from scipy.optimize import root
from math import cos, cosh
falpha = lambda x: cos(x)*cosh(x)+1
alpha = lambda n: root(falpha, (2*n+1)*pi/2.)['x'][0]

##### Set up file for exporting results
file_results = XDMFFile("modal_analysis_uniformBeam.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

eigenmodes = []
##### Extraction
for i in range(N_eig):
    # Extract eigenpair
    r, c, rx, cx = eigensolver.get_eigenpair(i)

    # 3D eigenfrequency
    freq_3D = sqrt(r)/2/pi

    # Beam eigenfrequency
    if i % 2 == 0: # exact solution should correspond to weak axis bending
        I_bend = H*B**3/12.
    else:          #exact solution should correspond to strong axis bending
        I_bend = B*H**3/12.
    freq_beam = alpha(i/2)**2*sqrt(float(E)*I_bend/(float(rho)*B*H*L**4))/2/pi

    print("Solid FE: {:8.5f} [Hz]   Beam theory: {:8.5f} [Hz]".format(freq_3D, freq_beam))

    # Initialize function and assign eigenvector
    eigenmode = Function(V,name="Eigenvector "+str(i))
    eigenmode.vector()[:] = rx

    eigenmodes.append(eigenmode)
    
##### Contribution of each mode
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




