import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

N = 500
RADIUS_FOCUS = 5.292e-11

PHI_CUTTOFF = 25
SPHERICAL_HARMONIC_CUTOFF = 64
HYDROGEN_ORBITAL_CUTOFF = 100

HBAR = 1.0546e-34
K = 1/(4*np.pi * 8.8542e-12)
MU = 9.1094e-31
ECHARGE = 1.6022e-19

def PhiOperator():
  dphi = 2*np.pi / (N)
  dif = np.diag(np.ones(N)) - np.diag(np.ones(N-1), k=-1)
  corner_one = np.zeros((N,N)); corner_one[0][N-1] = 1
  periodic_dif = dif - corner_one
  phi_operator = (periodic_dif.T + periodic_dif) * (1/dphi**2)
  return phi_operator

def ThetaOperator(B):
  dtheta = np.pi / (N+1)
  main_diagonal = np.diag([np.sin((3/2) * dtheta), *[
      (np.sin((n+1/2) * dtheta) + np.sin((n-1/2) * dtheta))
    for n in range(2,N)], np.sin(((N) - 1/2) * dtheta)])
  top_off_diagonal = np.diag([
      np.sin((n+1/2) * dtheta)
    for n in range(1,N)], k=1)
  bottom_off_diagonal = np.diag([
      np.sin((n-1/2) * dtheta)
    for n in range(2,N+1)], k=-1)
  differential_operator = (main_diagonal - bottom_off_diagonal - top_off_diagonal) * (1/dtheta**2)

  B_term = np.diag([
      B / np.sin((n) * dtheta)
    for n in range(1,N+1)])
  theta_operator = differential_operator + B_term

  W = np.diag([
      np.sin((n) * dtheta)
    for n in range(1,N+1)])
  return (theta_operator, W)

def ROperator(A, Z=1):
  r0 = RADIUS_FOCUS
  drho = (np.pi/2) / (N+1)
  main_diagonal = np.diag([r0 * np.sin((3/2) * drho)**2, *[
      (r0 * np.sin((n+1/2) * drho)**2 + r0 * np.sin((n-1/2) * drho)**2)
    for n in range(2,N)], r0 * np.sin(((N) - 1/2) * drho)**2])
  top_off_diagonal = np.diag([
      r0 * np.sin((n+1/2) * drho)**2
    for n in range(1,N)], k=1)
  bottom_off_diagonal = np.diag([
      r0 * np.sin((n-1/2) * drho)**2
    for n in range(2,N+1)], k=-1)
  differential_operator = (main_diagonal - bottom_off_diagonal - top_off_diagonal) * (1/drho**2)

  potential_term = np.diag([
      -(2*MU*K*Z*ECHARGE**2/HBAR**2) * r0**2 * np.sin((n) * drho) / np.cos((n) * drho)**3
    for n in range(1,N+1)])
  A_term = np.diag([
      A * r0 / np.cos((n) * drho)**2
    for n in range(1,N+1)])
  r_operator = differential_operator + potential_term + A_term

  W = np.diag([
      (2 * MU / HBAR**2) * r0**3 * np.sin((n) * drho)**2 / np.cos((n) * drho)**4
    for n in range(1,N+1)])
  return (r_operator, W)


# Solve the phi equation
phi_operator = PhiOperator()
phi_eigenvalues, phi_eigenstates = scipy.linalg.eigh(phi_operator)
phi_eigenstates = phi_eigenstates.transpose()
phi_indicies = np.argsort(phi_eigenvalues)
B = phi_eigenvalues[phi_indicies][:PHI_CUTTOFF]

# Solve the theta equation
theta_operators = [ThetaOperator(b) for b in B]
theta_eig = [scipy.linalg.eigh(op[0], b=op[1]) for op in theta_operators]
theta_eigenstates = np.array([eig[1].transpose() for eig in theta_eig]).reshape((-1,N))
unsorted_AB = np.array([[[b,a] for a in theta_eig[i][0]] for i,b in enumerate(B)]).reshape((-1,2)).transpose()
theta_indicies = np.lexsort(np.round(unsorted_AB, 2))
A = unsorted_AB[1][theta_indicies][:SPHERICAL_HARMONIC_CUTOFF]
phi_indicies_for_A = phi_indicies[theta_indicies // N]
B_sorted_for_A = phi_eigenvalues[phi_indicies_for_A][:SPHERICAL_HARMONIC_CUTOFF]

# Solve the r equation
r_operators = [ROperator(a) for a in A]
r_eig = [scipy.linalg.eigh(op[0], b=op[1]) for op in r_operators]
r_eigenstates = np.array([eig[1].transpose() for eig in r_eig]).reshape((-1,N))
unsorted_EA = np.array([[[B_sorted_for_A[i], a, 1/ECHARGE*e] for e in r_eig[i][0]] for i,a in enumerate(A)]).reshape((-1,3)).transpose()
r_indicies = np.lexsort(np.round(unsorted_EA, 2))

# Gather the eigenvalues
E = unsorted_EA[2][r_indicies][:HYDROGEN_ORBITAL_CUTOFF]
theta_indicies_for_E = theta_indicies[r_indicies // N]
A_sorted_for_E = unsorted_AB[1][theta_indicies_for_E][:HYDROGEN_ORBITAL_CUTOFF]
phi_indicies_for_E = phi_indicies_for_A[r_indicies // N]
B_sorted_for_E = phi_eigenvalues[phi_indicies_for_E][:HYDROGEN_ORBITAL_CUTOFF]


def get_eigenstate_index(n, l, m):
  quantum_numbers = []
  for n_i in range(1,7):
    for l_i in range(0,n_i):
      for m_i in range(0,l_i+1):
        quantum_numbers.append((n_i,l_i,m_i))
        if m_i != 0:
          quantum_numbers.append((n_i,l_i,-m_i))
  idx = quantum_numbers.index((n,l,m))
  return idx

def plot_wavefunction(n, l, m):
  if l >= 7 or l >= n or abs(m) > l:
    print("Invalid quantum numbers")
    return (0,0,0)
  idx = get_eigenstate_index(n, l, m)
  r_values = [0, *(RADIUS_FOCUS * np.tan(np.pi/2 / (N+1) * np.arange(1,N+1))), np.inf]
  r_wavefunction = [0, *(r_eigenstates[r_indicies][idx]), 0]
  theta_values = [0, *(np.pi / (N+1) * np.arange(1,N+1)), np.pi]
  theta_wavefunction = [0, *(theta_eigenstates[theta_indicies_for_E][idx]), 0]
  # phi_values = 2*np.pi / (N) * np.arange(N)
  # phi_wavefunction = phi_eigenstates[phi_indicies_for_E[idx]]

  r0 = 5.292e-11
  x_plotvals = r0 * np.linspace(-20, 20, 100)
  z_plotvals = r0 * np.linspace(-20, 20, 100)
  wavefunction = np.zeros((len(z_plotvals), len(x_plotvals)))
  for i,x in enumerate(x_plotvals):
    for j,z in enumerate(z_plotvals):
      r = np.sqrt(x**2 + z**2)
      theta = np.arccos(z / r)
      closest_r = np.searchsorted(r_values, r)
      closest_theta = np.searchsorted(theta_values, theta)
      wavefunction_at_r = 1/(r_values[closest_r] - r_values[closest_r-1]) * ((r_values[closest_r] - r) * r_wavefunction[closest_r] + (r - r_values[closest_r-1]) * r_wavefunction[closest_r-1])
      wavefunction_at_theta = 1/(theta_values[closest_theta] - theta_values[closest_theta-1]) * ((theta_values[closest_theta] - theta) * theta_wavefunction[closest_theta] + (theta - theta_values[closest_theta-1]) * theta_wavefunction[closest_theta-1])
      wavefunction[j,i] = wavefunction_at_r * wavefunction_at_theta
  
  prob_density = np.abs(wavefunction)**2

  plt.figure(figsize=(7,6))
  plt.contourf(x_plotvals, z_plotvals, prob_density, levels=80, cmap="viridis")

  cbar = plt.colorbar()
  cbar.set_label(r"$|\psi|^2$", rotation=0, labelpad=20)
  plt.title(f"Hydrogen orbital (n={n}, l={l}, m={m})")
  plt.xlabel("x")
  plt.ylabel("z", rotation = 'horizontal')
  plt.axis("equal")
  plt.show()

  return (E[idx], A_sorted_for_E[idx], B_sorted_for_E[idx])

