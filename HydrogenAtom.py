import numpy as np
import scipy.linalg

N = 500
RADIUS_FOCUS = 3e-11

PHI_CUTTOFF = 25
SPHERICAL_HARMONIC_CUTOFF = 64
HYDROGEN_ORBITAL_CUTOFF = 100

HBAR = 1.0546e-34
K = 1/(4*np.pi * 8.8542e-12)
MU = 9.1094e-31
ECHARGE = 1.6022e-19

def PhiOperator():
  dphi = 2*np.pi/(N)
  dif = np.diag(np.ones(N)) - np.diag(np.ones(N-1), k=-1)
  corner_one = np.zeros((N,N)); corner_one[0][N-1] = 1
  periodic_dif = dif - corner_one
  phi_operator = (periodic_dif.T + periodic_dif) * (1/dphi**2)
  return phi_operator

def ThetaOperator(B):
  dtheta = np.pi / (N+1)
  main_diagonal = np.diag([np.sin((1/2) * dtheta), *[
      (np.sin((n+1/2) * dtheta) + np.sin((n-1/2) * dtheta))
    for n in range(1,N-1)], np.sin(((N-1) - 1/2) * dtheta)])
  top_off_diagonal = np.diag([
      np.sin((n+1/2) * dtheta)
    for n in range(0,N-1)], k=1)
  bottom_off_diagonal = np.diag([
      np.sin((n-1/2) * dtheta)
    for n in range(1,N)], k=-1)
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
  main_diagonal = np.diag([r0 * np.sin((1/2) * drho)**2, *[
      (r0 * np.sin((n+1/2) * drho)**2 + r0 * np.sin((n-1/2) * drho)**2)
    for n in range(1,N-1)], r0 * np.sin(((N-1) - 1/2) * drho)**2])
  top_off_diagonal = np.diag([
      r0 * np.sin((n+1/2) * drho)**2
    for n in range(0,N-1)], k=1)
  bottom_off_diagonal = np.diag([
      r0 * np.sin((n-1/2) * drho)**2
    for n in range(1,N)], k=-1)
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
phi_eigenvalues, phi_eigenvectors = scipy.linalg.eigh(phi_operator)
phi_indicies = np.argsort(phi_eigenvalues)
B = phi_eigenvalues[phi_indicies][:PHI_CUTTOFF]

# Solve the theta equation
theta_operators = [ThetaOperator(b) for b in B]
theta_eig = [scipy.linalg.eigh(op[0], b=op[1]) for op in theta_operators]
unsorted_AB = np.array([[[b,a] for a in theta_eig[i][0]] for i,b in enumerate(B)]).reshape((-1,2)).transpose()
theta_indicies = np.lexsort(np.round(unsorted_AB, 2))
A = unsorted_AB[1][theta_indicies][:SPHERICAL_HARMONIC_CUTOFF]
phi_indicies_for_A = phi_indicies[theta_indicies // N]
B_sorted_for_A = phi_eigenvalues[phi_indicies_for_A][:SPHERICAL_HARMONIC_CUTOFF]

# Solve the r equation
r_operators = [ROperator(a) for a in A]
r_eig = [scipy.linalg.eigh(op[0], b=op[1]) for op in r_operators]
unsorted_EA = np.array([[[B_sorted_for_A[i], a, 1/ECHARGE*e] for e in r_eig[i][0]] for i,a in enumerate(A)]).reshape((-1,3)).transpose()
r_indicies = np.lexsort(np.round(unsorted_EA, 2))

# Gather the eigenvalues
E = unsorted_EA[2][r_indicies][:HYDROGEN_ORBITAL_CUTOFF]
theta_indicies_for_E = theta_indicies[r_indicies // N]
A_sorted_for_E = unsorted_AB[1][theta_indicies_for_E][:HYDROGEN_ORBITAL_CUTOFF]
phi_indicies_for_E = phi_indicies_for_A[r_indicies // N]
B_sorted_for_E = phi_eigenvalues[phi_indicies_for_E][:HYDROGEN_ORBITAL_CUTOFF]

