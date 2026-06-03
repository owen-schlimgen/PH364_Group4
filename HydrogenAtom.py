import numpy as np
import scipy.linalg

N = 500
PHI_CUTTOFF = 7
SPHERICAL_HARMONIC_CUTOFF = 16

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

def make_unique(arr):
  indicies = []
  for i,elem in enumerate(arr):
    if(all([(elem != arr[j]) for j in indicies])):
      indicies.append(i)
  return np.array(indicies)


# Solve the phi equation
phi_operator = PhiOperator()
phi_eigenvalues, phi_eigenvectors = scipy.linalg.eigh(phi_operator)
phi_indicies = np.argsort(phi_eigenvalues)
B = phi_eigenvalues[phi_indicies][:PHI_CUTTOFF]

# Solve the theta equation
theta_operators = [ThetaOperator(b) for b in B]
theta_eig = [scipy.linalg.eigh(top[0], b=top[1]) for top in theta_operators]
unsorted_AB = np.array([[[b,a] for a in theta_eig[i][0]] for i,b in enumerate(B)]).reshape((-1,2)).transpose()
theta_indicies = np.lexsort(np.round(unsorted_AB, 2))
A = unsorted_AB[1][theta_indicies][:SPHERICAL_HARMONIC_CUTOFF]
phi_indicies_for_A = phi_indicies[:PHI_CUTTOFF][theta_indicies // N]
B_sorted_for_A = phi_eigenvalues[phi_indicies_for_A][:SPHERICAL_HARMONIC_CUTOFF]

print(A)
print(B_sorted_for_A)

