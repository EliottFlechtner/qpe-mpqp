# Subject 5 - Implement the Quantum Phase Estimation algorithm with MPQP

# Students

- Eliott FLECHTNER
- Maxime BARDOUIL
- Félix BERTHIER


**Difficulty:** 3/5

## Introduction

The Quantum Phase Estimation algorithm aims at estimating the phase associated
with the eigenvalue of a given unitary $U_a$, with respect to an eigenstate
$\ket{\psi_a}$. Since all eigenvalues of a unitary operator have a complex norm
of 1, knowing their phase is sufficient to completely determine them.

## Goal of the project

The goal of the project is to implement the Quantum Phase Estimation algorithm by
implementing functions that generate the pieces of the global circuit of the
algorithm, measure the state, and retrieve the solution.

## Resources

- Wikipedia: https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm
- The Bible, Section 5.2, page 221: https://profmcruz.wordpress.com/wp-content/uploads/2017/08/quantum-computation-and-quantum-information-nielsen-chuang.pdf

## Expected deliverables

- A function that prepares a QCircuit in the initial state
  $\ket{\phi_1} = \ket{+}^{\otimes n}\ket{\psi_a}$, supposing that we give you as
  input the circuit that prepares the eigenstate $\ket{\psi_a}$,
- A function that decompose an arbitrary Control-Unitary gate into a succession
  of usual gates (U_a, U(theta,phi,lambda), CNOT, CZ, SWAP, TOF, ...)
- A function that implements the successive control-Unitaries^(2^j) between the
  first and second register.
- An implementation of the Quantum Fourier Transform, especially its inverse (you
  can have a look at the example in MPQP documentation website)
- A function that gets the counts of the measurements of the algorithm's circuit
  and post process it to return the approximation of the phase.
- A general algorithm regrouping all the pieces together, taking as input the
  unitary $U_a$, its eigenstate $\ket{\psi_a}$ with the circuit that generates
  it, the number of qubit for the precision of the estimation, and that returns
  the approximation of the phase associated with the eigenvalue.
