from mpqp import *
from mpqp.gates import *
from mpqp import Barrier
from collections import Counter
from numpy import pi as PI
from math import floor
from mpqp.measures import BasisMeasure
from mpqp.execution import run, IBMDevice


def prepare_intial_state(eigenstate_circuit, n_precision_qubits):
    """
    Prepare the initial state for the Quantum Phase Estimation (QPE) algorithm.
    This function creates a quantum circuit that initializes the precision qubits
    to the state |+> and appends the provided eigenstate circuit to the bottom qubits.
    The total number of qubits in the circuit is the sum of the precision qubits
    and the qubits in the eigenstate circuit.

    :param eigenstate_circuit: The quantum circuit representing the eigenstate.
    :param n_precision_qubits: The number of precision qubits to be used in the QPE.
    :return: A QCircuit object representing the initial state for QPE.
    """

    print(f"Preparing initial state for QPE with {n_precision_qubits} precision.")

    # Create an empty circuit with total number of qubits
    n, m = n_precision_qubits, eigenstate_circuit.nb_qubits
    circ = QCircuit(nb_qubits=n + m, label="QPE Circuit")

    # Append the eigenstate circuit to the bottom qubits (offset by n)
    circ.append(eigenstate_circuit, qubits_offset=n)
    circ.add(Barrier())

    # Initialize precision qubits to |+>
    for i in range(n):
        circ.add(H(i))
    circ.add(Barrier())

    # Return the prepared circuit
    print("Initial state prepared: |+>^n otimes |eigenstate>")
    return circ


def transform_to_controlled_gate(circ, gate, i, n, target):
    """
    Transform a single-target gate into its controlled version based on the precision qubit index.
    This function modifies the quantum circuit by adding the controlled version of the gate
    with the control qubit at position `i` and the target qubit at the specified
    target position (offset by `n`).

    :param circ: The quantum circuit to which the controlled gate will be added.
    :param gate: The single-target gate to be transformed into a controlled gate.
    :param i: The index of the precision qubit that acts as the control.
    :param n: The total number of precision qubits.
    :param target: The target qubit position (offset by `n`).
    :return: None
    """

    if isinstance(gate, CNOT):  # CNOT -> TOF
        control = gate.controls[0] + n
        circ.add(TOF([i, control], target))
    elif isinstance(gate, X):  # X -> CNOT
        circ.add(CNOT(i, target))
    elif isinstance(gate, Y):  # Y -> S + CNOT + S_dagger = CY
        circ.add(S(target))
        circ.add(CNOT(i, target))
        circ.add(S_dagger(target))
    elif isinstance(gate, Z):  # Z -> CZ
        circ.add(CZ(i, target))
    elif isinstance(gate, H):  # H -> CH
        circ.add(Rz(PI, target))
        circ.add(Rz(PI / 2, target))
        circ.add(CNOT(i, target))
        circ.add(Rz(-PI, target))
        circ.add(Rz(-PI / 2, target))
    else:
        raise ValueError("Unknown single-target gate in controlled unitary.")


def create_controlled_unitary(circ, unitary, n_precision_qubits):
    """
    Create controlled unitary operations for the Quantum Phase Estimation (QPE) algorithm.
    This function applies controlled unitary operations based on the provided unitary circuit
    and the number of precision qubits. The controlled unitary operations are applied
    in reverse order, starting from the highest precision qubit.
    The unitary circuit is transformed to its controlled version with the control qubit
    at the specified precision qubit position.

    :param circ: The quantum circuit to which the controlled unitary operations will be added.
    :param unitary: The quantum circuit representing the unitary operation to be controlled.
    :param n_precision_qubits: The number of precision qubits used in the QPE.
    :return: None
    """

    print("Applying controlled unitary operations.")

    # Loop through precision qubits in reverse order to start from bottom precision qubit
    n = n_precision_qubits
    reverse_precisions = range(n)
    for i in reverse_precisions[::-1]:  # i starting from n-1 down to 0
        print(f"Applying controlled unitary for precision qubit {i}")

        # Apply controlled unitary operator 2^(n-i-1) times (1, 2, 4, ...)
        iterations = 2 ** (n - i - 1)
        print(f"Applying controlled unitary {iterations} times.")
        for k in range(iterations):
            print(f"\t{iterations-k} remaining")

            # Transform the unitary circuit to its controlled version
            # with control qubit at position i
            for gate in unitary.gates:
                if len(gate.targets) == 1:
                    target = gate.targets[0] + n
                    transform_to_controlled_gate(circ, gate, i, n, target)
                else:
                    raise ValueError(
                        "Only single-target gates are supported in controlled unitary."
                    )
        circ.add(Barrier())
    print("Controlled unitary operations applied.")


def build_inverse_qft(n_qubits):
    """
    Build the inverse Quantum Fourier Transform (QFT) circuit.
    This function constructs the inverse QFT circuit by adding SWAP gates for qubit reordering
    and applying controlled rotations and Hadamard gates in reverse order.
    The inverse QFT is essential for the Quantum Phase Estimation (QPE) algorithm to extract
    the phase information from the quantum state.

    :param n_qubits: The number of qubits in the circuit.
    :return: A QCircuit object representing the inverse QFT circuit over n_qubits.
    """

    qftc = QCircuit(n_qubits, label="Inverse QFT Circuit")

    # Add SWAP gates for qubit reordering
    qftc.add([SWAP(i, n_qubits - 1 - i) for i in range(int(floor(n_qubits / 2)))])

    # Apply controlled rotations and Hadamard gates in reverse order
    j = n_qubits - 1
    while j >= 0:
        qftc.add([CRk(i + 1 - j, i, j).inverse() for i in range(j + 1, n_qubits)])
        qftc.add(H(j))
        j -= 1

    return qftc


def append_inverse_qft(circ, n_precision_qubits):
    """
    Append the inverse Quantum Fourier Transform (QFT) to the circuit.
    This function adds the inverse QFT gates to the circuit, which is essential for
    the Quantum Phase Estimation (QPE) algorithm to extract the phase information from the
    quantum state.

    :param circ: The quantum circuit to which the inverse QFT will be appended.
    :param n: The number of precision qubits used in the QPE.
    :return: None
    """

    print("Appending inverse Quantum Fourier Transform (QFT).")

    # Append the inverse QFT to the circuit on the precision qubits only (no offset)
    circ.append(build_inverse_qft(n_precision_qubits), qubits_offset=0)
    circ.add(Barrier())

    print("Inverse QFT appended to the circuit.")


def measure_precision_qubits(circ, n_precision_qubits):
    """
    Measure the precision qubits in the circuit.
    This function adds measurement operations to the precision qubits, allowing
    the Quantum Phase Estimation (QPE) algorithm to extract the phase information
    from the quantum state.

    :param circ: The quantum circuit to which the measurement operations will be added.
    :param n_precision_qubits: The number of precision qubits used in the QPE.
    :return: None
    """

    print("Measuring precision qubits.")

    # Measure each precision qubit and store results in classical bits
    circ.add(BasisMeasure(targets=list(range(n_precision_qubits)), shots=1024))
    circ.add(Barrier())

    print("Measurement added to precision qubits.")


def estimate_phase_from_counts(counts, n_precision_qubits):
    """
    Estimate the phase 'x' from the measurement results of the QPE circuit.

    Parameters:
    - counts: List of integers, each representing the measured value on the precision register (e.g., [3, 2, 3, ...])
    - n_precision_qubits: Number of qubits in the precision register

    Returns:
    - estimated_x: The most probable value of x = k / (2^n)
    - distribution: A dictionary mapping each possible x = k / (2^n) to its measured probability
    """
    total_shots = len(counts)
    max_val = 2**n_precision_qubits

    # Count occurrences of each measurement
    freq = Counter(counts)

    # Compute normalized distribution over x = k / 2^n
    distribution = {k / max_val: v / total_shots for k, v in freq.items()}

    # Get most probable outcome
    most_probable_k = max(freq, key=freq.get)
    estimated_x = most_probable_k / max_val

    return estimated_x, distribution


def QPE(eigenstate_circuit, unitary, n_precision_qubits):
    """
    Perform Quantum Phase Estimation (QPE) on the provided eigenstate circuit.
    This function prepares the initial state, applies controlled unitary operations,
    appends the inverse QFT, and measures the precision qubits to extract the phase information.

    :param eigenstate_circuit: The quantum circuit representing the eigenstate.
    :param unitary: The quantum circuit representing the unitary operation to be controlled.
    :param n_precision_qubits: The number of precision qubits used in the QPE.
    :return: A QCircuit object representing the complete QPE circuit.
    """

    print("Starting Quantum Phase Estimation (QPE).")

    # Prepare initial state for QPE
    circ = prepare_intial_state(eigenstate_circuit, n_precision_qubits)

    # Create controlled unitary operations
    create_controlled_unitary(circ, unitary, n_precision_qubits)

    # Append inverse QFT to the circuit
    append_inverse_qft(circ, n_precision_qubits)

    # Measure precision qubits
    measure_precision_qubits(circ, n_precision_qubits)

    # Pretty print the final circuit
    print("\nFinal QPE Circuit:")
    circ.pretty_print()

    # Simulate the circuit
    result = run(circ, IBMDevice.AER_SIMULATOR_STATEVECTOR)
    print("\nSimulation complete. Results:")
    print(result)

    # Estimate phase from counts
    print("\nEstimating phase from counts...")
    return estimate_phase_from_counts(result.counts, n_precision_qubits)


if __name__ == "__main__":

    def eigenstate_1():
        c = QCircuit(1)
        c.add(X(0))  # Prepare |1⟩
        return c

    def unitary_Z():
        c = QCircuit(1)
        c.add(Z(0))  # Z gate has eigenvalue -1 for |1⟩
        return c

    unitary = unitary_Z()
    eigenstate = eigenstate_1()
    estimated_phase = QPE(
        unitary=unitary, eigenstate_circuit=eigenstate, n_precision_qubits=3
    )

    print("\nEstimated Phase:")
    print(f"x = {estimated_phase[0]} with distribution: {estimated_phase[1]}")
