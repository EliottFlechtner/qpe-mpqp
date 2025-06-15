from mpqp import QCircuit
from mpqp.gates import H
from mpqp import Barrier


def prepare_initial_state(n_precision_qubits, eigenstate_circuit):
    # Create an empty circuit with total number of qubits
    n_eigenstate_qubits = eigenstate_circuit.nb_qubits
    total_qubits = n_precision_qubits + n_eigenstate_qubits
    circ = QCircuit(nb_qubits=total_qubits, label="QPE Circuit")
    precisions = range(n_precision_qubits)
    circuits = range(n_precision_qubits, total_qubits)

    # Apply Hadamard gates to precision qubits (index 0 to n-1)
    for i in range(n_precision_qubits):
        circ.add(H(i))

    # Append the given eigenstate circuit offset to the "bottom" qubits
    circ.append(eigenstate_circuit, qubits_offset=n_precision_qubits)

    # Return circuit in state |+>^n, |eigenstate>
    return circ
