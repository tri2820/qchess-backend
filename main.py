from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator, QasmSimulator
from qiskit.primitives import Sampler
import qiskit.quantum_info as qi
from qiskit.quantum_info import entropy
from qiskit import transpile
import numpy as np
import json

from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": 'ok'}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


def check_entanglement(circuit: QuantumCircuit, qubit1: int = 0, qubit2: int = 1):
    """
    Check entanglement between two qubits in a quantum circuit.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit to analyze
        qubit1 (int): First qubit index
        qubit2 (int): Second qubit index
        
    Returns:
        dict: Dictionary containing different entanglement measures
    """
    # Create simulator and get statevector
    simulator = AerSimulator()
    transpiled_circuit = transpile(circuit, simulator)
    
    # Get the state vector
    state = qi.Statevector.from_instruction(transpiled_circuit)
    
    # Convert to density matrix
    rho = qi.DensityMatrix(state)
    
    # Get the reduced density matrix for the two qubits of interest
    other_qubits = [i for i in range(circuit.num_qubits) if i not in [qubit1, qubit2]]
    reduced_rho = qi.partial_trace(rho, other_qubits)
    
    # Calculate reduced density matrices for individual qubits
    rho_1 = qi.partial_trace(reduced_rho, [1])
    rho_2 = qi.partial_trace(reduced_rho, [0])
    
    # Calculate von Neumann entropy
    entropy_1 = qi.entropy(rho_1)
    entropy_2 = qi.entropy(rho_2)
    entropy_12 = qi.entropy(reduced_rho)
    
    # Calculate mutual information
    mutual_info = entropy_1 + entropy_2 - entropy_12
    
    # Calculate concurrence for 2-qubit state
    if reduced_rho.dim == (4, 4):  # Only for 2-qubit states
        try:
            concurrence = qi.concurrence(reduced_rho)
        except:
            concurrence = None
    else:
        concurrence = None
    
    # Check if the state is entangled
    is_entangled = mutual_info > 1e-10
    return is_entangled
    
@app.post("/measure")
async def measure(payload: dict):
    # Create circuit with n qubits and n classical bits
    num_q = len(payload['qubits'])
    circuit = QuantumCircuit(num_q)

    for i, qubit in enumerate(payload['qubits']):
        amp = [1, 0] if qubit['classicalState'] == 0 else [ 0, 1]
        circuit.initialize(amp, [i])
    for action in payload['actions']:
        qubit_indexes = [
            (
                next(i for i, qubit in enumerate(payload['qubits']) if qubit['id'] == arg)
            )
             for arg in action['args']
        ]
        circuit.__getattribute__(action['gate'])(*qubit_indexes)


    
    state = qi.Statevector.from_instruction(circuit)
    entanglement = None
    last_action = payload['actions'][-1]
    if last_action and last_action['gate'] == 'cx':
        indicies_of_cx_qubits = [next(i for i, qubit in enumerate(payload['qubits']) if qubit['id'] == arg) for arg in last_action['args']] 
        entanglement = check_entanglement(circuit, indicies_of_cx_qubits[0], indicies_of_cx_qubits[1])
        entanglement = bool(entanglement)
        print("entanglement", entanglement)

    latex = state.draw('latex_source')

    # Add measurements
    circuit.measure_all()

    # For execution
    simulator = AerSimulator()
    compiled_circuit = transpile(circuit, simulator)
    sim_result = simulator.run(compiled_circuit).result()
    
    # Get the counts and calculate probabilities
    counts = sim_result.get_counts()
    total_shots = sum(counts.values())
    probabilities = {state: count / total_shots for state, count in counts.items()}

    # Find the most frequent measurement
    measurement = max(counts, key=counts.get)

    result = {
        "entanglement": entanglement,
        "circuit": str(circuit),
        "measurement": measurement,
        "probabilities": probabilities,
        "qubits": payload['qubits'],
        # state: state,
        "latex": latex
    }
    

    return result
