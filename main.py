from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler
import qiskit.quantum_info as qi

import numpy as np


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# # # Define a normalized complex state vector
# # # Example: |ψ> = (1/√2)|00> + (i/√2)|11>
# # bell_state_vector = [1/np.sqrt(2), 0, 0, 1j/np.sqrt(2)]



# # # Define the normalized amplitudes for qubit 0
# # q0_amplitudes = [1, 0]  # Qubit 0: |q0> = (1/sqrt(30))|0> + (2i/sqrt(30))|1>

# # Define the normalized amplitudes for qubit 0
# q0_amplitudes = [ (1 + 1j) / np.sqrt(2), (2 - 1j) / np.sqrt(5)]  # Qubit 0: |q0> = (1 + i)/sqrt(2) |0> + (2 - i)/sqrt(5) |1>


# # Normalize the amplitudes (if necessary)
# norm = np.sqrt(np.abs(q0_amplitudes[0])**2 + np.abs(q0_amplitudes[1])**2)
# q0_amplitudes = [amp / norm for amp in q0_amplitudes]

# print(f"Normalized amplitudes: {q0_amplitudes}")
# q1_amplitudes = [0, 1]  # Qubit 1: |q1> = (10/sqrt(104))|0> + (2i/sqrt(104))|1>

# # Tensor product to get the full system state
# full_state = np.kron(q0_amplitudes, q1_amplitudes)
# num_q = 15
n_sample=1000


@app.get("/")
async def root():

    # # Create circuit with 2 qubits and 2 classical bits
    # circuit = QuantumCircuit(num_q)

    # # # Create Bell state (Φ⁺)
    # circuit.h(0)      # Apply Hadamard gate to first qubit
    # circuit.cx(0, 1)  # CNOT with first qubit as control, second as target

    # # # Initialize the qubits to the desired state
    # # circuit.initialize(state_vector, [0, 1])


    # # Initialize the qubits to the desired state
    # # circuit.initialize(full_state, [0, 1])

    # circuit.initialize(q0_amplitudes, [0])
    # circuit.initialize(q1_amplitudes, [1])
    # # 85 3 10 0.5

    
    # # circuit.save_statevector()  # Save statevector at this point

    # # stv1 = qi.Statevector.from_instruction(circuit)
    # # print(stv1)
    # # latex = stv1.draw('latex_source')
    # # print(latex)


    # # Add measurements
    # circuit.measure_all()

    # # Execute the circuit using the new Sampler primitive
    # sampler = Sampler()
    # job = sampler.run(circuit, shots=n_sample)
    # result = job.result()
    # counts = result.quasi_dists[0]

    # message = ''
    # for state, probability in counts.items():
    #     # Convert binary number to bit string
    #     bit_string = format(state, f'0{num_q}b')
    #     count = int(probability * n_sample)
    #     message += f"|{bit_string}> : {count} times ({probability*100:.1f}%)\n"


    return {"message": 'ok'}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.post("/measure")
async def measure(payload: dict):
    # Create circuit with 2 qubits and 2 classical bits
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

    
    # Add measurements
    circuit.measure_all()

    # Execute the circuit using the new Sampler primitive
    sampler = Sampler()
    job = sampler.run(circuit, shots=n_sample)
    result = job.result()
    counts = result.quasi_dists[0]

    message = ''
    for state, probability in counts.items():
        # Convert binary number to bit string
        bit_string = format(state, f'0{num_q}b')
        count = int(probability * n_sample)
        message += f"|{bit_string}> : {count} times ({probability*100:.1f}%)\n"

    print('message', message)

    return {"message": "Data received", "payload": payload, message: message}
