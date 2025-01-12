from typing import Optional
from fastapi import FastAPI

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler

app = FastAPI()


@app.get("/")
async def root():

    # Create circuit with 2 qubits and 2 classical bits
    circuit = QuantumCircuit(2)

    # Create Bell state (Φ⁺)
    circuit.h(0)      # Apply Hadamard gate to first qubit
    circuit.cx(0, 1)  # CNOT with first qubit as control, second as target

    # Add measurements
    circuit.measure_all()

    # Execute the circuit using the new Sampler primitive
    sampler = Sampler()
    job = sampler.run(circuit, shots=1000)
    result = job.result()
    counts = result.quasi_dists[0]

    result = ''
    for state, probability in counts.items():
        # Convert binary number to bit string
        bit_string = format(state, '02b')
        count = int(probability * 1000)
        result += f"|{bit_string}> : {count} times ({probability*100:.1f}%)\n"

    return {"message": result}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}