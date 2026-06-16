# Setup Guide: IBM Qiskit Environment

## 1. Create a Python environment

Use Python 3.10 or newer.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2. Install dependencies

From the project folder:

```bash
pip install -r requirements_IBM.txt
```

The main packages are:

```text
qiskit
qiskit-aer
qiskit-ibm-runtime
qiskit-optimization
qiskit-algorithms
numpy
matplotlib
jupyterlab
ipykernel
```

## 3. Running on a local simulator

For local testing, use the notebook cells that run with either:

```python
from qiskit.primitives import StatevectorSampler
```

or:

```python
from qiskit_aer.primitives import SamplerV2 as AerSampler
```

## 4. IBM Quantum account setup

### 4.1 Get your IBM Quantum API key

1. Go to the IBM Quantum Platform.
2. Sign in.
3. Open your account dashboard.
4. Create or copy your API key.
5. Keep it private.

### 4.2 Save IBM Quantum credentials locally

Run this once in a trusted local environment:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    token="<YOUR_IBM_QUANTUM_API_KEY>",
    instance="<YOUR_INSTANCE_CRN_OR_NAME>",  # recommended if you have one
    set_as_default=True,
    overwrite=True,
)
```

If you do not know your instance yet, you can temporarily save only the token:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    token="<YOUR_IBM_QUANTUM_API_KEY>",
    set_as_default=True,
    overwrite=True,
)
```

Then verify:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
print(service.active_account())
print(service.instances())
```

### 4.3. Use IBM Quantum hardware

After credentials are saved, the notebook can load the service with:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
```

A typical backend selection pattern is:

```python
backend = service.least_busy(
    operational=True,
    simulator=False,
    min_num_qubits=num_qubits,
)
print(backend.name)
```

Before sending a job to hardware, transpile the circuit for the selected backend:

```python
from qiskit.transpiler import generate_preset_pass_manager

pm = generate_preset_pass_manager(
    optimization_level=1,
    backend=backend,
)

isa_circuit = pm.run(circuit)
```

Then run with Runtime Sampler:

```python
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler

sampler = RuntimeSampler(mode=backend)
job = sampler.run([isa_circuit], shots=1024)
result = job.result()
```

## 5. Hardware execution checklist

Before running on a real QPU:

- Confirm the number of qubits required by the QUBO.
- Confirm the selected backend has at least that many qubits.
- Run the same problem locally first.
- Start with a small number of shots, for example `shots=100` or `shots=256`.
- Increase shots only after the workflow is verified.
- Expect queue delays and noisy results.
- Do not submit large jobs repeatedly while debugging.