# Benchmarking different solvers for Traveling Salesman Problem (TSP) QUBO Instances

In this project, we compare the solution quality and runtime for Traveling Salesman Problem (TSP) instances encoded as QUBO problems. We test whether warm-start QAOA can improve practical performance compared with standard QAOA when both are run through IBM Qiskit/IBM Quantum workflows.

## Goal

We benchmark three solver approaches on TSP instances with increasing numbers of cities:

1. A classical baseline solver
2. Standard QAOA using IBM Qiskit
3. Warm-start QAOA using IBM Qiskit

For each value of `n` cities, we compare:

- solution accuracy / quality
- time to produce a solution
- feasibility of the returned route
- scalability as the number of cities increases

The goal is not only to report whether a route was found, but also whether the returned route is valid, how close it is to the best known or classical solution, and how much runtime is required.

## Problem Encoding

The TSP is encoded as a QUBO using binary variables:

```text
x[i, t] = 1 if city i is visited at time step t
```

For `n` cities, this encoding requires:

```text
n^2 binary variables
```

This is important because each binary variable maps to a qubit in the QAOA formulation before transpilation and hardware overhead. As a result:

- `3` cities require `9` binary variables
- `4` cities require `16` binary variables
- `5` cities require `25` binary variables
- `15` cities require `225` binary variables

This `n^2` scaling is a major limitation for near-term quantum hardware. Even moderate TSP sizes can become too large for current IBM devices, especially after circuit depth, routing, noise, and transpilation constraints are considered.

## Benchmark Cases

### Classical Solver

The classical solver acts as the reference point. It gives a baseline for route quality and runtime. Depending on available tooling, this may be an exact brute-force method for very small `n`, a mixed-integer/quadratic optimizer, or another classical heuristic.

The classical result is used to estimate the best known solution for each test instance.

### Standard QAOA on IBM

Standard QAOA starts from the usual uniform quantum state and uses a classical optimizer to tune QAOA parameters. For each QUBO instance, it attempts to sample low-energy bitstrings that correspond to short TSP routes.

This gives the baseline quantum approach.

### Warm-Start QAOA on IBM

Warm-start QAOA initializes the quantum algorithm using information from a classical relaxation or approximate classical solution. A better initial state may guide QAOA toward higher-quality solutions with fewer optimization steps or better sampling behavior.

This is the main focus of the project. The benchmark tests whether warm-start QAOA provides measurable improvement over standard QAOA in:

- solution quality
- probability of returning a valid TSP route
- runtime or optimizer convergence time
- robustness across different random TSP instances

## Metrics

The metrics considered are:

- `n`: number of cities
- number of QUBO variables, equal to `n^2`
- solver type
- best route distance found
- relative error compared with the classical baseline
- whether the solution satisfies all TSP constraints
- wall-clock runtime
- number of optimizer iterations
- number of circuit shots
- backend used, such as local simulator or IBM hardware

For quantum runs, it is also useful to record the best measured bitstring, its QUBO energy, and how frequently it appears in the sampled distribution.

## Limitations

The main limitation is scaling. The TSP-to-QUBO encoding grows quadratically in variables, so the qubit requirement grows as `n^2`. This makes large values such as `n = 15` unrealistic for direct execution on current IBM quantum hardware because that would require `225` logical binary variables before considering hardware connectivity and transpilation overhead.

Other limitations include:

- IBM hardware noise can reduce the probability of measuring valid TSP routes.
- QAOA performance depends strongly on optimizer settings, QAOA depth, shot count, and random seed.
- Standard QAOA and warm-start QAOA may need many repeated trials for a fair comparison.
- Runtime measurements can vary due to IBM queue time, backend availability, and calibration state.
- A small benchmark may not prove general advantage; it can only provide evidence for the tested instances.
- Warm-start QAOA may improve initialization but does not remove the underlying `n^2` QUBO scaling.

## Assumptions

This project assumes:

- TSP instances are small enough to encode and run in Qiskit.
- The same distance matrices are used across classical, standard QAOA, and warm-start QAOA tests.
- Accuracy is measured relative to a classical baseline or best known route.
- Runtime is separated into meaningful categories where possible, such as problem construction, classical optimization, circuit execution, and post-processing.
- IBM hardware runs are limited to small `n` because of qubit count, circuit depth, and queue constraints.
- Simulator results and hardware results are reported separately.

## Novelty and Expected Contribution

Warm-start QAOA itself is not new, and QUBO formulations of TSP are also well known. The expected contribution of this project is therefore not a new algorithmic invention. Instead, the project is an implementation and benchmarking study focused on whether warm-start QAOA provides practical benefits for small TSP QUBO instances in an IBM Qiskit workflow.
