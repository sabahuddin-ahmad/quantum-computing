# To build pdf from markdown, open terminal and run this script from the root of the repository
# Activate the environment: conda activate ./.envs/pandoc-md-pdf
# Generate PDF: ./build_pdf.sh

#!/usr/bin/env bash
set -e

pandoc docs/qubo_to_qaoa_qiskit/qubo_to_warm_qaoa_qiskit_guide.md \
  -o docs/qubo_to_qaoa_qiskit/qubo_to_warm_qaoa_qiskit_guide.pdf \
  --pdf-engine=tectonic \
  -V geometry:margin=0.7in \
  -V fontsize=11pt