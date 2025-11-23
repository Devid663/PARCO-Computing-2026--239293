SpMV operation using CSR matrix format with OpenMO
# Setup
SSH Client: Use MobaXterm for Windows. For macOS and Linux, the built-in SSH client is sufficient. VPN: To access the University network from an external network, establish a secure connection using a VPN.

# Compiling
1. Access the cluster:
  <pre>ssh username@hpc.unitn.it</pre>
2. Reserve a Node and Enter an Interactive Session: on a node with the wanted specification. 
3. Clone the repo
<pre>git clone https://github.com/Devid663/PARCO-Computing-2026--239293.git</pre>
4. Move to head node run to submit a request using the **spmv.pbs** in the /scripts directory. It will automatically run all the experiment executions. Submit the job to the queue for compile and execution: 
<pre>qsub commands.pbs</pre>
5. View the results in the /results directory.

If personalized execution is required you can run the **spmv_exec**, specifyfing:
<pre>OMP_NUM_THREADS=8 ./programma_spmv_exec</pre>
If you want to modify the file, access the /src directory and re-compile the code:
<pre>gcc -std=c99 -O3 -fopenmp main.c utility.c mmio.c -o spmv_exec</pre>

# Directory

<pre>
├── README.md               
├── data/
│   ├── matrix1.mtx
│   ├── matrix2.mtx
│   └── ...
├── src/
│   ├── main.c
│   ├── utility.c
│   ├── mmio.c
|   ├── mmio.h
│   └── utility.h
├── scripts/
│   ├── spmv.pbs             
├── results/
│   ├── logs.txt
│   ├── spmv_exec
│   ├── ...
│   └── spmv_results.csv
└── report.pdf

</pre>
