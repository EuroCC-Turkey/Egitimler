# Running an OMP + MPI program

We can create programs that take advantage of both OMP multi-threading and MPI multi-node capabilities. In the example, we will be using the following program `omp_mpi_example.c` that calculates the value of Pi. It can use multiple MPI tasks (multiple nodes) and utilizes multi-threading on each task.

```bash
#include <mpi.h>
#include <stdio.h>
#include <omp.h>
static long long num_steps=1024.0*1024.0*1024.0*2.0;
double step;
int main(int argc, char** argv){
        unsigned long long i;
        int  myid, num_procs;
        double x, pi, remote_sum, sum=0, start=0, end=0;;
        MPI_Init(&argc, &argv);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        start = MPI_Wtime();
        step = 1.0/(double) num_steps;
        omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for reduction(+:sum)
        for (i = myid; i< num_steps; i=i+num_procs){
                x =(i+0.5)*step;
                sum +=4.0/(1.0+x*x);
        }
        if (myid==0){
                for (i = 1; i< num_procs;i++){
                        MPI_Status status;
                        MPI_Recv(&remote_sum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
                        sum +=remote_sum;
                }
                pi=sum*step;
        } else {
                MPI_Send(&sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        MPI_Finalize();

        if (myid ==0){
                end = MPI_Wtime();
                printf("Used %d tasks and %d threads per task and took %f seconds \n", num_procs, omp_get_max_threads(), end-start);
        }
        return 0;
}
```

### Executing an MPI program that uses OMP multi-threading

TRUBA enables the usage of both MPI and OpenMP in the same program. MPI allows for multiple nodes to work together on the same procedure, while OpenMP allows each contributing node to utilize multiple threads. To execute a multi-threaded application with an MPI program, we need to first set the number of CPUs that each MPI process will take by setting the `--cpus-per-task` value for the SLURM bash script. Also, in the `mpirun` call which creates the job-step that will run the MPI program, we need to add the option `--bind-to none`. The script `mpi_omp.slurm` shown below demonstrates this process:

```bash
#!/bin/bash

#SBATCH --account=<my_account>
#SBATCH --job-name=<job_name>
#SBATCH --partition=<part>
#SBATCH --time=d-hh:mm:ss
#SBATCH --nodes=<N> # Number of nodes that will take part in the MPI procedure
#SBATCH --ntasks-per-node=<npn> # maximum limit of processes that can run in parallel 
				# on a single node
#SBATCH --cpus-per-task=<c> # Number of CPUs given to each task
	# The total number of CPUs that will be reserved for this job is:
	# <c> * <n>
#SBATCH --workdir=<dir>
#SBATCH --output=<out>
#SBATCH --error=<err>

mpirun -np=<n1> --bind-to none ./my_mpi_omp_program

# This MPI execution will use <n> tasks
mpirun --bind-to none ./my_mpi_omp_program
```

Then, we dispatch this script to TRUBA for execution using the command:

```bash
$ sbatch example_mt1.slurm
```

`<my_account>`: account name on TRUBA

`<job_name>`: the name of the dispatched job that appears on the job queue.

`<part>`: the name of the partition on which to enqueue the work.

`<time>`: the *maximum* amount of time that your work will be running for. The format of this input is `d-hh:mm:ss` where `d` is the number of days, `hh` is the number of hours, `mm`is the number of minutes, and `ss` is the number of seconds. **Note:** if the executable does not terminate within this specified time window, **it will be terminated automatically.** 

`<N>`: the number of nodes (servers) to use to run the tasks in this script.

`<n>`: the maximum number of tasks that will run in parallel within the script. 

`<n1>`: the number of tasks that will contribute to the respective MPI job-step

`<c>`: the number of CPUs dedicated for each task's execution

`<dir>`: the path in TRUBA where the script will execute. This is where the input and output files are usually located. All the relative paths defined in the script will be relative to `<out>`

`<out>`: the file to which the `stdout` of this job is going to be printed. This includes all the outputs that the executions in the script generate.

`<err>`: the file to which the `stderr` of this job is going to be printed. 

When we call `sbatch`, we will enqueue the job into the TRUBA queue. Once resources are available and our job is at the top of the queue, the following will occur:

1.  The requested resources will be allocated for the requested time period, and in this case we requested:
    1. `<N>` nodes
    2. The ability to execute `<npn> * <N>` tasks
    3. `<c>` CPU for each task, in other words, `<c> * (<npn> * <N>)` CPUs.
2. the lines starting with `mpirun` will start job-steps that will  run the program my_mpi_omp_program using `<N>` nodes. The first job-step will use `<n1>` tasks to run its procedure. The second will use `<n>` tasks.  Both of these job-steps' tasks will use `<c>` threads. 

**Example:**

The script `sbatch_omp_mpi_example.slurm` below demonstrates the usage of OMP with an MPI job that uses multiple nodes. Each node will have 2 tasks, and each task will utilize 4 threads. We begin the script by setting the options required for the job including the number of nodes to be used, the number of tasks running on each node, and the number of CPUs that each task will use. Then, we set-up the work environment by loading the needed modules (`gcc` and `openmpi`) and compiling the code that will be used in the execution. Afterward, we execute three job-steps. Each job-step execution will execute multiple tasks and *all* the tasks within the same MPI job-step will work together on the same procedure. Each job-step execution will use a different number of logical threads that we determine by setting the `OMP_NUM_THREADS` environment variable.

This job is going to be added to the partition `short` and it will finish within 20 minutes of execution. The file `omp_mpi_example.c` is located in `/truba/home/my_account/`. The outputs from executing the jobs will be printed to the file `/truba/home/my_account/output.txt` and the errors will be printed to `/truba/home/my_account/error.txt`. Please note how the number of hardware CPUs given to job-steps changes with the flag `--cpus-per-task` and the option `--exclusive`. 

```bash
#SBATCH --account=<my_account>
#SBATCH --job-name=my_job
#SBATCH --partition=short
#SBATCH --time=0-00:02:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --workdir=/truba/home/my_account/
#SBATCH --output=output.txt
#SBATCH --error=error.txt

# Setup the environment
# 1. load the modules required for compilation
module purge # remove any modules that were loaded on the server that was used 
	     # when dispatching the job
module load centos7.3/comp/gcc/7
module load centos7.3/lib/openmpi/3.0.0-gcc-7.0.1
# 2. compile code
mpicc -lgomp -fopenmp omp_mpi_example.c -o mpi

# Execute MPI job-steps - all the tasks created for a single mpirun job-step will work on the same procedure
echo "Using a single logical thread per task"
export OMP_NUM_THREADS=1
mpirun --bind-to none ./mpi
echo ""
echo "Using 2 logical threads per task"
export OMP_NUM_THREADS=2
mpirun --bind-to none ./mpi
echo ""
echo "Using 4 logical threads per task"
export OMP_NUM_THREADS=4
mpirun --bind-to none ./mpi
```

Then, we dispatch this script to TRUBA for execution using the command:

```bash
$ sbatch mpi_omp_example.slurm
```

Here is an output we observed when dispatching this SLURM script:

```
Using a single logical thread per task
Used 4 tasks and 1 threads per task and took 7.506095 seconds 

Using 2 logical threads per task
Used 4 tasks and 2 threads per task and took 3.905180 seconds 

Using 4 logical threads per task
Used 4 tasks and 4 threads per task and took 2.834785 seconds
```