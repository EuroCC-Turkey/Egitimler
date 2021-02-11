# Running an MPI program

MPI allows something that is very important for HPC - it enables multiple nodes (servers) to work on the same procedure. This collaboration between servers is quite seamless and easy to work with on a system like TRUBA. To execute tasks that involve MPI, we need to start job-steps using the `mpirun` command (instead of the command `srun` used so far). Job-steps that are created using the `mpirun` command will still create multiple tasks, however, these tasks will *work on the same MPI procedure.* On the other hand, tasks that are created using the `srun` command are independent from each other. 

We will use the following MPI program to experiment with OMP:

```c
#include <stdio.h>
#include <omp.h>
#include <limits.h>
#include <unistd.h>
static long long num_steps=1000000000;
double step;
char hostname[HOST_NAME_MAX];

int main(int argc, char** argv){
        gethostname(hostname, HOST_NAME_MAX);
        int i, myid, num_procs;
        double x, pi, remote_sum, sum=0, start=0, end=0;;
        MPI_Init(&argc, &argv);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        start = MPI_Wtime();
        step = 1.0/(double) num_steps;
        for (i = myid; i< num_steps; i=i+num_procs){
                x =(i+0.5)*step;
                sum +=4.0/(1.0+x*x);
        }
        printf("Process %d running on %s\n", myid, hostname);
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
                printf("Processes %d, took %f \n", num_procs, end-start);
        }
        return 0;
}
```

### Executing MPI programs using multiple nodes (and multiple tasks per node) to work on the same procedure

The script `mpi.slurm` shown below demonstrates how we could use multiple nodes to execute MPI tasks:

```bash
#!/bin/bash

#SBATCH --account=<my_account>
#SBATCH --job-name=<job_name>
#SBATCH --partition=<part>
#SBATCH --time=d-hh:mm:ss
#SBATCH --nodes=<N> # Number of nodes that will take part in the MPI procedure
#SBATCH --ntasks-per-node=<npn> # maximum limit of processes that can run in parallel
				# on a single node
#SBATCH --workdir=<dir>
#SBATCH --output=<out>
#SBATCH --error=<err>

# This MPI procedure will use <n1> tasks only
mpirun -np=<n1> ./my_mpi_program

# Thsi MPI procedure will use <npn> * <N> tasks
mpirun ./my_mpi_program
```

Then, we dispatch this script to TRUBA for execution using the command:

```bash
$ sbatch mpi_example.slurm
```

`<my_account>`: account name on TRUBA

`<job_name>`: the name of the dispatched job that appears on the job queue.

`<part>`: the name of the partition on which to enqueue the work.

`<time>`: the *maximum* amount of time that your work will be running for. The format of this input is `d-hh:mm:ss` where `d` is the number of days, `hh` is the number of hours, `mm`is the number of minutes, and `ss` is the number of seconds. **Note:** if the executable does not terminate within this specified time window, **it will be terminated automatically.** 

`<N>`: the number of nodes (servers) to use to run the tasks in this script.

`<npn>`: the maximum number of tasks that will run in parallel on a single script

The total number of tasks that can be used to run a job-step in the script is `<N> * <npn>`

`<n1>`: the number of tasks that will contribute to the respective MPI job-step.

`<dir>`: the path in TRUBA where the script will execute. This is where the input and output files are usually located. All the relative paths defined in the script will be relative to `<out>`

`<out>`: the file to which the `stdout` of this job is going to be printed. This includes all the outputs that the executions in the script generate.

`<err>`: the file to which the `stderr` of this job is going to be printed. 

When we call `sbatch`, we will enqueue the job into the TRUBA queue. Once resources are available and our job is at the top of the queue, the following will occur:

1.  The requested resources will be allocated for the requested time period, and in this case we requested:
    1. `<N>` nodes
    2. The ability to execute `<npn> * <N>` tasks
    3. 1 CPU for each task, in other words, `(<npn> * <N>)` CPUs.
2. the lines starting with `mpirun` will start job-steps that will  run the program my_mpi_program using `<N1>` nodes. The first job-step will use `<n1>` tasks to run its procedure. The second will use `<n>` tasks.   

**Example:**

The script `mpi_example.slurm` shown below demonstrates how MPI programs can be run on TRUBA and shows how the number of tasks dedicated for execution can be varied using the `-np` option. We start the script by providing the number of nodes we wish to utilize, as well as the number of tasks we wish to use on each node. Then, we set up the environment of execution by loading the needed modules and compiling our MPI code. Finally we execute two MPI job-steps. Each job step uses a different number of tasks, however, unlike `srun` when a job-step is started using `mpirun`, the tasks it creates will all work on the same procedure instead of being independent.  

This job is going to be added to the partition `short` and it will finish within 20 minutes of execution. The file `mpi.c` is located in `/truba/home/my_account/`. The outputs from executing the jobs will be printed to the file `/truba/home/my_account/output.txt` and the errors will be printed to `/truba/home/my_account/error.txt`. 

```bash
#!/bin/bash

#SBATCH --account=<my_account>
#SBATCH --job-name=my_job
#SBATCH --partition=short
#SBATCH --time=0-00:02:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
		# Job-steps created in this job will be able to create up to 8 tasks
#SBATCH --workdir=/truba/home/my_account/
#SBATCH --output=output.txt
#SBATCH --error=error.txt

# Setup the environment
# 1. load the modules required for compilation
module purge # remove any modules that were loaded on the client server to start fresh
module load centos7.3/comp/gcc/7
module load centos7.3/lib/openmpi/3.0.0-gcc-7.0.1
# 2. compile code
mpicc mpi.c -o mpi

echo "Using four tasks"
mpirun -np 4 ./mpi
echo ""
echo ""
echo "Using six tasks"
mpirun -np 6 ./mpi
echo ""
echo ""
echo "Using eight tasks"
mpirun ./mpi
echo ""
echo ""
```

Then, we dispatch this script to TRUBA for execution using the command:

```bash
$ sbatch mpi_example.slurm
```

Here is an output we observed when dispatching this SLURM script:

```
Using four tasks
Process 1 running on akya14.yonetim
Process 3 running on akya14.yonetim
Process 0 running on akya14.yonetim
Process 2 running on akya14.yonetim
Processes 4, took 3.254500 

Using six tasks
Process 4 running on akya15.yonetim
Process 0 running on akya14.yonetim
Process 2 running on akya14.yonetim
Process 5 running on akya15.yonetim
Process 3 running on akya14.yonetim
Process 1 running on akya14.yonetim
Processes 6, took 2.209622 

Using eight tasks
Process 1 running on akya14.yonetim
Process 6 running on akya15.yonetim
Process 0 running on akya14.yonetim
Process 3 running on akya14.yonetim
Process 4 running on akya15.yonetim
Process 5 running on akya15.yonetim
Process 2 running on akya14.yonetim
Process 7 running on akya15.yonetim
Processes 8, took 1.722208
```