# Running a CUDA program

There are many servers equipped with GPUs that can be used for scientific computing on TRUBA. We can utilize these GPUs in our TRUBA scripts easily. In the TRUBA example below, we will use the following CUDA code `cuda_example.cu` which calculates the value of PI using one or more GPUs.

```cpp
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#define NUM_BLOCKS 1024
#define NUM_THREADS_PER_BLOCK 1024
#define THREADS_PER_GPU  (NUM_BLOCKS*NUM_THREADS_PER_BLOCK)
inline void CudaCheck(cudaError_t error, const char *file_name, int line);
#define CUDA_CHECK(error) CudaCheck((error), __FILE__, __LINE__)

// Used for error handling
inline void CudaCheck(cudaError_t error, const char *file_name, int line) {
  if(error != cudaSuccess)
    std::cout << "CUDA error " << cudaGetErrorString(error) << " at " << file_name << ":" << line << std::endl;
}

// Used to sum
#define FULL_MASK 0xffffffff
__inline__ __device__
double warpAllReduceSum(double val) {

  for (int i =1; i<32; i*=2){
    val+=__shfl_xor_sync(FULL_MASK,val,i);
  }
  return val;
}
__global__ void pi(unsigned long long i_start, unsigned long long iters, float step, float* storage){
        unsigned long long tid = threadIdx.x + blockIdx.x * blockDim.x;
        double p_sum = 0;
        for (long long i =i_start+iters*tid; i< i_start+iters*(tid+1); i++){
            double x = (i+0.5)*step;
            double sum = 4.0/(1.0+x*x);
            double nsum = warpAllReduceSum(sum);
            p_sum+=nsum;
        }
        if (tid%32 ==0)
                storage[tid/32] = p_sum;
}

int main(int argc, char *argv[]){
        float sum=0;
        int num_gpus= atoi(argv[1]);
        unsigned long long num_steps = 1024.0*1024.0*1024.0*256.0;
        double step = 1.0/(double)num_steps;
        float ** d_values = new float*[num_gpus], **h_values = new float*[num_gpus];
        for (int i =0; i<num_gpus; i++){
            CUDA_CHECK(cudaSetDevice(i));
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, I);
	    printf("Device Number: %d\n", i);
            printf("  Device name: %s\n", prop.name);
            CUDA_CHECK(cudaMalloc((void**)&d_values[i],(THREADS_PER_GPU/32)*sizeof(float)));
            h_values[i] = new float[(THREADS_PER_GPU/32)];
        }
        double start = omp_get_wtime();
        int iters = num_steps/(long long)THREADS_PER_GPU/(long long)num_gpus;
        omp_set_nested(true);
        omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel num_threads(num_gpus)
        {
            double gpu_sum=0;
            int gpu_id = omp_get_thread_num();
            cudaSetDevice(gpu_id);
            pi<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(1.0*THREADS_PER_GPU*gpu_id*iters, iters, step, d_values[gpu_id]);
            CUDA_CHECK(cudaMemcpy(h_values[gpu_id], d_values[gpu_id], THREADS_PER_GPU/32*sizeof(float), cudaMemcpyDeviceToHost));
#pragma omp parallel for reduction(+:gpu_sum)
            for (int j =0; j<THREADS_PER_GPU/32;j++){
                    gpu_sum+=h_values[gpu_id][j];
            }
            #pragma omp atomic
                sum+=gpu_sum;
        }
        sum*=step;
        printf("pi %f with %lld steps in %f seconds\n", sum, num_steps, omp_get_wtime()-start);
        return 0;
}
```

### Executing tasks that use one or more GPUs

We can dipatch jobs that utilize one or more GPUs to TRUBA by simply requesting the number of GPUs we need, as well as load any modules required to compile and execute our program. The SLURM script `usbatch_cuda.slurm` shown below demonstrates this process:

 

```bash
#!/bin/bash

#SBATCH --account=<my_account>
#SBATCH --job-name=<job_name>
#SBATCH --partition=<part>
#SBATCH --time=d-hh:mm:ss
#SBATCH --nodes=<N>
#SBATCH --gres=gpu:<g> # the number of GPUs required on each node
#SBATCH --ntasks=<n> # maximum limit of tasks that can run in parallel
#SBATCH --cpus-per-task=<c> # Number of CPUs given to each task
	# The total number of CPUs that will be reserved for this job is:
	# <c> * <n>
#SBATCH --workdir=<dir>
#SBATCH --output=<out>
#SBATCH --error=<err>

# Setup the environment
# 1. load the modules required for compilation
module load <nvcc_version> 
# 2. compile code
nvcc ...

srun --nodes=<N1> --ntasks=<n1> --cpus-per-task=<c1> ./my_cuda_program
```

`<my_account>`: account name on TRUBA

`<job_name>`: the name of the dispatched job that appears on the job queue.

`<part>`: the name of the partition on which to enqueue the work.

`<time>`: the *maximum* amount of time that your work will be running for. The format of this input is `d-hh:mm:ss` where `d` is the number of days, `hh` is the number of hours, `mm`is the number of minutes, and `ss` is the number of seconds. **Note:** if the executable does not terminate within this specified time window, **it will be terminated automatically.** 

`<N>`: the number of nodes (servers) to use to run the tasks in this script.

`<gpu>`: the number of GPUs that will be reserved in each node.

`<n>`: the maximum number of tasks that will run in parallel within the script. 

`<n1>`: the number of tasks that will be executed by the respective job-step, where `<n1> <= <n>`.

`<c>`: the number of CPUs dedicated for each task's execution

`<c1>`: the number of CPUs dedicated for each task's executions where `<c1> <= <c>` .

`<dir>`: the path in TRUBA where the script will execute. This is where the input and output files are usually located. All the relative paths defined in the script will be relative to `<out>`

`<out>`: the file to which the `stdout` of this job is going to be printed. This includes all the outputs that the executions in the script generate.

`<err>`: the file to which the `stderr` of this job is going to be printed. 

When we call `sbatch`, we will enqueue the job into the TRUBA queue. Once resources are available and our job is at the top of the queue, the following will occur:

1.  The requested resources will be allocated for the requested time period, and in this case we requested:
    1. `<N>` nodes
    2. `<gpu>` GPUs on each node.
    3. The ability to execute `<n>` tasks in parallel
    4. `<c>` CPUs for each task, in other words, `<c> * <n>` CPUs.
2. We prepare the environment to run the program by loading the right modules and compiling the code.
3. the line starting with `srun` will start a job-step that will  run the program my_cuda_program using `<N1>` execute `<n1>` parallel tasks. Each job-step will use as many CPUs per task as is specified.

**Example:**

The following script, namely `sbatch_cuda_example.slurm` will execute a job that utilizes two GPUs, as well as multi-threading using OMP. First, we set-up the work environment by loading the required modules for compiling and running our code. Then, we compile the code (making sure that we also link our code to the OMP library). Finally, we execute two job-steps that will each execute a single task on a single node and use 16 CPUs (the values for `--ntasks`, `--nodes`, and `--cpus-per-task` weren't specified for the job-steps, but job-steps use the value given to the SLURM bash script by default when no options are specified). 

This job is going to be added to the partition `short` and it will finish within 20 minutes of execution. The file `cuda_mg.cu` is located in `/truba/home/my_account/`. The outputs from executing the jobs will be printed to the file `/truba/home/my_account/output.txt` and the errors will be printed to `/truba/home/my_account/error.txt.` 

```bash
#!/bin/bash

#SBATCH --account=<my_account>
#SBATCH --job-name=my_job
#SBATCH --partition=short
#SBATCH --time=0-00:02:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --workdir=/truba/home/my_account/
#SBATCH --output=output.txt
#SBATCH --error=error.txt

# Setup the environment
# 1. load the modules required for compilation
module purge # remove any modules that were loaded on the client server to start fresh
module load centos7.3/comp/gcc/7
module load centos7.3/lib/cuda/10.1
# 2. compile code
nvcc -std=c++11 -Xcompiler -fopenmp -lgomp -O3 cuda_mg.cu -o cuda_program
# 3. set the environment variable used by OMP
export OMP_NUM_THREADS=16

# Execute job steps
echo "Single GPU"
echo "***"
srun ./cuda_program 1
echo "***"

echo ""
echo "Two GPU"
echo "***"
srun ./cuda_program 2
echo "***"
```

Then, we dispatch this script to TRUBA for execution using the command:

```bash
$ sbatch cuda_example.slurm
```

Here is an output we observed when dispatching this SLURM script:

```bash
Single GPU
***
Device Number: 0
  Device name: Tesla V100-SXM2-16GB
pi 3.141593 with 274877906944 steps in 1.339614 seconds
***

Two GPU
***
Device Number: 0
  Device name: Tesla V100-SXM2-16GB
Device Number: 1
  Device name: Tesla V100-SXM2-16GB
pi 3.141593 with 274877906944 steps in 0.687545 seconds
***
```