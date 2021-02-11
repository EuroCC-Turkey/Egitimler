# Running multi-threaded programs

We can execute programs that make use of multi-threading libraries on TRUBA. Doing so requires adding some configuration options that determine the number nodes and CPUs that we wish to use for our tasks. TRUBA includes a myriad of such options, and we will cover some of the most essential ones in these examples.

In the TRUBA examples below, we will be using the code in the file `omp_example.cpp` shown below:

```cpp
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <vector>
#include <unistd.h>
#include <limits.h>
using namespace std;

// Retrieve the name of the running host
string get_host_name(){
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    return string(hostname);
}
// splits the string "input" by the character "delim" and returns
// tokens as a vector
vector<string > tokenize_string(string input, char delim){
    stringstream ss(input);
    vector<string > tokens;
    string token;
    while (getline(ss, token, delim)) tokens.push_back(token);
    return tokens;
}

// inspects the current processe's information and gets the number
// of hardware CPUs allocated for this process, as well as their IDs
pair<int, string> get_cpus(){
    ifstream fin("/proc/self/status");
    string cpus_allowed_string;
    while (getline(fin, cpus_allowed_string)){
        if (cpus_allowed_string.find("Cpus_allowed_list") != string::npos) break;
    }
    vector<string > cpu_ranges_vec = tokenize_string(cpus_allowed_string, '\t');
    string cpu_ranges = cpu_ranges_vec[1];
    vector<string > ranges = tokenize_string(cpu_ranges,',');
    int cpus = 0;
    for (auto range : ranges){
        vector<string >first_last = tokenize_string(range, '-');
        if (first_last.size()>1)
            cpus+=stoi(first_last[1])-stoi(first_last[0])+1;
        else cpus+=1;
    }
    return make_pair(cpus, cpu_ranges);
}

// Calculates Pi parallely using <num_threads> number 
// of logical threads
void calculate_pi(int num_threads){
    double pi, sum=0;

    int num_steps = (1024 * 1024 * 1024);
    double step = 1.0/num_steps;

    omp_set_num_threads(num_threads);
#pragma omp parallel 
    {
        int nt = num_threads, id;
        double x;
        id = omp_get_thread_num();
        double mysum=0;
#pragma omp for       
        for (int i=0; i < num_steps; i ++){
            x = (i + 0.5) * step;
            mysum += 4.0/(1.0+x*x);
        }
#pragma omp atomic
        sum+=mysum;
    }

    pi = sum * step;

}

int main(int argc, char *argv[]){
    // Number and IDs of hardware CPUs allocated to the task by SLURM
    pair<int, string> hardware_cpus = get_cpus();
    int hardware_cpus_num = hardware_cpus.first;
    string hardware_cpus_ids = hardware_cpus.second;
    // Number of logical threads assigned by the user for the program
    // by setting the environment variable OMP_NUM_THREADS
    int logical_threads_num = omp_get_max_threads();
    short args = 0xff;
    // the information that will be printed depends on the first arguemt
    if (argc>1)
        args = atoi(argv[1]);
    if (args & 1)
        cout << "Hostname: " << get_host_name() << endl;
    if (args & 2)
        cout << "IDs of CPUs given by SLURM: " << hardware_cpus_ids << endl;
    if (args & 4)
        cout << "Number of CPUs given by SLURM: " << hardware_cpus_num << endl;
    if (args & 8)
        cout << "Number of logical threads: " << logical_threads_num << endl;
    double start_time, end_time;
    start_time = omp_get_wtime();
    calculate_pi(logical_threads_num);
    end_time = omp_get_wtime();
    if (args & 16)
    cout << "Finished calculation in " << end_time-start_time << " seconds" << endl;
}
```

### Executing a multi-threaded task and allocating multiple CPUs for the task

We can instruct SLURM to allocate for tasks multiple CPUs instead of using a single CPU per task. This is be done by adding the `--cpus-per-task` option to the SLURM bash script options, and adding that flag to the job-step executions, i.e. the `srun` commands. This is shown in the following script `sbatch_mt1.slurm`:

```bash
#!/bin/bash

#SBATCH --account=<my_account>
#SBATCH --job-name=<job_name>
#SBATCH --partition=<part>
#SBATCH --time=d-hh:mm:ss
#SBATCH --nodes=<N>
#SBATCH --ntasks=<n> # maximum limit of tasks that can run in parallel
#SBATCH --cpus-per-task=<c> # Number of CPUs given to each task
	# The total number of CPUs that will be reserved for this job is:
	# <c> * <n>
#SBATCH --workdir=<dir>
#SBATCH --output=<out>
#SBATCH --error=<err>

# Since we did not specify --nodes and --ntasks for the two job-steps below,
# the default values will be used, which are <N> and <n>, respectively.
srun --cpus-per-task=<c1> --exclusive ./my_mt_program
srun --cpus-per-task=<c2> --exclusive ./my_mt_program
```

Then, we dispatch this script to TRUBA for execution using the command:

```bash
$ sbatch sbatch_mt1.slurm
```

`<my_account>`: account name on TRUBA

`<job_name>`: the name of the dispatched job that appears on the job queue.

`<part>`: the name of the partition on which to enqueue the work.

`<time>`: the *maximum* amount of time that your work will be running for. The format of this input is `d-hh:mm:ss` where `d` is the number of days, `hh` is the number of hours, `mm`is the number of minutes, and `ss` is the number of seconds. **Note:** if the executable does not terminate within this specified time window, **it will be terminated automatically.** 

`<N>`: the number of nodes (servers) to use to run the tasks in this script.

`<n>`: the maximum number of tasks that will run in parallel within the script. 

`<c>`: the number of CPUs dedicated for each task's execution

`<c1,2>`: the number of CPUs dedicated for the execution of each task in each of the two job-steps. It should hold that `<c1,2> <= <c>` .

`<dir>`: the path in TRUBA where the script will execute. This is where the input and output files are usually located. All the relative paths defined in the script will be relative to `<out>`

`<out>`: the file to which the `stdout` of this job is going to be printed. This includes all the outputs that the executions in the script generate.

`<err>`: the file to which the `stderr` of this job is going to be printed. 

When we call `sbatch`, we will enqueue the job into the TRUBA queue. Once resources are available and our job is at the top of the queue, the following will occur:

1.  The requested resources will be allocated for the requested time period, and in this case we requested:
    1. `<N>` nodes
    2. The ability to execute `<n>` tasks
    3. `<c>` CPUs for each task, in other words, `<c> * <n>` CPUs.
2. the lines starting with `srun` will start job-steps that will  run the program my_mt_program in`<n>` parallel tasks. The first job-step will use `<c1>` CPUs for each task, and the second job-step will use `<c2>` CPUs for each task.

**Example:**

The script  `sbatch_mt_example1.slurm`fdefines a job named `multi_threaded_job` that will compile the program `omp_example.cpp` into the binary `omp_example`. Then, it will execute `omp_example` with 4 different CPU assignments. Each time varying the number of logical threads assigned using the environment variable `OMP_NUM_THREADS` as well as changing the number of hardware CPUs assigned using the `--cpus-per-task` flag. This job is going to be added to the partition `short` and it will finish within 20 minutes of execution. The file `omp_example.cpp` is located in `/truba/home/my_account/`. The outputs from executing the jobs will be printed to the file `/truba/home/my_account/output.txt` and the errors will be printed to `/truba/home/my_account/error.txt`. Please note how the number of hardware CPUs given to job-steps changes with the flag `--cpus-per-task` and the option `--exclusive`. 

```bash
#!/bin/bash

#SBATCH --account=my_account
#SBATCH --job-name=multi_threaded_job
#SBATCH --partition=short
#SBATCH --time=0-00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --workdir=/truba/home/my_account/
#SBATCH --output=output.txt
#SBATCH --error=error.txt

# Setup the working environment - compile the C++ code into an executable
g++ omp_example.cpp -lgomp -fopenmp -O3 -o omp_example

export OMP_NUM_THREADS=2
srun --ntasks=1 --cpus-per-task=2 --exclusive ./omp_example
echo "***"
export OMP_NUM_THREADS=4
srun --ntasks=1 --cpus-per-task=2 --exclusive ./omp_example
echo "***"
echo "Increasing logical threads without increasing CPUs does not benefit runtime!"
echo "********"
echo ""

export OMP_NUM_THREADS=4
srun --ntasks=1 --cpus-per-task=4 --exclusive ./omp_example
echo "***"
echo "2x speedup over using 2 CPUs!"
echo "********"
echo ""

export OMP_NUM_THREADS=8
srun --ntasks=1 --cpus-per-task=8 --exclusive ./omp_example 
echo "***"
srun --ntasks=1 --cpus-per-task=8 ./omp_example
echo "***"
echo "Removing the --exclusve option from the srun command lead to this job-step using "
echo "all allocated CPUs!"
echo "********"
echo ""

export OMP_NUM_THREADS=16
srun --ntasks=1 --cpus-per-task=16 --exclusive ./omp_example
echo "***"
echo "Fastest performance!"
```

Then, we dispatch this script to TRUBA for execution using the command:

```bash
$ sbatch sbatch_mt_example1.slurm
```

The following is an output we observed in the file `output.txt`:

```bash
IDs of CPUs given by SLURM: 20-21
Number of CPUs given by SLURM: 2
Number of logical threads: 2
Finished calculation in 0.655512 seconds
***
IDs of CPUs given by SLURM: 20-21
Number of CPUs given by SLURM: 2
Number of logical threads: 4
Finished calculation in 0.658881 seconds
***
Increasing logical threads without increasing CPUs does not benefit runtime!
********

IDs of CPUs given by SLURM: 20-23
Number of CPUs given by SLURM: 4
Number of logical threads: 4
Finished calculation in 0.352392 seconds
***
2x speedup over using 2 CPUs!
********

IDs of CPUs given by SLURM: 20-27
Number of CPUs given by SLURM: 8
Number of logical threads: 8
Finished calculation in 0.18384 seconds
***
IDs of CPUs given by SLURM: 20-35
Number of CPUs given by SLURM: 16
Number of logical threads: 8
Finished calculation in 0.181694 seconds
***
Removing the --exclusve option from the srun command lead to this job-step using 
all allocated CPUs!
********

IDs of CPUs given by SLURM: 20-35
Number of CPUs given by SLURM: 16
Number of logical threads: 16
Finished calculation in 0.103454 seconds
***
Fastest performance!
```

### Executing multiple multi-threaded tasks in parallel

We can execute multiple multi-threaded tasks concurrently and control how they share CPUs. This is done by increasing the number of tasks in the SLURM bash script options and allocating these tasks to multiple concurrent job-steps. The script `sbatch_mt2.slurm` shown demonstrates this:

```bash
#!/bin/bash

#SBATCH --account=<my_account>
#SBATCH --job-name=<job_name>
#SBATCH --partition=<part>
#SBATCH --time=d-hh:mm:ss
#SBATCH --nodes=<N>
#SBATCH --ntasks=<n> # maximum limit of tasks that can run in parallel
#SBATCH --cpus-per-task=<c> # Number of CPUs given to each task
	# The total number of CPUs that will be reserved for this job is:
	# <c> * <n>
#SBATCH --workdir=<dir>
#SBATCH --output=<out>
#SBATCH --error=<err>

# Executes the same program <n1> times concurrently, where all executions will share 
# the same <c1> * <n1> CPUs
srun --cpus-per-task=<c1> --exclusive --ntasks=<n1> ./my_mt_program
# This job step will use <c1> * <n1> CPUs

# Execute three job-steps concurrently, but every executions will use independent CPUs
srun --cpus-per-task=<c2> --exclusive --ntasks=<n2> ./my_mt_program &
# The above job-step will use <c2> * <n2> CPUs
srun --cpus-per-task=<c3> --exclusive --ntasks=<n3> ./my_mt_program &
# The above job-step will use <c3> * <n3> CPUs
srun --cpus-per-task=<c4> --exclusive --ntasks=<n4> ./my_mt_program &
# The above job-step will use <c4> * <n4> CPUs
wait
```

Then, we dispatch this script to TRUBA for execution using the command:

```bash
$ sbatch sbatch_mt2.slurm
```

`<my_account>`: account name on TRUBA

`<job_name>`: the name of the dispatched job that appears on the job queue.

`<part>`: the name of the partition on which to enqueue the work.

`<time>`: the *maximum* amount of time that your work will be running for. The format of this input is `d-hh:mm:ss` where `d` is the number of days, `hh` is the number of hours, `mm`is the number of minutes, and `ss` is the number of seconds. **Note:** if the executable does not terminate within this specified time window, **it will be terminated automatically.** 

`<N>`: the number of nodes (servers) to use to run the tasks in this script.

`<n>`: the maximum number of tasks that will run in parallel within the script. 

`<n1,2,3,4>`: the number of tasks that will be executed by the respective job-step, where `<n1>, <n2>, <n3>, <n4> < <n>`.

`<c>`: the number of CPUs dedicated for each task's execution

`<c1,2,3,4>`: the number of CPUs dedicated for each task's executions where `<c1>, <c2>, <c3> < <c>` .

`<dir>`: the path in TRUBA where the script will execute. This is where the input and output files are usually located. All the relative paths defined in the script will be relative to `<out>`

`<out>`: the file to which the `stdout` of this job is going to be printed. This includes all the outputs that the executions in the script generate.

`<err>`: the file to which the `stderr` of this job is going to be printed. 

When we call `sbatch`, we will enqueue the job into the TRUBA queue. Once resources are available and our job is at the top of the queue, the following will occur:

1.  The requested resources will be allocated for the requested time period, and in this case we requested:
    1. `<N>` nodes
    2. The ability to execute `<n>` tasks
    3. `<c>` CPUs for each task, in other words, `<c> * <n>` CPUs.
2. the lines starting with `srun` will start job-steps that will  run the program my_mt_program in`<n1,2,3,4>` parallel tasks. The first job-step will use `<c1>` CPUs for each task, and the second job-step will use `<c2>` CPUs for each task, and so on. Notice that the CPUs used for the first job-step's tasks are shared among all tasks in that job-step. However, for the following three job-steps, each one of them will use independent CPUs. That is because they were all created as seperate job-steps, and because we used the `--exclusive` flag.
3. The script will block execution at the `wait` command until all started job-steps are done executing.

**Example:**

The script `sbatch_mt_example2.slurm` defines a job named `multi_threaded_job` that will compile the program `omp_example.cpp` into the binary `omp_example`. Then, it will execute `omp_example` three times in a single job-step (`srun`). Each task is assigned 2 CPUs using the `--cpus-per-task` flag. This means that this job-step will use 6 CPUs. Notice that all the tasks running within the job-step will share the same 6 CPUs. Afterwards, the same three tasks are executed, but this time, each of them will be in its own job-step. We make sure they all run in parallel using the `&` option at the end of the `srun` command. Notice that each of them uses a different set of CPUs.

 This job is going to be added to the partition `short` and it will finish within 20 minutes of execution. The file `omp_example.cpp` is located in `/truba/home/my_account/`. The outputs from executing the jobs will be printed to the file `/truba/home/my_account/output.txt` and the errors will be printed to `/truba/home/my_account/error.txt`. Please note how the number of hardware CPUs given to job-steps changes with the flag `--cpus-per-task` and the option `--exclusive`. 

```bash
#!/bin/bash

#SBATCH --account=my_account
#SBATCH --job-name=multi_threaded_job
#SBATCH --partition=short
#SBATCH --time=0-00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
	# The total number of CPUs that will be reserved for this job is:
	# 4 * 3 = 12 CPUs
#SBATCH --workdir=/truba/home/my_account/
#SBATCH --output=output.txt
#SBATCH --error=error.txt

# Setup the working environment - compile the C++ code into an executable
g++ omp_example.cpp -lgomp -fopenmp -O3 -std=c++11 -o omp_example

echo "Three tasks running in a single job-step (srun) will share CPUs"
echo "***"
export OMP_NUM_THREADS=2
srun --ntasks=3 --cpus-per-task=2 --exclusive ./omp_example
echo "***"
echo ""
echo "Three tasks running in separate job-steps will not share CPUs"
echo "***"
export OMP_NUM_THREADS=4
srun --ntasks=1 --cpus-per-task=2 --exclusive ./omp_example &
srun --ntasks=1 --cpus-per-task=2 --exclusive ./omp_example &
srun --ntasks=1 --cpus-per-task=2 --exclusive ./omp_example &
wait
echo "********"
echo ""
```

Then, we dispatch this script to TRUBA for execution using the command:

```bash
$ sbatch sbatch_mt_example2.slurm
```

Here is an output we observed when dispatching this SLURM script:

```
Three tasks running in a single job-step (srun) will share CPUs
Hostname: akya5.yonetim
IDs of CPUs given by SLURM: 10-15
Number of CPUs given by SLURM: 6
Number of logical threads: 2
Hostname: akya5.yonetim
IDs of CPUs given by SLURM: 10-15
Number of CPUs given by SLURM: 6
Number of logical threads: 2
Hostname: akya5.yonetim
IDs of CPUs given by SLURM: 10-15
Number of CPUs given by SLURM: 6
Number of logical threads: 2
Finished calculation in 0.687023 seconds
Finished calculation in 0.691435 seconds
Finished calculation in 0.691626 seconds
***

Three tasks running in seperate job-steps will not share CPUs
Hostname: akya5.yonetim
IDs of CPUs given by SLURM: 10-11
Number of CPUs given by SLURM: 2
Number of logical threads: 2
Finished calculation in 0.671502 seconds
Hostname: akya5.yonetim
IDs of CPUs given by SLURM: 10-11
Number of CPUs given by SLURM: 2
Number of logical threads: 2
Finished calculation in 0.668974 seconds
Hostname: akya5.yonetim
IDs of CPUs given by SLURM: 10-11
Number of CPUs given by SLURM: 2
Number of logical threads: 2
Finished calculation in 0.671319 seconds
***

```

### Executing multiple multi-threaded tasks on multiple servers

We can instruct TRUBA to use more than a single server to execute the tasks within a job. We do so using the `--nodes` option in the SLURM batch options. We can also specify the number of tasks we wish to run on each server by using the `--ntasks-per-node` flag. Please note that the SLURM batch script option `#SBATCH --ntasks` is **implicitly added** (automatically added based on the values of `--nodes` and `--ntasks-per-node`). The following file, `sbatch_mt3.slurm`, demonstrates using multiple nodes:

```bash
#!/bin/bash

#SBATCH --account=<my_account>
#SBATCH --job-name=<job_name>
#SBATCH --partition=<part>
#SBATCH --time=d-hh:mm:ss
#SBATCH --nodes=<N>
#SBATCH --ntasks-per-node=<npn> # maximum limit of tasks that can run in parallel
				# on each node
#SBATCH --cpus-per-task=<c> # Number of CPUs given to each task
	# The total number of CPUs that will be reserved for this job is:
	# <c> * ( <npn> * <N> )
	# This configuration leads to the following option being **implicitly** added
	# as an option:
	# #SBATCH --ntasks=<npn>*<N> 
#SBATCH --workdir=<dir>
#SBATCH --output=<out>
#SBATCH --error=<err>

# Executes the same program <n1> times concurrently. The jobs will be distributed
# across <N1> nodes (servers). Each node will run less than or equal to <npn> tasks
# Total number of CPUs used on all <N1> nodes = <c1> * <n1>
# Total number of CPUs used on each nodes < <npn> * <c>
srun --nodes=<N1>  --ntasks=<n1> --cpus-per-task=<c1> --exclusive ./my_mt_program

# Executes the same program <n2> times concurrently. Each node will 
# execute at most <npn2> tasks
# Total number of CPUs used on all servers = <n2> * <c2>
# Total number of CPUs used on each server < <npn2> * <c2>
srun --nodes=<N2>  --ntasks=<n2> --ntasks-per-node=<npn2> --cpus-per-task=<c2> --exclusive ./my_mt_program

# Executes the same program <n> times concurrently. Each server will run at most 
# <npn> tasks (this job-step will use the default values used for the SLURM bash
# script.) 
# Total number of CPUs used on all servers = <n3> * <c3>
# Total number of CPUs used per server <= <N3> * <npn3>
srun --nodes=<N3> --cpus-per-task=<c3> --exclusive --ntasks-per-node=<n3> ./my_mt_program
```

Then, we dispatch this script to TRUBA for execution using the command:

```bash
$ sbatch sbatch_mt3.slurm
```

`<my_account>`: account name on TRUBA

`<job_name>`: the name of the dispatched job that appears on the job queue.

`<part>`: the name of the partition on which to enqueue the work.

`<time>`: the *maximum* amount of time that your work will be running for. The format of this input is `d-hh:mm:ss` where `d` is the number of days, `hh` is the number of hours, `mm`is the number of minutes, and `ss` is the number of seconds. **Note:** if the executable does not terminate within this specified time window, **it will be terminated automatically.** 

`<N>`: the number of nodes (servers) to use to run the tasks in this script.

`<n>`: the maximum number of tasks that will run in parallel within the script. 

`<n1,2,3,4>`: the number of tasks that will be executed by the respective job-step, where `<n1>, <n2>, <n3>, <n4> < <n>`.

`<c>`: the number of CPUs dedicated for each task's execution

`<c1,2,3,4>`: the number of CPUs dedicated for each task's executions where `<c1>, <c2>, <c3> < <c>` .

`<dir>`: the path in TRUBA where the script will execute. This is where the input and output files are usually located. All the relative paths defined in the script will be relative to `<out>`

`<out>`: the file to which the `stdout` of this job is going to be printed. This includes all the outputs that the executions in the script generate.

`<err>`: the file to which the `stderr` of this job is going to be printed. 

When we call `sbatch`, we will enqueue the job into the TRUBA queue. Once resources are available and our job is at the top of the queue, the following will occur:

1.  The requested resources will be allocated for the requested time period, and in this case we requested:
    1. `<N>` nodes
    2. The ability to execute `<npn> * <N>` tasks
    3. `<c>` CPUs for each task, in other words, `<c> * (<npn> * <N>)` CPUs.
2. the lines starting with `srun` will start job-steps that will  run the program my_mt_program using `<N1>`, `<N2>`, and `<N2>` nodes, and for each, execute `<n1>, <n2>` and `<n>` parallel tasks. Each job-step will use as many CPUs per task as is specified, and the `--ntasks-per-node` flag will serve to balance the number of tasks run across nodes  

**Example:**

The script `sbatch_mt_example3.slurm` shown below demonstrates how `--ntasks-per-node` affects the execution. When given as an option to the entire SLURM bash script (at the beginning of the file), its value is used to calculate the total number of tasks that can be run in the script, and that value is added to the script as shown below. After compiling the program that will be used during execution, we start by creating a job-step that will execute 8 tasks on 2 nodes. Because of the `--ntasks-per-node` option given to the script, this job-step will not execute more than 4 tasks on each node. Afterwards, we run a job-step that will execute a task 4 times on two nodes, and we instruct it to run 2 tasks on each node. Finally, we create a job-step and we don't specify the number of tasks, nodes, or tasks per node. This leads to the job-step using the default values given to the entire SLURM bash script, i.e. `--nodes=3`, `--ntasks=4`, and `--ntasks-per-node=2`.

 This job is going to be added to the partition `short` and it will finish within 20 minutes of execution. The file `omp_example.cpp` is located in `/truba/home/my_account/`. The outputs from executing the jobs will be printed to the file `/truba/home/my_account/output.txt` and the errors will be printed to `/truba/home/my_account/error.txt`. Please note how the number of hardware CPUs given to job-steps changes with the flag `--cpus-per-task` and the option `--exclusive`. 

```bash
#!/bin/bash

#SBATCH --account=my_account
#SBATCH --job-name=multi_threaded_job
#SBATCH --partition=short
#SBATCH --time=0-00:20:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4 # maximum limit of tasks that can run in parallel
				# on each node
#SBATCH --cpus-per-task=2 # Number of CPUs given to each task
	# The total number of CPUs that will be reserved for this job is:
	# 2 * ( 4 * 3 ) = 24
	# This configuration leads to the following option being **implicitly** added
	# as an option:
	# #SBATCH --ntasks=12 
#SBATCH --workdir=/truba/home/my_account/
#SBATCH --output=output.txt
#SBATCH --error=error.txt

# Setup the working environment - compile the C++ code into an executable
g++ omp_example.cpp -lgomp -fopenmp -O3 -o omp_example

# Executes the same program 8 times concurrently. The jobs will be distributed
# across 2 nodes (servers). Each node will run 4 tasks
# Total number of CPUs used on each nodes = 8*1 = 8
srun --nodes=2  --ntasks=8 --cpus-per-task=1 --exclusive ./omp_example 1

echo ""
echo ""
echo ""
# Executes the same program 4 times concurrently. Each node will execute 2 tasks
# Total number of CPUs used on each server = 2 * 1 = 2
srun --nodes=2  --ntasks=4 --ntasks-per-node=2 --cpus-per-task=1 --exclusive ./omp_example 1
echo ""
echo ""
echo ""

# Executes the same program 12 times concurrently. Each server will run at most 
# 4 tasks (this job-step will use the default values used for the SLURM bash
# script.) 
# Total number of CPUs used per server = 12
srun --cpus-per-task=2 --exclusive ./omp_example 1
```

Then, we dispatch this script to TRUBA for execution using the command:

```bash
$ sbatch sbatch_mt_example3.slurm
```

Here is an output we observed when dispatching this SLURM script:

```
Hostname: barbun130.yonetim
Hostname: barbun130.yonetim
Hostname: barbun130.yonetim
Hostname: barbun130.yonetim
Hostname: barbun131.yonetim
Hostname: barbun131.yonetim
Hostname: barbun131.yonetim
Hostname: barbun131.yonetim

Hostname: barbun130.yonetim
Hostname: barbun130.yonetim
Hostname: barbun131.yonetim
Hostname: barbun131.yonetim

Hostname: barbun132.yonetim
Hostname: barbun130.yonetim
Hostname: barbun132.yonetim
Hostname: barbun132.yonetim
Hostname: barbun132.yonetim
Hostname: barbun130.yonetim
Hostname: barbun130.yonetim
Hostname: barbun130.yonetim
Hostname: barbun131.yonetim
Hostname: barbun131.yonetim
Hostname: barbun131.yonetim
Hostname: barbun131.yonetim
```