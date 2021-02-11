# Running a script of one or more programs on TRUBA using sbatch

Using scripts to execute work on TRUBA allows users to execute multiple commands to run on the cluster instead of feeding them one by one and waiting for each to finish. Also, users can dispatch the work and leave TRUBA and the work will be finished in the background automatically.

Executing work using scripts is done by defining a `bash` script that contains the steps to be taken during the work, and using the `sbatch` command to send this script to TRUBA. This process does not block the terminal and is completely in the background.

In the upcoming examples, we will be using the executable `print_argument` which executes the code in the file `print_argument.cpp` shown below:

```cpp
// print_argument.cpp
#include <iostream>
#include <chrono>
#include <thread>
using namespace std;

// Given that the first argument passed to the executable is arg,
// will print the line "My argument is <arg> and I started", wait
// for 1 second, then print "My argument is <arg> and I finished"
int main(int argc, char * argv[]){
    cout << "My argument is " << argv[1] << " and I started\n";
    this_thread::sleep_for(chrono::seconds(1));
    cout << "My argument is " << argv[1] << " and I finished\n";
    return 0;
}
```

### Executing a single task

To run the executable `my_program` we can define the script named `sbatch1.slurm`:

```bash
#!/bin/bash

#SBATCH --account=<my_account>
#SBATCH --job-name=<job_name>
#SBATCH --partition=<part>
#SBATCH --time=d-hh:mm:ss
#SBATCH --workdir=<dir>
#SBATCH --output=<out>
#SBATCH --error=<err>

srun ./my_program
```

Then, we dispatch this script to TRUBA for execution using the command:

```bash
$ sbatch sbatch1.slurm
```

`<my_account>`: account name on TRUBA

`<job_name>`: the name of the dispatched job that appears on the job queue.

`<part>`: the name of the partition on which to enqueue the work.

`<time>`: the *maximum* amount of time that your work will be running for. The format of this input is `d-hh:mm:ss` where `d` is the number of days, `hh` is the number of hours, `mm`is the number of minutes, and `ss` is the number of seconds. **Note:** if the executable does not terminate within this specified time window, **it will be terminated automatically.** 

`<dir>`: the path in TRUBA where the script will execute. This is where the input and output files are usually located. All the relative paths defined in the script will be relative to `<out>`

`<out>`: the file to which the `stdout` of this job is going to be printed. This includes all the outputs that the executions in the script generate.

`<err>`: the file to which the `stderr` of this job is going to be printed. 

When we call `sbatch`, we will enqueue the job into the TRUBA queue. Once resources are available and our job is at the top of the queue, the following will occur:

1.  The requested resources will be allocated for the requested time period, and in this case we requested:
    1. 1 node
    2. 1 CPU on the node
    3. The ability to execute 1 task
2. the line starting with `srun` will start a job-step that will  run the program my_program.

**Example:**

The script `sbatch_example1.slurm` defines a job named `my_job` that will compile the code in `print_argument.cpp`, then execute the compiled program `print_argument` three times. This job is going to be added to the partition `short` and it will finish within 20 minutes of execution. The file `print_argument.cpp` is located in `/truba/home/my_account/`. The outputs from executing the jobs will be printed to the file `/truba/home/my_account/output.txt` and the errors will be printed to `/truba/home/my_account/error.txt`

```bash
#!/bin/bash

#SBATCH --account=my_account
#SBATCH --job-name=my_job
#SBATCH --partition=short
#SBATCH --time=0-00:20:00
#SBATCH --workdir=/truba/home/my_account/
#SBATCH --output=output.txt
#SBATCH --error=error.txt

# Setup the working environment - compile the C++ code into an executable
g++ print_argument.cpp -o print_argument

# Execute job-steps (tasks)
srun ./print_argument First
srun ./print_argument Second
srun ./print_argument Third
```

We dispatch this script to TRUBA using the command

```bash
$ sbatch sbatch_example1.slurm
```

The contents of the file `output.txt` are:

```
My argument is First and I started
My argument is First and I finished
My argument is Second and I started
My argument is Second and I finished
My argument is Third and I started
My argument is Third and I finished
```

### Executing a task multiple times in parallel within a single job-step

We can run the same program execution multiple times in parallel by adding the `--ntasks` option to the SLURM bash script options to define the maximum number of parallel executions that will run in the job, and adding the option `--ntasks` to the job-step, i.e. the `srun` command, to specify the number of tasks that will run the program. This is shown in the following script `sbatch2.slurm`:

 

```bash
#!/bin/bash

#SBATCH --account=<my_account>
#SBATCH --job-name=<job_name>
#SBATCH --partition=<part>
#SBATCH --time=d-hh:mm:ss
#SBATCH --ntasks=<n> # maximum limit of tasks that can run in parallel
#SBATCH --workdir=<dir>
#SBATCH --output=<out>
#SBATCH --error=<err>

srun --ntasks=<n1> ./my_program1
srun --ntasks=<n2> ./my_program2
srun ./my_program3 # when --ntasks is not given, the default value
		  # will be <n> (value given to the SLURM bash script)
```

Then, we dispatch this script to TRUBA for execution using the command:

```bash
$ sbatch sbatch2.slurm
```

`<my_account>`: account name on TRUBA

`<job_name>`: the name of the dispatched job that appears on the job queue.

`<part>`: the name of the partition on which to enqueue the work.

`<time>`: the *maximum* amount of time that your work will be running for. The format of this input is `d-hh:mm:ss` where `d` is the number of days, `hh` is the number of hours, `mm`is the number of minutes, and `ss` is the number of seconds. **Note:** if the executable does not terminate within this specified time window, **it will be terminated automatically.** 

`<n>`: the maximum number of tasks that will run in parallel within the script

`<n1,2>`: the number of tasks that will run in each job-step where `<n1,2> <= <n>`

`<dir>`: the path in TRUBA where the script will execute. This is where the input and output files are usually located. All the relative paths defined in the script will be relative to `<out>`

`<out>`: the file to which the `stdout` of this job is going to be printed. This includes all the outputs that the executions in the script generate.

`<err>`: the file to which the `stderr` of this job is going to be printed. 

**Note:** If an `srun` call does not specify a value for `--ntasks`, the value given at the beginning of the SLURM script is used.

When we call `sbatch`, we will enqueue the job into the TRUBA queue. Once resources are available and our job is at the top of the queue, the following will occur:

1. the requested resources will be allocated for the requested time period, and in this case we requested:
    1. The ability to execute `<n>` tasks in parallel
    2. `<n>` CPUs (TRUBA assumes each task will require 1 CPU)
    3. As many nodes as TRUBA thinks is enough since we did not specify an exact number of nodes.
2. the lines starting with `srun` will start job-steps that will run the programs `my_program1, my_program2`, and `my_program3`.

Example:

The script `sbatch_example2.slurm` defines a job named `my_job` that will compile the code in `print_argument.cpp`, and then will execute the program `print_argument` at first three times, then two times, and finally three times. This job is going to be added to the partition `short` and it will finish within 20 minutes of execution. The file `print_argument.cpp` is located in `/truba/home/my_account/`. The outputs from executing the jobs will be printed to the file `/truba/home/my_account/output.txt` and the errors will be printed to `/truba/home/my_account/error.txt`

```bash
#!/bin/bash

#SBATCH --account=my_account
#SBATCH --job-name=my_job
#SBATCH --partition=short
#SBATCH --time=0-00:20:00
#SBATCH --ntasks=3
#SBATCH --workdir=/truba/home/my_account/
#SBATCH --output=output.txt
#SBATCH --error=error.txt

# Setup the working environment - compile the C++ code into an executable
g++ print_argument.cpp -o print_argument

# Execute job-steps (tasks)
srun --ntasks=3 ./print_argument ThreeTimes # will execute 3 times
srun --ntasks=2 ./print_argument TwoTimes # will execute 2 times
srun ./my_program Third # will execute 3 times (default is the
			# value set for the entire bash script
###### WRONG ######
# srun --ntasks=4 ./print_argument WONT_WORK
# Cannot issue more tasks than declared for the SLURM script
```

We dispatch this script to TRUBA using the command

```bash
$ sbatch sbatch_example2.slurm
```

The following is the output we observe in the file `output.txt`:

```bash
My argument is ThreeTimes and I started
My argument is ThreeTimes and I started
My argument is ThreeTimes and I started
My argument is ThreeTimes and I finished
My argument is ThreeTimes and I finished
My argument is ThreeTimes and I finished
My argument is TwoTimes and I started
My argument is TwoTimes and I started
My argument is TwoTimes and I finished
My argument is TwoTimes and I finished
My argument is ThreeTimesAgain and I started
My argument is ThreeTimesAgain and I started
My argument is ThreeTimesAgain and I started
My argument is ThreeTimesAgain and I finished
My argument is ThreeTimesAgain and I finished
My argument is ThreeTimesAgain and I finished
```

### Executing multiple different tasks in parallel (in different job-steps)

We can execute multiple programs in parallel by executing each of them in a separate job-step, i.e., different `srun` command, and add the symbol `&` to the end of the `srun` command.  This tells the script to move the command to the background. Please note that after starting job-steps in the background, we **must** add a `wait` command to make sure the job doesn't end before the job-steps are over. An example of executing multiple job-steps in parallel is demonstrated in the following script, `sbatch3.slurm`:

```bash
#!/bin/bash

#SBATCH --account=<my_account>
#SBATCH --job-name=<job_name>
#SBATCH --partition=<part>
#SBATCH --time=d-hh:mm:ss
#SBATCH --ntasks=<n> # maximum number of tasks that can run in parallel
#SBATCH --workdir=<dir>
#SBATCH --output=<out>
#SBATCH --error=<err>

srun --ntasks=<n1> ./my_program1 &
srun --ntasks=<n2> ./my_program2 &
srun --ntasks=<n3> ./my_program3 &
# Where  <n1> + <n2> + <n3> <= <n>
wait

```

Then, we dispatch this script to TRUBA for execution using the command:

```bash
3$ sbatch sbatch3.slurm
```

`<my_account>`: account name on TRUBA

`<job_name>`: the name of the dispatched job that appears on the job queue.

`<part>`: the name of the partition on which to enqueue the work.

`<time>`: the *maximum* amount of time that your work will be running for. The format of this input is `d-hh:mm:ss` where `d` is the number of days, `hh` is the number of hours, `mm`is the number of minutes, and `ss` is the number of seconds. **Note:** if the executable does not terminate within this specified time window, **it will be terminated automatically.** 

`<n>`: the maximum number of tasks that will run in parallel within the script

`<n1,2,3>`: the number of parallel tasks that will run `my_program1`, `my_program2`, and `my_program3`, respectively.  It must hold that `<n1> + <n2> + <n3>` ≤ `<n>`

`<dir>`: the path in TRUBA where the script will execute. This is where the input and output files are usually located. All the relative paths defined in the script will be relative to `<out>`

`<out>`: the file to which the `stdout` of this job is going to be printed. This includes all the outputs that the executions in the script generate.

`<err>`: the file to which the `stderr` of this job is going to be printed. 

When we call `sbatch`, we will enqueue the job into the TRUBA queue. Once resources are available and our job is at the top of the queue, the following will occur:

1. the requested resources will be allocated for the requested time period, and in this case we requested:
    1. The ability to execute `<n>` tasks in parallel
    2. `<n>` CPUs (TRUBA assumes each task will require 1 CPU)
    3. As many nodes as TRUBA thinks is enough since we did not specify an exact number of nodes.
2.  the lines starting with `srun` will start job-steps that will run the programs `my_program1, my_program2`, and `my_program3`. However, these job-steps will run *in parallel* since we added the `&` symbol at the end of their line.
3. The script will block execution at the `wait` command until all started job-steps are done executing.

**Example:**

The script `sbatch_example3.slurm` defines a job named `my_parallel_job` that will execute the program `print_argument` three times *in parallel*, each execution will have a different parameter. The script will wait at the `wait` command, and once all three executions are over, we execute the program three more times in parallel with a different set of arguments and wait for their termination at the second `wait` command. This job is going to be added to the partition `short` and it will finish within 20 minutes of execution. The file `print_argument.cpp` is located in `/truba/home/my_account/`. The outputs from executing the jobs will be printed to the file `/truba/home/my_account/output.txt` and the errors will be printed to `/truba/home/my_account/error.txt`

```bash
#!/bin/bash

#SBATCH --account=my_account
#SBATCH --job-name=my_parallel_job
#SBATCH --partition=short
#SBATCH --time=0-00:20:00
#SBATCH --ntasks=3
#SBATCH --workdir=/truba/home/my_account/
#SBATCH --output=output.txt
#SBATCH --error=error.txt

# Setup the working environment - compile the C++ code into an executable
g++ print_argument.cpp -o print_argument

srun --ntasks=1 ./print_argument First &
srun --ntasks=1 ./print_argument Second &
srun --ntasks=1 ./print_argument Third &
wait # this command will block execution until all previous started tasks are done
echo "Finished tasks 1,2,3"
srun --ntasks=1 ./print_argument Fourth &
srun --ntasks=1 ./print_argument Fifth &
srun --ntasks=1 ./print_argument Sixth &
wait
echo "Finished tasks 4,5,6"
```

Then, we dispatch this script to TRUBA for execution using the command:

```bash
$ sbatch  sbatch_example3.slurm
```

One possible output text file is the following:

```
My argument is Second and I started
My argument is Third and I started
My argument is First and I started
My argument is Second and I finished
My argument is Third and I finished
My argument is First and I finished
Finished tasks 1,2,3
My argument is Fourth and I started
My argument is Sixth and I started
My argument is Fifth and I started
My argument is Fourth and I finished
My argument is Sixth and I finished
My argument is Fifth and I finished
Finished tasks 4,5,6
```

### Executing multiple tasks in parallel and dedicating more than one server (node) for their execution

We can instruct SLURM to dedicate more than one server for the execution of our tasks in a SLURM script. This is done by adding the `--nodes` option to the SLURM script options. This is demonstrated in the following SLURM script `sbatch4.slurm`:

```bash
#!/bin/bash

#SBATCH --account=<my_account>
#SBATCH --job-name=<job_name>
#SBATCH --partition=<part>
#SBATCH --time=d-hh:mm:ss
#SBATCH --nodes=<N>
#SBATCH --ntasks=<n> # maximum number of tasks that can run in parallel
#SBATCH --cpus-per-task=<cpus>
#SBATCH --workdir=<dir>
#SBATCH --output=<out>
#SBATCH --error=<err>

srun --ntasks=<n1> ./my_program1 &
srun --ntasks=<n2> ./my_program2 &
srun --ntasks=<n3> ./my_program3 &
wait
```

Then, we dispatch this script to TRUBA for execution using the command:

```bash
$ sbatch sbatch4.slrum
```

`<my_account>`: account name on TRUBA

`<job_name>`: the name of the dispatched job that appears on the job queue.

`<part>`: the name of the partition on which to enqueue the work.

`<time>`: the *maximum* amount of time that your work will be running for. The format of this input is `d-hh:mm:ss` where `d` is the number of days, `hh` is the number of hours, `mm`is the number of minutes, and `ss` is the number of seconds. **Note:** if the executable does not terminate within this specified time window, **it will be terminated automatically.** 

`<N>`: the number of nodes (servers) we use to run the tasks in this script.

`<n>`: the maximum number of tasks that will run in parallel within the script

`<n1,2,3>`: the number of parallel tasks that will run `my_program1`, `my_program2`, and `my_program3`, respectively.  It must hold that `<n1> + <n2> + <n3>` ≤ `<n>`

`<dir>`: the path in TRUBA where the script will execute. This is where the input and output files are usually located. All the relative paths defined in the script will be relative to `<out>`

`<out>`: the file to which the `stdout` of this job is going to be printed. This includes all the outputs that the executions in the script generate.

`<err>`: the file to which the `stderr` of this job is going to be printed. 

When we call `sbatch`, we will enqueue the job into the TRUBA queue. Once resources are available and our job is at the top of the queue, the following will occur:

1. the requested resources will be allocated for the requested time period, and in this case we requested:
    1. `<N>` nodes (servers)
    2. The ability to execute `<n>` tasks in parallel
    3. `<n>` CPUs (TRUBA assumes each task will require 1 CPU)
2.  the lines starting with `srun` will start job-steps that will run the programs `my_program1, my_program2`, and `my_program3`. However, these job-steps will run *in parallel* since we added the `&` symbol at the end of their line.
3. The script will block execution at the `wait` command until all started job-steps are done executing.

**Example:**

The script`sbatch_example4.slurm` defines a job named `multi_server_job` that will execute multiple the program `hostname`which will print the name of the host that is currently running the task. This job is going to be added to the partition `short` and it will finish within 20 minutes of execution. The file `print_argument.cpp` is located in `/truba/home/my_account/`. The outputs from executing the jobs will be printed to the file `/truba/home/my_account/output.txt` and the errors will be printed to `/truba/home/my_account/error.txt`

```bash
#!/bin/bash

#SBATCH --account=my_account
#SBATCH --job-name=multi_server_job
#SBATCH --partition=short
#SBATCH --time=0-00:20:00
#SBATCH --nodes=2
#SBATCH --ntasks=3
#SBATCH --workdir=/truba/home/my_account/
#SBATCH --output=output.txt
#SBATCH --error=error.txt

srun --ntasks=3 hostname &
```

Then, we dispatch this script to TRUBA for execution using the command:

```bash
$ sbatch sbatch_example4.slurm
```

One possible output text file is the following:

```
sardalya117.yonetim
sardalya117.yonetim
sardalya118.yonetim
```