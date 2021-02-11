# Running a single program using srun

If we wish to execute a single program on TRUBA, we can do so in a single line using the `srun` command. This command provides a robust interface and enables a large degree of configuration. However, it should be noted that using `srun` will block the executing terminal. This means that after executing the command, you must wait for the completion of the execution before using the terminal again. Calling `srun` will both create a job (allocate resources) and execute a job-step (use the allocated resources to run some code) in a single command. If you wish to allocate the job and execute multiple job-steps on the same allocated resource, please refer to the examples on using `sbatch`.

### Execute a single program

To run the executable `my_program` located in the current directory, we can use the following command:

```bash
$ srun --partition=<partition> --time=<time> ./my_program
```

`<partition>`:  the name of the partition that you wish the job to be enqueued to.

`<time>`: the *maximum* amount of time that your work will be running for. The format of this input is `d-hh:mm:ss` where `d` is the number of days, `hh` is the number of hours, `mm`is the number of minutes, and `ss` is the number of seconds. **Note:** if the executable does not terminate within this specified time window, **it will be terminated automatically.** 

This command will,

a) create a job, i.e. allocate a node (server) for the amount of time specified, and 

b) execute a job-step (code execution) that will run `my_program`.

**Example**:

 the following command will enqueue `my_program` to the `short` partition and will give it a maximum time limit of 20 minutes.

```bash
$ srun --partition=short --time=0-00:20:00 ./my_program
```

### Execute a single program and dedicate for its execution multiple CPUs

To run the executable `my_program` that can use multiple CPUs, we use the following command:

  

```bash
$ srun --partition=<partition> --time=<time> --cpus-per-task=<cpus> ./my_program
```

`<cpus>`: the number of CPUs dedicate for the execution.

`<partition>`:  the name of the partition that you wish the job to be enqueued to.

`<time>`: the *maximum* amount of time that your work will be running for. The format of this input is `d-hh:mm:ss` where `d` is the number of days, `hh` is the number of hours, `mm`is the number of minutes, and `ss` is the number of seconds. **Note:** if the executable does not terminate within this specified time window, **it will be terminated automatically.** 

This command will,

a) create a job, i.e. allocate a node (server) for the amount of time specified, as well as allocate `<cpus>` number of CPUs on the node, and 

b) execute a job-step (code execution) that will run `my_program` using the allocated CPUs.

**Example:** 

the following command will execution `my_program` on the partition `short`, will finish within 20 minutes, and will dedicate for this execution 8 CPUs:

```bash
$ srun --partition=short --time=0-00:20:00 --cpus-per-task=8 ./my_program
```

### Execute a single program multiple times in parallel

To run the executable `my_program` located in the current directory *multiple times in parallel* (for averaging purposes, for example), we use the following command:

```bash
$ srun --partition=<partition> --time=<time> --ntasks=<n> ./my_program
```

`<partition>`:  the name of the partition that you wish the job to be enqueued to.

`<time>`: the *maximum* amount of time that your work will be running for. The format of this input is `d-hh:mm:ss` where `d` is the number of days, `hh` is the number of hours, `mm`is the number of minutes, and `ss` is the number of seconds. **Note:** if the executable does not terminate within this specified time window, **it will be terminated automatically.** 

`<n>`: is the number of parallel executions of the program required. 

This command will, 

a) create a job, i.e. allocate a node (server) for the amount of time specified and request the ability to run `<n>` tasks, and 

b) execute a job-step (code execution) that will run `my_program` a total of `<n>` times. Each execution of `my_program` is called a "task".

**Example:** 

the following command will execute `my_program` 5 times in parallel and will use a somewhere between 2 and 4 servers to carry out these executions. It will use the `short` partition and will finish within 20 minutes.

```bash
$ srun --partition=short --time=0-00:20:00 --ntasks=5 --nodes=2-4 ./my_program
```

### Execute a single program multiple times in parallel, and allocate for each one of its execution multiple CPUs

To run the executable `my_program` located in the current directory *multiple times in parallel* and dedicate for each one of its executions multiple CPUSs,  we use the following command:

```bash
$ srun --partition=<partition> --time=<time> --ntasks=<n> --cpus-per-task=<cpus> ./my_program
```

`<partition>`:  the name of the partition that you wish the job to be enqueued to.

`<time>`: the *maximum* amount of time that your work will be running for. The format of this input is `d-hh:mm:ss` where `d` is the number of days, `hh` is the number of hours, `mm`is the number of minutes, and `ss` is the number of seconds. **Note:** if the executable does not terminate within this specified time window, **it will be terminated automatically.** 

`<n>`: is the number of parallel executions of the program required. 

`<cpus>`: the number of CPUs dedicate for each task execution.

This command will,

a) create a job, i.e. allocate a node (server) for the amount of time specified, and 

b) execute a job-step (code execution) that will run `my_program` a total of `<n>` times. Each execution of `my_program` is called a "task". Also, each "task" will be given `<cpus>` number of CPUs to carry out its work. In other words, the job-step that will run will use `<n> * <cpus>` number of CPUs.

**Example**:

 the following command will execute `my_program` 5 times in parallel. Each execution will use 8 CPUs. It will use the `short` partition and will finish within 20 minutes.

```bash
$ srun --partition=short --time=0-00:20:00 --ntasks=5 -cpus-per-task=8 ./my_program
```