# Running an interactive shell script

Generally, when working with TRUBA, the user would prepare their work and send it to the SLURM system to be executed. This is different from a typical workflow in which work is done interactively (execute some code, observe results, execute another operation, modify execution and execute, etc.) TRUBA provides a method to run work interactively. It allows users to start a shell terminal on which work can be dispatched and results can be observed directly.

### Creating an interactive shell

The following is the format of the command we would use to create an interactive job:

```
$ srun --partition=interactive --time=d-hh-mm-ss /bin/bash
```

`<time>`: the *maximum* amount of time that your work will be running for. The format of this input is `d-hh:mm:ss` where `d` is the number of days, `hh` is the number of hours, `mm`is the number of minutes, and `ss` is the number of seconds. **Note:** if the executable does not terminate within this specified time window, **it will be terminated automatically.** 

**Example:**

In this example, we create an interactive job and then we execute a few commands on the device that was allocated by the interactive job. Please note that the lines starting with `$` were written by the user, and other lines are produced outputs:

```
$ srun --partition=interactive --time=0-00:10:00 /bin/bash
srun: job 6529835 queued and waiting for resourcesve
srun: job 6529835 has been allocated resourcesctive
$ pwd
/truba/home/<my_account>/
$ hostname
levrek165.yonetim
```