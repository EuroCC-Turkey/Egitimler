# Bir OMP + MPI programını çalıştırma

Hem OMP çok iplikli hem de MPI çok düğümlü programlama özelliklerinden yararlanan programlar oluşturabiliriz. Örnekte, Pi'nin değerini hesaplayan aşağıdaki  `omp_mpi_example.c`  programını kullanacağız. Bu program birden çok MPI görevi (birden çok düğüm) kullanabilir ve her görevde çoklu iş parçacığı kullanır.

```c
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

## OMP çoklu iş parçacığı kullanan bir MPI programını yürütme

TRUBA, aynı programda hem MPI hem de OpenMP'nin kullanılmasını sağlar. MPI, birden çok düğümün aynı prosedür üzerinde birlikte çalışmasına izin verirken, OpenMP, katkıda bulunan her düğümün birden çok iş parçacığı kullanmasına izin verir. Bir MPI programıyla çok iş parçacıklı bir uygulamayı yürütmek için, önce SLURM bash betiği için `--cpus-per-task` değerini ayarlayarak her MPI işleminin alacağı CPU sayısını belirlememiz gerekir. Ayrıca, MPI programını çalıştıracak iş adımını oluşturan `mpirun` çağrısında `--bind-to none` seçeneğini eklememiz gerekir. Aşağıda gösterilen `mpi_omp.slurm` komut dosyası bu işlemi göstermektedir:

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

Ardından, bu komut dosyasını yürütmek üzere TRUBA'ya aşağıdaki komutu kullanarak göndeririz:

```bash
$ sbatch example_mt1.slurm
```

`<my_account>`: TRUBA'daki hesap adı

`<job_name>`: iş kuyruğunda görünen gönderilen işin adı.

`<part>`: çalışmayı sıraya alacağınız bölümün adı.

`<time>`: Çalışmanızın çalışacağı maksimum süre. Bu girdinin biçimi `d-hh: mm: ss` şeklindedir, burada `d` günü, `hh` saati, `mm` dakikayı ve `ss` saniyeyi temsil eder. Not: Yürütülebilir dosya belirtilen bu zaman aralığında sona ermezse, otomatik olarak sonlandırılacaktır.

`<N>`: bu komut dosyasındaki görevleri çalıştırmak için kullanılacak düğüm (sunucu) sayısı.

`<n>`: komut dosyası içinde paralel olarak çalışacak maksimum görev sayısı.

`<n1>`: ilgili MPI iş adımına katkıda bulunacak görev sayısı

`<c>`: her görevin yürütülmesi için ayrılmış CPU sayısı

`<dir>`: TRUBA'da komut dosyasının yürütüleceği yol. Burası genellikle girdi ve çıktı dosyalarının bulunduğu yerdir. Komut dosyasında tanımlanan tüm göreli yollar `<out>` ile göreli olacaktır.

`<out>`: bu işin `stdout` unun yazdırılacağı dosya. Bu, koddaki yürütmelerin ürettiği tüm çıktıları içerir.

`<err>`: bu işin `stderr` inin yazdırılacağı dosya.

`Sbatch` komutunu çağırdığımızda, işi TRUBA kuyruğuna kaydedeceğiz. Kaynaklar mevcut olduğunda ve işimiz sıranın en üstünde olduğunda, aşağıdakiler gerçekleşecektir:

1.  Talep edilen kaynaklar, talep edilen zaman aralığı için tahsis edilecektir ve bu durumda talep ettiklerimiz:
    1. `<N>` düğüm
    2. `<npn> * <N>` görev yürütme yetkisi
    3. her görev için `<c>` işemci, yani totalde `<c> * (<npn> * <N>)` işlemci
2. the lines starting with `mpirun` will start job-steps that will  run the program my_mpi_omp_program using `<N>` nodes. The first job-step will use `<n1>` tasks to run its procedure. The second will use `<n>` tasks.  Both of these job-steps' tasks will use `<c>` threads. 
3. `mpirun` ile başlayan satırlar, my_mpi_omp_program programını `<N>` düğüm kullanarak çalıştıracak iş adımlarını başlatacaktır. İlk iş adımı, prosedürünü çalıştırmak için `<n1>` görev kullanacaktır. İkincisi, `<n>` görev kullanacaktır. Bu iş adımlarının her iki görevi de `<c>` iş parçacığı kullanacaktır.

Örnek**:**

Aşağıdaki `sbatch_omp_mpi_example.slurm` komut dosyası, OMP'nin birden çok düğüm kullanan bir MPI işi ile kullanımını göstermektedir. Her düğümün 2 görevi olacak ve her görev 4 iş parçacığı kullanacak. Komut dosyasını, kullanılacak düğüm sayısı, her düğümde çalışan görev sayısı ve her görevin kullanacağı CPU sayısı dahil olmak üzere iş için gerekli seçenekleri ayarlayarak başlatıyoruz. Daha sonra gerekli modülleri (`gcc` ve `openmpi`) yükleyerek ve yürütmede kullanılacak kodu derleyerek çalışma ortamını kuruyoruz. Daha sonra, üç iş adımı gerçekleştiriyoruz. Her iş adımı, birden çok görevi yürütür ve aynı MPI iş adımındaki tüm görevler aynı prosedür üzerinde birlikte çalışır. Her iş adımı, `OMP_NUM_THREADS` ortam değişkenini ayarlayarak belirlediğimiz farklı sayıda iş parçacığı kullanacaktır.

Bu iş `short` bölümüne eklenecek ve 20 dakika içinde bitecek. `omp_mpi_example.c` dosyası `/truba/home/my_account/` konumunda bulunur. İşlerin çıktıları `/truba/home/my_account/output.txt` dosyasına yazdırılacak ve hatalar `/truba/home/my_account/error.txt` dosyasına yazdırılacaktır. Lütfen iş adımlarına verilen işlemci sayısının `--cpus-per-task` bayrağı ve `--exclusive` seçeneğiyle nasıl değiştiğine dikkat edin.

```bash
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
```

Ardından, bu komut dosyasını yürütmek üzere TRUBA'ya aşağıdaki komutu kullanarak göndeririz:

```bash
$ sbatch mpi_omp_example.slurm
```

Bu SLURM komut dosyasını gönderirken gözlemlediğimiz bir çıktı:

```
Using a single logical thread per task
Used 4 tasks and 1 threads per task and took 7.506095 seconds 

Using 2 logical threads per task
Used 4 tasks and 2 threads per task and took 3.905180 seconds 

Using 4 logical threads per task
Used 4 tasks and 4 threads per task and took 2.834785 seconds
```