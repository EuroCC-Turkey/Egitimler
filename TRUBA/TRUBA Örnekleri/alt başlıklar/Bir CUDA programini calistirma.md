# Bir CUDA programını çalıştırma

TRUBA'da bilimsel işlem için kullanılabilecek GPU'larla donatılmış birçok sunucu vardır. Bu GPU'ları TRUBA betiklerimizde kolayca kullanabiliriz. Aşağıdaki TRUBA örneğinde, bir veya daha fazla GPU kullanarak PI değerini hesaplayan aşağıdaki CUDA kodunu `cuda_example.cu` kullanacağız.

```jsx
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

### Bir veya birden fazla GPU kullanan program çalıştırma

Bir veya daha fazla GPU kullanan işleri, ihtiyaç duyduğumuz GPU sayısını talep ederek ve programımızı derlemek ve yürütmek için gerekli tüm modülleri yükleyerek TRUBA'da çalıştırabiliriz. Aşağıda gösterilen SLURM komut dosyası `usbatch_cuda.slurm`, bu işlemi gösterir:

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

`<my_account>`: TRUBA'daki hesap adı

`<job_name>`: iş kuyruğunda görünen gönderilen işin adı.

`<part>`: çalışmayı sıraya alacağınız bölümün adı.

`<time>`: Çalışmanızın çalışacağı maksimum süre. Bu girdinin biçimi `d-hh: mm: ss` şeklindedir, burada `d` günü, `hh` saati, `mm` dakikayı ve `ss` saniyeyi temsil eder. **Not:** Yürütülebilir dosya belirtilen bu zaman aralığında sona ermezse, otomatik olarak sonlandırılacaktır.

`<N>`: bu komut dosyasındaki görevleri çalıştırmak için kullanılacak düğüm (sunucu) sayısı.

`<gpu>`: her düğüm için ayrılacak GPU sayısı.

`<n>`: komut dosyası içinde paralel olarak çalışacak maksimum görev sayısı.

`<n1>`: ilgili MPI iş adımına katkıda bulunacak görev sayısı, `<n1> <= <n>`.

`<c>`: her görevin yürütülmesi için ayrılmış CPU sayısı.

`<c1>`: her görevin yürütülmesi için ayrılmış CPU sayısı, `<c1> <= <c>` .

`<dir>`: TRUBA'da komut dosyasının yürütüleceği yol. Burası genellikle girdi ve çıktı dosyalarının bulunduğu yerdir. Komut dosyasında tanımlanan tüm göreli yollar `<out>` ile göreli olacaktır.

`<out>`: bu işin `stdout` unun yazdırılacağı dosya. Bu, koddaki yürütmelerin ürettiği tüm çıktıları içerir.

`<err>`: bu işin `stderr` inin yazdırılacağı dosya.

`Sbatch` komutunu çağırdığımızda, işi TRUBA kuyruğuna kaydedeceğiz. Kaynaklar mevcut olduğunda ve işimiz sıranın en üstünde olduğunda, aşağıdakiler gerçekleşecektir:

1.  Talep edilen kaynaklar, talep edilen zaman aralığı için tahsis edilecektir ve bu durumda talep ettiklerimiz:
    1. `<N>` düğüm
    2. Her düğümde `<gpu>` GPU.
    3. `<n>` görevi parallel çalıştırma yetkisi.
    4. her görev için `<c>` işemci, yani totalde `<c> * <n>` işlemci
2. Doğru modülleri yükleyerek ve kodu derleyerek programı çalıştıracak ortamı hazırlıyoruz.
3. `srun` ile başlayan satır, `<n1>` paralel görevleri yürütmek için `my_cuda_program` programını çalıştıracak bir iş adımını başlatacaktır. Her iş adımı, görev başına belirtildiği kadar CPU kullanır.

### Örnek:

Aşağıdaki komut dosyası, yani `sbatch_cuda_example.slurm`, iki GPU kullanan bir işin yanı sıra OMP kullanarak çoklu iş parçacığı çalıştıracaktır. Öncelikle kodumuzu derlemek ve çalıştırmak için gerekli modülleri yükleyerek çalışma ortamını kuruyoruz. Ardından kodu derliyoruz (kodumuzu OMP kütüphanesine de bağladığımızdan emin oluyoruz). Son olarak, her biri tek bir düğümde tek bir görevi yürütecek ve 16 CPU kullanacak iki iş adımı yürütüyoruz (`--ntasks`, `--nodes` ve `--cpus-per-task` için değerler  belirtilmemiştir. Ancak iş adımları, hiçbir seçenek belirtilmediğinde varsayılan olarak SLURM bash betiğine verilen değeri kullanır)

Bu iş `short` bölümüne eklenecek ve 20 dakika içinde bitecektir. `cuda_mg.cu` dosyası `/truba/home/my_account/` konumunda bulunur. İşlerin çıktıları `/truba/home/my_account/output.txt` dosyasına ve hataları `/truba/home/my_account/error.txt` dosyasına yazdırılacaktır.

```bash
#!/bin/bash

#SBATCH --account=aalabsialjundi
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

Ardından, bu komut dosyasını yürütmek üzere TRUBA'ya aşağıdaki komutu kullanarak göndeririz:

```bash
$ sbatch cuda_example.slurm
```

Bu SLURM komut dosyasını gönderirken gözlemlediğimiz bir çıktı:

```
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