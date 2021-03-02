# Çok iplikli programları çalıştırma

TRUBA üzerinde çok iş parçacıklı kütüphanelerden yararlanan programları çalıştırabiliriz. Bunu yapmak, görevlerimiz için kullanmak istediğimiz düğümleri ve CPU'ları belirleyen bazı yapılandırma seçeneklerinin eklenmesini gerektirir. TRUBA bu tür sayısız seçeneği içerir. Bu örneklerde en önemli olanlardan bazılarını ele alacağız.

Aşağıdaki TRUBA örneklerinde, aşağıda gösterilen `omp_example.cpp` dosyasındaki kodu kullanacağız:

```c
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

### Çok iş parçacıklı bir görevi yürütme ve görev için birden çok CPU tahsis etme

SLURM ile görev başına tek bir CPU kullanmak yerine, görevler için birden fazla CPU tahsis etmesi talimatını verebiliriz. Bu, SLURM bash komut dosyası seçeneklerine `--cpus-per-task` seçeneği eklenerek ve bu bayrağı iş adımı yürütmelerine, yani `srun` komutlarına eklenerek yapılır. Bu, aşağıdaki `sbatch_mt1.slurm` komut dosyasında gösterilir:

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

Ardından, bu komut dosyasını yürütmek üzere TRUBA'ya aşağıdaki komutu kullanarak göndeririz:

```bash
$ sbatch sbatch_mt1.slurm
```

`<my_account>`: TRUBA'daki hesap adı

`<job_name>`: iş kuyruğunda görünen gönderilen işin adı.

`<part>`: çalışmayı sıraya alacağınız bölümün adı.

`<time>`: Çalışmanızın çalışacağı maksimum süre. Bu girdinin biçimi `d-hh: mm: ss` şeklindedir, burada `d` günü, `hh` saati, `mm` dakikayı ve `ss` saniyeyi temsil eder. **Not:** Yürütülebilir dosya belirtilen bu zaman aralığında sona ermezse, otomatik olarak sonlandırılacaktır.

`<N>`: bu komut dosyasındaki görevleri çalıştırmak için kullanılacak düğüm (sunucu) sayısı.

`<n>`: komut dosyası içinde paralel olarak çalışacak maksimum görev sayısı.

`<c>`: her görevin yürütülmesi için ayrılmış CPU sayısı.

`<c1,2>`: iki iş adımının her birinde her bir görevin yürütülmesi için ayrılmış CPU sayısı. `<c1,2> <= <c>` tutmalıdır.

`<dir>`: TRUBA'da komut dosyasının yürütüleceği yol. Burası genellikle girdi ve çıktı dosyalarının bulunduğu yerdir. Komut dosyasında tanımlanan tüm göreli yollar `<out>` ile göreli olacaktır.

`<out>`: bu işin `stdout` unun yazdırılacağı dosya. Bu, koddaki yürütmelerin ürettiği tüm çıktıları içerir.

`<err>`: bu işin `stderr` inin yazdırılacağı dosya.

`Sbatch` komutunu çağırdığımızda, işi TRUBA kuyruğuna kaydedeceğiz. Kaynaklar mevcut olduğunda ve işimiz sıranın en üstünde olduğunda, aşağıdakiler gerçekleşecektir:

1. Talep edilen kaynaklar, talep edilen zaman aralığı için tahsis edilecektir ve bu durumda talep ettiklerimiz:
    1. `<N>` düğüm
    2. `<n>` görevi parallel çalıştırma yetkisi.
    3. her görev için `<c>` işemci, yani totalde `<c> * <n>` işlemci
2. `srun` ile başlayan satırlar, my_mt_program programını `<n>` paralel görevlerde çalıştıracak iş adımlarını başlatacaktır. İlk iş adımı her görev için `<c1>` CPU'ları kullanır ve ikinci iş adımı her görev için `<c2>` CPU kullanır.

### Örnek:

`sbatch_mt_example1.slurm` betiği, `omp_example.cpp` programını `omp_example` olarak derler ve `multi_threaded_job` adlı bir işi tanımlar. Daha sonra `omp_example`'ı 4 farklı CPU ataması ile çalıştıracaktır. Her seferinde `OMP_NUM_THREADS` ortam değişkeni kullanılarak atanan mantıksal iş parçacığı sayısını ve ayrıca `--cpus-task-task` bayrağı kullanılarak atanan donanım CPU sayısı değiştirerilir. Bu iş `short` bölümüne eklenecek ve 20 dakika içinde bitecektir. `omp_example.cpp` dosyası `/truba/home/my_account/` konumunda bulunur. İşlerin `/truba/home/my_account/output.txt` dosyasına ve hatalar `/truba/home/my_account/error.txt` dosyasına yazdırılacaktır. Lütfen iş adımlarına verilen CPU sayısının `--cpus-per-task` bayrağı ve `--exclusive` seçeneğiyle nasıl değiştiğine dikkat edin.

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

Ardından, bu komut dosyasını yürütmek üzere TRUBA'ya aşağıdaki komutu kullanarak göndeririz:

```bash
$ sbatch sbatch_mt_example1.slurm
```

Aşağıdaki, output.txt dosyasında gözlemlediğimiz bir çıktıdır:

```
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

### Birden çok iş parçacıklı görevi paralel olarak yürütme

Birden çok iş parçacıklı görevi aynı anda yürütebilir ve CPU'ları nasıl paylaştıklarını kontrol edebiliriz. Bu, SLURM bash komut dosyası seçeneklerindeki görevlerin sayısını artırarak ve bu görevleri birden çok eşzamanlı iş adımına tahsis ederek yapılır. Gösterilen `sbatch_mt2.slurm` komut dosyası bunu göstermektedir:

```bash
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
```

Ardından, bu komut dosyasını yürütmek üzere TRUBA'ya aşağıdaki komutu kullanarak göndeririz:

```bash
$ sbatch sbatch_mt2.slurm
```

`<my_account>`: TRUBA'daki hesap adı

`<job_name>`: iş kuyruğunda görünen gönderilen işin adı.

`<part>`: çalışmayı sıraya alacağınız bölümün adı.

`<time>`: Çalışmanızın çalışacağı maksimum süre. Bu girdinin biçimi `d-hh: mm: ss` şeklindedir, burada `d` günü, `hh` saati, `mm` dakikayı ve `ss` saniyeyi temsil eder. **Not:** Yürütülebilir dosya belirtilen bu zaman aralığında sona ermezse, otomatik olarak sonlandırılacaktır.

`<N>`: bu komut dosyasındaki görevleri çalıştırmak için kullanılacak düğüm (sunucu) sayısı.

`<n>`: komut dosyası içinde paralel olarak çalışacak maksimum görev sayısı.

`<n1,2,3,4>`: ilgili iş adımı tarafından yürütülecek görev sayısı, burada `<n>, <n1>, <n2>, <n3>`.

`<c>`: her görevin yürütülmesi için ayrılmış CPU sayısı.

`<c1,2,3,4>`: `<c1>, <c2>, <c3> <c>` her görevin yürütülmesi için ayrılmış CPU sayısı.

`<dir>`: TRUBA'da komut dosyasının yürütüleceği yol. Burası genellikle girdi ve çıktı dosyalarının bulunduğu yerdir. Komut dosyasında tanımlanan tüm göreli yollar `<out>` ile göreli olacaktır.

`<out>`: bu işin `stdout` unun yazdırılacağı dosya. Bu, koddaki yürütmelerin ürettiği tüm çıktıları içerir.

`<err>`: bu işin `stderr` inin yazdırılacağı dosya.

`Sbatch` komutunu çağırdığımızda, işi TRUBA kuyruğuna kaydedeceğiz. Kaynaklar mevcut olduğunda ve işimiz sıranın en üstünde olduğunda, aşağıdakiler gerçekleşecektir:

1. Talep edilen kaynaklar, talep edilen zaman aralığı için tahsis edilecektir ve bu durumda talep ettiklerimiz:
    1. `<N>` düğüm
    2. `<n>` görevi parallel çalıştırma yetkisi.
    3. her görev için `<c>` işemci, yani totalde `<c> * <n>` işlemci
2. `srun` ile başlayan satırlar, `<n1,2,3,4>` paralel görevlerde my_mt_program programını çalıştıracak iş adımlarını başlatacaktır. İlk iş adımı her görev için `<c1>` CPU'ları kullanır ve ikinci iş adımı her görev için `<c2>` CPU'ları kullanır vb. İlk iş adımının görevleri için kullanılan CPU'ların o iş adımındaki tüm görevler arasında paylaşıldığına dikkat edin. Ancak, aşağıdaki üç iş adımı için, her biri bağımsız CPU'lar kullanacaktır. Bunun nedeni, hepsinin ayrı iş adımları olarak yaratılmış olmaları ve `--exclusive` bayrağını kullanmamızdır.
3. Betik, başlatılan tüm iş adımlarının yürütülmesi tamamlanana kadar `wait` komutunda yürütmeyi engelleyecektir.

### Örnek:

`sbatch_mt_example2.slurm` betiği `multi_threaded_job` adlı bir işi tanımlar ve `omp_example.cpp` programını `omp_example` olarak derler. Ardından, `omp_example`'ı tek bir iş adımında (`srun`) üç kez çalıştıracaktır. Her göreve `--cpus-task-görev` bayrağı kullanılarak 2 CPU atanır. Bu, bu iş adımının 6 CPU kullanacağı anlamına gelir. İş adımında çalışan tüm görevlerin aynı 6 CPU'yu paylaşacağına dikkat edin. Daha sonra aynı üç görev yürütülür, ancak bu sefer her biri kendi iş adımında olacaktır. `Srun` komutunun sonundaki `&` seçeneğini kullanarak hepsinin paralel çalıştığından emin oluruz. Her birinin farklı bir CPU seti kullandığına dikkat edin.

Bu iş `short` bölümüne eklenecek ve 20 dakika içinde bitecektir. `omp_example.cpp` dosyası `/truba/home/my_account/` konumunda bulunur. İşlerin çıktıları `/truba/home/my_account/output.txt` dosyasına ve hatalar `/truba/home/my_account/error.txt` dosyasına yazdırılacaktır. Lütfen iş adımlarına verilen CPU sayısının `--cpus-per-task bayrağı` ve `--exclusive` seçeneğiyle nasıl değiştiğine dikkat edin.

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

Ardından, bu komut dosyasını yürütmek üzere TRUBA'ya aşağıdaki komutu kullanarak göndeririz:

```bash
$ sbatch sbatch_mt_example2.slurm
```

Aşağıdaki, output.txt dosyasında gözlemlediğimiz bir çıktıdır:

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

### Birden çok sunucuda birden çok iş parçacıklı görev yürütme

TRUBA'ya bir iş içindeki görevleri yürütmek için birden fazla sunucu kullanması talimatını verebiliriz. Bunu SLURM toplu iş seçeneklerinde `--nodes` seçeneğini kullanarak yapıyoruz. Ayrıca `--ntasks-per-node` bayrağını kullanarak her sunucuda çalıştırmak istediğimiz görev sayısını belirleyebiliriz. Lütfen SLURM toplu komut dosyası seçeneği `#SBATCH --ntasks`'ın örtük olarak eklendiğini unutmayın (`--nodes` ve `--ntasks-per-node` değerlerine göre otomatik olarak eklenir). Aşağıdaki dosya olan `sbatch_mt3.slurm`, birden çok düğüm kullanımını gösterir:

```bash
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
```

Ardından, bu komut dosyasını yürütmek üzere TRUBA'ya aşağıdaki komutu kullanarak göndeririz:

```bash
$ sbatch sbatch_mt3.slurm
```

`<my_account>`: TRUBA'daki hesap adı

`<job_name>`: iş kuyruğunda görünen gönderilen işin adı.

`<part>`: çalışmayı sıraya alacağınız bölümün adı.

`<time>`: Çalışmanızın çalışacağı maksimum süre. Bu girdinin biçimi `d-hh: mm: ss` şeklindedir, burada `d` günü, `hh` saati, `mm` dakikayı ve `ss` saniyeyi temsil eder. **Not:** Yürütülebilir dosya belirtilen bu zaman aralığında sona ermezse, otomatik olarak sonlandırılacaktır.

`<N>`: bu komut dosyasındaki görevleri çalıştırmak için kullanılacak düğüm (sunucu) sayısı.

`<n>`: komut dosyası içinde paralel olarak çalışacak maksimum görev sayısı.

`<n1,2,3,4>`: ilgili iş adımı tarafından yürütülecek görev sayısı, burada `<n>, <n1>, <n2>, <n3>`.

`<c>`: her görevin yürütülmesi için ayrılmış CPU sayısı.

`<c1,2,3,4>`: `<c1>, <c2>, <c3> <c>` her görevin yürütülmesi için ayrılmış CPU sayısı.

`<dir>`: TRUBA'da komut dosyasının yürütüleceği yol. Burası genellikle girdi ve çıktı dosyalarının bulunduğu yerdir. Komut dosyasında tanımlanan tüm göreli yollar `<out>` ile göreli olacaktır.

`<out>`: bu işin `stdout` unun yazdırılacağı dosya. Bu, koddaki yürütmelerin ürettiği tüm çıktıları içerir.

`<err>`: bu işin `stderr` inin yazdırılacağı dosya.

`Sbatch` komutunu çağırdığımızda, işi TRUBA kuyruğuna kaydedeceğiz. Kaynaklar mevcut olduğunda ve işimiz sıranın en üstünde olduğunda, aşağıdakiler gerçekleşecektir:

1. Talep edilen kaynaklar, talep edilen zaman aralığı için tahsis edilecektir ve bu durumda talep ettiklerimiz:
    1. `<N>` düğüm
    2. `<npn> * <N>`  görevi parallel çalıştırma yetkisi.
    3. her görev için `<c>` işemci, yani totalde `<c> * (<npn> * <N>)` işlemci
2. `srun` ile başlayan satırlar, my_mt_program programını `<N1>`, `<N2>` ve `<N>` düğümlerini kullanarak çalıştıracak ve her biri için `<n1>`, `<n2>` ve `<n>` paralel görevleri yürütecek iş adımlarını başlatacaktır. Her iş adımı, görev başına belirtildiği kadar CPU kullanır ve düğüm başına belirtilen `—ntasks-per-code` bayrağı düğümler arasında çalıştırılan görevlerin sayısını dengelemeye hizmet eder.

### Örnek:

Aşağıda gösterilen `sbatch_mt_example3.slurm` betiği `--ntasks-per-node`'un yürütmeyi nasıl etkilediğini gösterir. Tüm SLURM bash betiğine bir seçenek olarak verildiğinde (dosyanın başlangıcında), değeri, betikte çalıştırılabilecek toplam görev sayısını hesaplamak için kullanılır ve bu değer, aşağıda gösterildiği gibi koda eklenir. . Yürütme sırasında kullanılacak programı derledikten sonra, 2 düğümde 8 görevi yürütecek bir iş adımı oluşturarak başlıyoruz. Betiğe verilen `--ntasks-per-node` seçeneği nedeniyle, bu iş adımı her düğümde 4'ten fazla görev yürütmeyecektir. Daha sonra, bir görevi iki düğümde 4 kez yürütecek bir iş adımı çalıştırıyoruz ve her düğümde 2 görev çalıştırması talimatını veriyoruz. Son olarak, bir iş adımı oluşturuyoruz ve düğüm başına görev, düğüm veya görev sayısını belirtmiyoruz. Bu, tüm SLURM bash betiğine verilen varsayılan değerleri kullanan iş adımına yol açar, yani `--nodes = 3`, `--ntasks = 4` ve `--ntasks-per-node = 2`.

Bu iş `short` bölümüne eklenecek ve 20 dakika içinde bitecektir. `omp_example.cpp` dosyası `/truba/home/my_account/` konumunda bulunur. İşlerin çıktıları `/truba/home/my_account/output.txt` dosyasına ve hatalar `/truba/home/my_account/error.txt` dosyasına yazdırılacaktır. Lütfen iş adımlarına verilen CPU sayısının `--cpus-per-task bayrağı` ve `--exclusive` seçeneğiyle nasıl değiştiğine dikkat edin.

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

Ardından, bu komut dosyasını yürütmek üzere TRUBA'ya aşağıdaki komutu kullanarak göndeririz:

```bash
$ sbatch sbatch_mt_example3.slurm
```

Aşağıdaki, output.txt dosyasında gözlemlediğimiz bir çıktıdır:

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