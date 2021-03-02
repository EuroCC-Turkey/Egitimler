# Sbatch kullanarak TRUBA'da bir veya daha fazla programın komut dosyasını çalıştırma

TRUBA üzerinde programları yürütmek için komut dosyalarının kullanılması, kullanıcıların tek tek yürütmek ve her birinin bitmesini beklemek yerine, küme üzerinde çalıştırmak için birden fazla komut yürütmesine olanak tanır. Ayrıca, kullanıcılar işi gönderebilir ve TRUBA'dan çıkabilir ve iş arka planda otomatik olarak biter.

Komut dosyaları kullanılarak çalışma yürütülmesi, çalışma sırasında atılacak adımları içeren bir `bash` betiği tanımlanarak ve bu betiği TRUBA'ya göndermek için `sbatch` komutu kullanılarak yapılır. Bu işlem terminali engellemez ve tamamen arka plandadır.

Gelecek örneklerde, aşağıda gösterilen `print_argument.cpp` dosyasındaki kodun derlenmiş hali olan çalıştıran çalıştırılabilir `print_argument`'ı kullanacağız:

```c
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

### Tek bir görev yürütme

Yürütülebilir `my_program`'ı çalıştırmak için `sbatch1.slurm` adlı betiğini tanımlayabiliriz:

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

Ardından, bu komut dosyasını yürütmek üzere TRUBA'ya aşağıdaki komutu kullanarak göndeririz:

```bash
$ sbatch sbatch1.slurm
```

`<my_account>`: TRUBA'daki hesap adı

`<job_name>`: iş kuyruğunda görünen gönderilen işin adı.

`<part>`: çalışmayı sıraya alacağınız bölümün adı.

`<time>`: Çalışmanızın çalışacağı maksimum süre. Bu girdinin biçimi `d-hh: mm: ss` şeklindedir, burada `d` günü, `hh` saati, `mm` dakikayı ve `ss` saniyeyi temsil eder. **Not:** Yürütülebilir dosya belirtilen bu zaman aralığında sona ermezse, otomatik olarak sonlandırılacaktır.

`<dir>`: TRUBA'da komut dosyasının yürütüleceği yol. Burası genellikle girdi ve çıktı dosyalarının bulunduğu yerdir. Komut dosyasında tanımlanan tüm göreli yollar `<out>` ile göreli olacaktır.

`<out>`: bu işin `stdout` unun yazdırılacağı dosya. Bu, koddaki yürütmelerin ürettiği tüm çıktıları içerir.

`<err>`: bu işin `stderr` inin yazdırılacağı dosya.

`Sbatch` komutunu çağırdığımızda, işi TRUBA kuyruğuna kaydedeceğiz. Kaynaklar mevcut olduğunda ve işimiz sıranın en üstünde olduğunda, aşağıdakiler gerçekleşecektir:

1. Talep edilen kaynaklar, talep edilen zaman aralığı için tahsis edilecektir ve bu durumda talep ettiklerimiz:
    1. 1 düğüm
    2. `<n>` görevi parallel çalıştırma yetkisi.
    3. her görev için `<c>` işemci, yani totalde `<c> * <n>` işlemci
2. `srun` ile başlayan satır my_mt_program programını başlatan bir iş adımı yürütecektir.

### Örnek:

`sbatch_example1.slurm` betiği, `print_argument.cpp` içindeki kodu derler ve ardından derlenmiş `print_argument` programını üç kez çalıştıran `my_job` adlı bir işi tanımlar. Bu iş `short` bölümüne eklenecek ve 20 dakika içinde bitecektir. `print_argument.cpp` dosyası `/truba/home/my_account/` konumunda bulunur. İşlerin çıktıları `/truba/home/my_account/output.txt` dosyasına ve hatalar `/truba/home/my_account/error.txt` dosyasına yazdırılacaktır.

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

Sonra aşağıdaki komutu kullanarak betiği sıraya ekliyoruz

```bash
$ sbatch sbatch_example1.slurm
```

Bu adım sonrasında`output.txt`'nin içindekiler aşağıdaki gibi olur:

```bash
My argument is First and I started
My argument is First and I finished
My argument is Second and I started
My argument is Second and I finished
My argument is Third and I started
My argument is Third and I finished
```

### Tek bir iş adımında bir görevi birden çok kez paralel olarak yürütmek

İşte çalışacak maksimum paralel yürütme sayısını tanımlamak için SLURM bash betiği seçeneklerine `--ntasks` seçeneğini ekleyebiliriz. Aynı zamanda bu bayrağı programı çalıştıracak görevlerin sayısını belirtmek için iş adımına, yani `srun` komutuna da ekleyebiliriz. Bu, aşağıdaki `sbatch2.slurm` komut dosyasında gösterilir:

```bash
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
```

Sonra aşağıdaki komutu kullanarak betiği sıraya ekliyoruz

```bash
sbatch sbatch2.slurm
```

`<my_account>`: TRUBA'daki hesap adı

`<job_name>`: iş kuyruğunda görünen gönderilen işin adı.

`<part>`: çalışmayı sıraya alacağınız bölümün adı.

`<time>`: Çalışmanızın çalışacağı maksimum süre. Bu girdinin biçimi `d-hh: mm: ss` şeklindedir, burada `d` günü, `hh` saati, `mm` dakikayı ve `ss` saniyeyi temsil eder. **Not:** Yürütülebilir dosya belirtilen bu zaman aralığında sona ermezse, otomatik olarak sonlandırılacaktır.

`<n>`: komut dosyası içinde paralel olarak çalışacak maksimum görev sayısı.

`<n1,2>`: ilgili iş adımı tarafından yürütülecek görev sayısı, burada `<n1,2> <= <n>`.

`<dir>`: TRUBA'da komut dosyasının yürütüleceği yol. Burası genellikle girdi ve çıktı dosyalarının bulunduğu yerdir. Komut dosyasında tanımlanan tüm göreli yollar `<out>` ile göreli olacaktır.

`<out>`: bu işin `stdout` unun yazdırılacağı dosya. Bu, koddaki yürütmelerin ürettiği tüm çıktıları içerir.

`<err>`: bu işin `stderr` inin yazdırılacağı dosya.

`Sbatch` komutunu çağırdığımızda, işi TRUBA kuyruğuna kaydedeceğiz. Kaynaklar mevcut olduğunda ve işimiz sıranın en üstünde olduğunda, aşağıdakiler gerçekleşecektir:

1. Talep edilen kaynaklar, talep edilen zaman aralığı için tahsis edilecektir ve bu durumda talep ettiklerimiz:
    1. `<n>` görevi paralel olarak yürütme yetkisi
    2. `<n>` CPU
    3. Yeterli sayıda düğüm (Truba tarafından belirlenir.).
2. `srun` ile başlayan satırlar `my_program1`, `my_program2` ve `my_program3` iş adımlarını yürütecektir.

Örnek:

`sbatch_example2.slurm` betiği, `print_argument.cpp` içindeki kodu derler ve ardından derlenmiş `print_argument` programını ilk önce üç kez, sonra iki kez ve son olarak üç kez çalıştıran `my_job` adlı bir işi tanımlar. Bu iş `short` bölümüne eklenecek ve 20 dakika içinde bitecektir. `print_argument.cpp` dosyası `/truba/home/my_account/` konumunda bulunur. İşlerin çıktıları `/truba/home/my_account/output.txt` dosyasına ve hatalar `/truba/home/my_account/error.txt` dosyasına yazdırılacaktır.

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

Sonra aşağıdaki komutu kullanarak betiği sıraya ekliyoruz

```bash
$ sbatch sbatch_example2.slurm
```

Bu adım sonrasında`output.txt`'nin içindekiler aşağıdaki gibi olur:

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

### Birden çok farklı görevi paralel olarak yürütme (farklı iş adımlarında)

Her birini ayrı bir iş adımında, yani farklı `srun` komutunda çalıştırarak birden fazla programı paralel olarak çalıştırabilir ve `srun` komutunun sonuna `&` sembolünü ekleyebiliriz. Bu, komut dosyasına komutu arka plana taşımasını söyler. Arka planda iş adımlarını başlattıktan sonra, iş adımları bitmeden işin bitmediğinden emin olmak için bir bekleme(`wait`) komutu eklememiz gerektiğini lütfen unutmayın. Birden çok iş adımını paralel olarak yürütmenin bir örneği, aşağıdaki komut dosyası olan `sbatch3.slurm`'da gösterilmektedir:

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

Sonra aşağıdaki komutu kullanarak betiği sıraya ekliyoruz

```bash
$ sbatch sbatch3.slurm
```

`<my_account>`: TRUBA'daki hesap adı

`<job_name>`: iş kuyruğunda görünen gönderilen işin adı.

`<part>`: çalışmayı sıraya alacağınız bölümün adı.

`<time>`: Çalışmanızın çalışacağı maksimum süre. Bu girdinin biçimi `d-hh: mm: ss` şeklindedir, burada `d` günü, `hh` saati, `mm` dakikayı ve `ss` saniyeyi temsil eder. **Not:** Yürütülebilir dosya belirtilen bu zaman aralığında sona ermezse, otomatik olarak sonlandırılacaktır.

`<n>`: komut dosyası içinde paralel olarak çalışacak maksimum görev sayısı.

`<n1,2,3>`: ilgili iş adımı tarafından yürütülecek görev sayısı, burada `<n1> + <n2> + <n3> <= <n>`.

`<dir>`: TRUBA'da komut dosyasının yürütüleceği yol. Burası genellikle girdi ve çıktı dosyalarının bulunduğu yerdir. Komut dosyasında tanımlanan tüm göreli yollar `<out>` ile göreli olacaktır.

`<out>`: bu işin `stdout` unun yazdırılacağı dosya. Bu, koddaki yürütmelerin ürettiği tüm çıktıları içerir.

`<err>`: bu işin `stderr` inin yazdırılacağı dosya.

`Sbatch` komutunu çağırdığımızda, işi TRUBA kuyruğuna kaydedeceğiz. Kaynaklar mevcut olduğunda ve işimiz sıranın en üstünde olduğunda, aşağıdakiler gerçekleşecektir:

1. Talep edilen kaynaklar, talep edilen zaman aralığı için tahsis edilecektir ve bu durumda talep ettiklerimiz:
    1. `<n>` görevi paralel olarak yürütme yetkisi
    2. `<n>` CPU (TRUBA her görevin bir CPU gerektirdiğini varsayar)
    3. Yeterli sayıda düğüm (Truba tarafından belirlenir.).
2. `srun` ile başlayan satırlar, `my_program1`, `my_program2` ve `my_program3` programlarını çalıştıracak iş adımlarını başlatacaktır. Bununla birlikte, bu iş adımları, satırlarının sonuna `&` sembolünü eklediğimiz için paralel olarak çalışacaktır.
3. Betik, başlatılan tüm iş adımlarının yürütülmesi tamamlanana kadar `wait` komutunda yürütmeyi engelleyecektir.

Örnek:

`sbatch_example3.slurm` betiği, `print_argument` programını paralel olarak üç kez çalıştıracak `my_parallel_job` adlı bir işi tanımlar, her yürütmenin farklı bir parametresi olacaktır. Betik, `wait` komutunda bekleyecek ve üç yürütme işleminin tamamı bittikten sonra, programı farklı bir argüman kümesiyle paralel olarak üç kez daha çalıştırıcaktır ve ikinci bekleme(`wait`) komutunda sona erecektir. Bu iş `short` bölümüne eklenecek ve 20 dakika içinde bitecektir. `print_argument.cpp` dosyası `/truba/home/my_account/` konumunda bulunur. İşlerin çıktıları `/truba/home/my_account/output.txt` dosyasına ve hatalar `/truba/home/my_account/error.txt` dosyasına yazdırılacaktır.

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

Sonra aşağıdaki komutu kullanarak betiği sıraya ekliyoruz

```bash
$ sbatch  sbatch_example3.slurm
```

Bu adım sonrasında gözlemleyebileceğimiz bir çıktı aşağıdaki gibidir:

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

### Birden fazla görevi paralel olarak yürütmek ve bunların yürütülmesi için birden fazla sunucuyu (düğümü) ayırmak

SLURM'e, görevlerimizin bir SLURM komut dizisinde yürütülmesi için birden fazla sunucuyu tahsis etmesi talimatını verebiliriz. Bu, SLURM komut dosyası seçeneklerine `--nodes` seçeneği eklenerek yapılır. Bu, aşağıdaki SLURM betiği `sbatch4.slurm`'da gösterilmiştir:

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

Sonra aşağıdaki komutu kullanarak betiği sıraya ekliyoruz

```bash
$ sbatch sbatch4.slrum
```

`<my_account>`: TRUBA'daki hesap adı

`<job_name>`: iş kuyruğunda görünen gönderilen işin adı.

`<part>`: çalışmayı sıraya alacağınız bölümün adı.

`<time>`: Çalışmanızın çalışacağı maksimum süre. Bu girdinin biçimi `d-hh: mm: ss` şeklindedir, burada `d` günü, `hh` saati, `mm` dakikayı ve `ss` saniyeyi temsil eder. **Not:** Yürütülebilir dosya belirtilen bu zaman aralığında sona ermezse, otomatik olarak sonlandırılacaktır.

`<N>`: bu komut dosyasındaki görevleri çalıştırmak için kullanılacak düğüm (sunucu) sayısı.

`<n>`: komut dosyası içinde paralel olarak çalışacak maksimum görev sayısı.

`<n1,2,3>`: ilgili iş adımı tarafından yürütülecek görev sayısı, burada `<n1> + <n2> + <n3> <= <n>`.

`<dir>`: TRUBA'da komut dosyasının yürütüleceği yol. Burası genellikle girdi ve çıktı dosyalarının bulunduğu yerdir. Komut dosyasında tanımlanan tüm göreli yollar `<out>` ile göreli olacaktır.

`<out>`: bu işin `stdout` unun yazdırılacağı dosya. Bu, koddaki yürütmelerin ürettiği tüm çıktıları içerir.

`<err>`: bu işin `stderr` inin yazdırılacağı dosya.

`Sbatch` komutunu çağırdığımızda, işi TRUBA kuyruğuna kaydedeceğiz. Kaynaklar mevcut olduğunda ve işimiz sıranın en üstünde olduğunda, aşağıdakiler gerçekleşecektir:

1. Talep edilen kaynaklar, talep edilen zaman aralığı için tahsis edilecektir ve bu durumda talep ettiklerimiz:
    1. `<n>` görevi paralel olarak yürütme yetkisi
    2. `<n>` CPU (TRUBA her görevin bir CPU gerektirdiğini varsayar)
    3. `<N>` düğüm (sunucu)
2. `srun` ile başlayan satırlar, `my_program1`, `my_program2` ve `my_program3` programlarını çalıştıracak iş adımlarını başlatacaktır. Satırlarının sonuna `&` sembolünü eklediğimiz için bu iş adımları paralel olarak çalışacaktır.
3. Betik, başlatılan tüm iş adımlarının yürütülmesi tamamlanana kadar `wait` komutunda yürütmeyi engelleyecektir.

Örnek:

`sbatch_example4.slurm`, `multi_server_job` adlı bir işi tanımlar. Bu iş `short` bölümüne eklenecek ve 20 dakika içinde bitecektir. `print_argument.cpp` dosyası `/truba/home/my_account/` konumunda bulunur. İşlerin çıktıları `/truba/home/my_account/output.txt` dosyasına ve hatalar `/truba/home/my_account/error.txt` dosyasına yazdırılacaktır.

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

Sonra aşağıdaki komutu kullanarak betiği sıraya ekliyoruz

```bash
$ sbatch sbatch_example4.slurm
```

Bu adım sonrasında gözlemleyebileceğimiz bir çıktı aşağıdaki gibidir:

```
sardalya117.yonetim
sardalya117.yonetim
sardalya118.yonetim
```