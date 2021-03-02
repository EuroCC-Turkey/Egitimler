# Srun kullanarak tek bir program çalıştırma

TRUBA üzerinde tek bir program çalıştırmak istersek, `srun` komutunu kullanarak bunu tek satırda yapabiliriz. Bu komut, sağlam bir arayüz sağlar ve büyük esnek konfigürasyon yetkisi sağlar. Ancak, `srun` kullanımının yürütme terminalini bloke edeceği unutulmamalıdır. Bu, komutu çalıştırdıktan sonra, terminali tekrar kullanmadan önce yürütmenin tamamlanmasını beklemeniz gerektiği anlamına gelir. `Srun`'u çağırmak, hem bir iş yaratır (kaynakları ayırır) hem de tek bir komutta bir iş adımını (kodu çalıştırmak için ayrılan kaynakları kullanır) yürütür. İşi tahsis etmek ve tahsis edilmiş kaynak üzerinde birden fazla iş adımı yürütmek istiyorsanız, lütfen `sbatch` kullanımıyla ilgili örneklere bakın.

### Tek bir program yürütme

Mevcut dizinde bulunan `my_program` ı çalıştırmak için aşağıdaki komutu kullanabiliriz:

```bash
$ srun --partition=<partition> --time=<time> ./my_program
```

`<partition>`:  işin sıralanmasını istediğimiz bölümün adı.

`<time>`: Çalışmanızın çalışacağı maksimum süre. Bu girdinin biçimi `d-hh: mm: ss` şeklindedir, burada `d` günü, `hh` saati, `mm` dakikayı ve `ss` saniyeyi temsil eder. **Not:** Yürütülebilir dosya belirtilen bu zaman aralığında sona ermezse, otomatik olarak sonlandırılacaktır.

Bu komut;

1. bir iş oluşturur, yani belirtilen süre boyunca bir düğüm (sunucu) ayırır ve
2. `my_program`'ı çalıştıracak bir iş adımı yürütür.

### Örnek:

Aşağıdaki komut, `my_program` ı `short` bölümünde sıralar ve ona maksimum 20 dakikalık bir zaman sınırı verir.

```bash
$ srun --partition=short --time=0-00:20:00 ./my_program
```

### Tek bir programı yürütme ve birden çok CPU'u tahsis etme

Birden çok CPU kullanabilen `my_program` ı çalıştırmak için aşağıdaki komutu kullanıyoruz:

```bash
$ srun --partition=<partition> --time=<time> --cpus-per-task=<cpus> ./my_program
```

`<cpus>`: yürütme için ayrılmış CPU sayısı.

`<partition>`: işin sıralanmasını istediğimiz bölümün adı.

`<time>`: Çalışmanızın çalışacağı maksimum süre. Bu girdinin biçimi `d-hh: mm: ss` şeklindedir, burada `d` günü, `hh` saati, `mm` dakikayı ve `ss` saniyeyi temsil eder. **Not:** Yürütülebilir dosya belirtilen bu zaman aralığında sona ermezse, otomatik olarak sonlandırılacaktır.

Bu komut,

1. bir iş oluşturur, yani belirtilen süre içersinde bir düğüm (sunucu) ayırır ve düğümde `<cpus>` kadar CPU tahsis eder ve
2. tahsis edilen CPU'ları kullanarak `my_program`'ı çalıştıracak bir iş adımı (kod yürütme) yürütür.

### Örnek:

Aşağıdaki komut, `short` bölümünde `my_program`'ı çalıştıracak, 20 dakika içinde bitecek ve bu yürütme için 8 CPU ayıracaktır:

```bash
$ srun --partition=short --time=0-00:20:00 --cpus-per-task=8 ./my_program
```

### Tek bir programı parallel olarak birden çok kez yürütme

Geçerli dizinde bulunan `my_program`'ı birden çok kez paralel olarak çalıştırmak için (örneğin, ortalama amaçları için), aşağıdaki komutu kullanırız:

```bash
$ srun --partition=<partition> --time=<time> --ntasks=<n> ./my_program
```

`<partition>`: işin sıralanmasını istediğimiz bölümün adı.

`<time>`: Çalışmanızın çalışacağı maksimum süre. Bu girdinin biçimi `d-hh: mm: ss` şeklindedir, burada `d` günü, `hh` saati, `mm` dakikayı ve `ss` saniyeyi temsil eder. **Not:** Yürütülebilir dosya belirtilen bu zaman aralığında sona ermezse, otomatik olarak sonlandırılacaktır.

`<n>`: komut dosyası içinde paralel olarak çalışacak maksimum görev sayısı.

Bu komut,

1. bir iş oluşturur, belirtilen süre için bir düğüm (sunucu) ayırır, `<n>` görevlerini çalıştırma yetkisini talep eder ve,
2. `my_program`'ı toplam `<n>` kez çalıştıracak bir iş adımı (kod yürütme) yürütür. `my_program`'ın her çalıştırılmasına "görev" adı verilir.

### Örnek:

Aşağıdaki komut, my_program'ı 5 kez paralel olarak çalıştıracak ve bu yürütmeleri gerçekleştirmek için 2 ile 4 sunucu arasında sunucu kullanacaktır. `short` bölümünü kullanacak ve 20 dakika içinde bitecektir.

```bash
$ srun --partition=short --time=0-00:20:00 --ntasks=5 --nodes=2-4 ./my_program
```

### Tek bir programı birden çok kez paralel olarak yürütme ve yürütme işlemlerinin her biri için birden çok CPU tahsis etme

Geçerli dizinde bulunan `my_program`'ı paralel olarak birden çok kez çalıştırmak ve yürütmelerin her biri için birden çok CPU atamak için aşağıdaki komutu kullanırız:

```bash
$ srun --partition=<partition> --time=<time> --ntasks=<n> --cpus-per-task=<cpus> ./my_program
```

`<partition>`: işin sıralanmasını istediğimiz bölümün adı.

`<time>`: Çalışmanızın çalışacağı maksimum süre. Bu girdinin biçimi `d-hh: mm: ss` şeklindedir, burada `d` günü, `hh` saati, `mm` dakikayı ve `ss` saniyeyi temsil eder. **Not:** Yürütülebilir dosya belirtilen bu zaman aralığında sona ermezse, otomatik olarak sonlandırılacaktır.

`<n>`: komut dosyası içinde paralel olarak çalışacak maksimum görev sayısı.

`<cpus>`: her görevi yürütmek için ayrılmış işlemci sayısı.

Bu komut,

1. bir iş oluşturur, yani belirtilen süre boyunca bir düğüm (sunucu) ayırır ve,
2. `my_program`'ı toplam `<n>` kez çalıştıracak bir iş adımı yürütür. `my_program`'ın her çalıştırılmasına "görev" adı verilir. Ayrıca, her "göreve", yürütmek için `<cpus>` CPU verilecektir. Başka bir deyişle, çalışacak iş adımı `<n> * <cpus>` sayıda CPU yürütecektir.

### Örnek:

Aşağıdaki komut, `my_program`'ı 5 kez paralel olarak çalıştıracaktır. Her yürütme 8 CPU kullanacaktır. `short` bölümünü kullanacak ve 20 dakika içinde bitecektir.

```bash
$ srun --partition=short --time=0-00:20:00 --ntasks=5 -cpus-per-task=8 ./my_program
```