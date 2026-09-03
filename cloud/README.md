# Bulutta koşmak — Colab ve Kaggle

Bu klasör **ana projeye hiç dokunmuyor.** Altındaki her dosya ek; hat kodu,
testler ve `docs/STATUS.md` olduğu gibi duruyor. Koşan şey yerelde koşanın
aynısı: `experiments/m1_run.py`. Buradaki iki dosya yalnızca ücretsiz bir
oturumun ihtiyaç duyduğu şeyi ekliyor — **duvar saati bütçesi** ve
**varsayılan olarak devam etme**.

Ayrı bir "bulut sürümü" yazmamanın sebebi bu: sapan bir çatal, deneyin ikinci
bir uygulaması olur ve hangisinin sonucu raporlanacağı sorusu ortaya çıkar.

---

## Neden buluta çıkmak mantıklı — ölçülmüş gerekçe

Kısıt hesap gücü değil, **VRAM**.

| | ölçülen |
|---|---|
| bu dizüstünde tek katman sıkıştırma tepesi | **5.4 GiB** (kullanılabilir 6.8) |
| sürücü koşarken kart | **7.5–7.8 GiB / 8.15** — %94 dolu, 200 sn boyunca |
| bir denemede sonuç | `cudaErrorUnknown`, cuSOLVER'ın içinde |
| Hessian'ları tutmanın bedeli (§6.17) | **1.46×** — ayırıcı tahliye edip yeniden istiyor |
| sürücü blok 0 (08-25) | **27.0 dk**; modelin dediği ~11 |

Yani bu kartta yavaşlığın büyük kısmı aritmetik değil, **tavana dayanmak**.
16 GB'lık bir T4'te bu sınıf yok oluyor — dolayısıyla ücretsiz katmanın T4'ü
bu iş için bu dizüstünden **hızlı** olabilir, daha güçlü olduğu için değil,
sığdığı için.

Buna karşılık: T4'ün fp32 **tepe** değeri 8.1 TFLOP/s, bu kartın *elde ettiği*
8.2. Ham hesapta kazanç yok. Ve **TF32 hattı kırıyor** (§6.9), yani A100/H100'ün
tensor çekirdeği avantajı da erişilebilir değil.

---

## Platformlar — sert sayılar

| | Kaggle | Colab (ücretsiz) |
|---|---|---|
| GPU | T4 ×2 veya P100, 16 GB | T4 16 GB, **garantisiz** |
| oturum sınırı | 12 saat | nominal 12, pratikte 3–6 |
| haftalık kota | **30 GPU-saat** | yok, ama önceliksiz |
| kalıcı disk | `/kaggle/working` **20 GB** | Drive **15 GB** (Gmail'le ortak) |
| **host RAM** | **~29 GB** | **~12.7 GB** — model sığmıyor |
| model | Dataset olarak bağlanır, bir kez | her oturumda 13 GB iner |

**Kaggle bu iş için açık ara daha uygun**, ve iki bağımsız sebeple.

**Disk.** Bir noktanın checkpoint'i **16.06 GiB**'e ulaşıyor (aşağıdaki tablo) ve
Colab'ın ücretsiz 15 GB Drive'ına model ile birlikte değil, **tek başına bile**
sığmıyor.

**Host RAM.** Bundan daha sert olanı bu. `hf_llama.load_llama` `device_map=None`
ile yüklüyor — model bilerek CPU'da duruyor ve kalibrasyon döngüsü karta blok
blok taşıyor, 7B'yi küçük bir kartta çalıştırılabilir kılan şey bu. Ama bedeli
**13.5 GiB** host RAM, üstüne `capture_block_inputs`'ın tuttuğu 2.00 GiB
aktivasyon biniyor. Kaggle'ın ~29 GB'ı bunu kaldırıyor; Colab ücretsiz katmanın
~12.7 GB'ı **kaldırmıyor** ve süreç blok 0'dan önce OOM-kill yiyor.
`cloud/preflight.py` bunu göremiyor: yalnız VRAM'e bakıyor, host RAM'e değil.

### Bir noktanın disk matematiği

```
32 blok × 0.38 GiB   = 12.1 GiB   sıkıştırılmış ağırlıklar
inputs.pt              2.00 GiB   128 × 2048 kalibrasyonda (defterin koştuğu)
inputs.pt.tmp          2.00 GiB   atomik yazma sırasında, bir torch.save boyu
                       4.00 GiB   ön-kaydın 128 × 4096'sında
model önbelleği       13.0  GiB   (Kaggle'da Dataset, Colab'da her sefer)
```

Blok dosyalarının **hepsi sonuna kadar gerekiyor**: değerlendirme birleştirilmiş
sıkıştırılmış model üzerinde koşuyor (`Checkpoint.apply_saved_blocks`). Yani
disk yetmezse koşu **geç** çuvallar, erken değil.

---

## Tasarım F'i bölüştürmek

F = tek bütçe (B=1.5), tek çekiliş, 7 tile boyutu. Noktalar **tamamen
bağımsız** — aralarında hiçbir veri akışı yok, bu yüzden ayrı makinelere
dağıtılabilirler.

| T | modelin dediği | devam gerekir mi |
|---|---|---|
| 1 | 4.01 saat | muhtemelen |
| 2 | 3.34 saat | sınırda |
| 4 | 2.07 saat | hayır |
| 8 | 1.54 saat | hayır |
| 16 | 1.14 saat | hayır — **ilk nokta için bu** |
| 32 | 0.92 saat | hayır |
| max | 0.66 saat | hayır |
| **toplam** | **13.7 saat** | |

**Hepsi Kaggle'a, sırayla.** Bu tablo bir zamanlar ucuz noktaları Colab'a
dağıtıyordu; o dağıtım **yanlıştı** ve sebebi yukarıdaki host RAM satırı:
Colab'ın ücretsiz katmanı modeli hiç tutamıyor, noktanın ucuz ya da pahalı
olmasından bağımsız olarak. Paralellik isteniyorsa iki ayrı Kaggle oturumu
gerekiyor — **aynı oturumda iki süreç değil**, çünkü iki host kopyası (2 × 13.5
GiB) ve iki checkpoint (2 × 16.06 GiB) tek kutuya sığmıyor.

Kaggle kotası: yedi noktanın tamamı ≈ 13.7 GPU-saat, haftalık 30'un içinde
rahat — model saatleri tutarsa. Tutmazsa kota bağlayıcı olabilir, ve bunu
söyleyecek şey ilk noktanın blok süresi.

> **Bu saatler modelin dediği.** Yerel sürücü modelden çok daha yavaş koşuyor ve
> sebebin bellek olduğuna dair kanıt yukarıda. 16 GB'ta ne kadarının kalktığı
> **ölçülmedi** — bulutta koşacak ilk nokta aynı zamanda o ölçüm oluyor.

---

## Önce: kodu buluta nasıl götürürsünüz

Depo **GitHub'da**: `https://github.com/BurakKeles0/subfloor`, dal `main`.
Defterin `REPO_URL`'i onu gösteriyor, yani varsayılan yol klonlamak. Zip yolu
özel bir fork için ya da Internet'i kapalı bir oturum için duruyor.

**A — klon (varsayılan).** Defter `git clone --depth 1 $REPO_URL` çalıştırır ve
başka bir şey istemez. Kaggle'da oturum **Internet kapalı** başlar: Settings →
Internet'i açın, yoksa bu yol da HuggingFace indirmesi de düşer.

**B — zip (Internet kapalıysa ya da özel fork).** Yerelde:

```bash
git archive --format=zip -o ~/subfloor-code.zip HEAD
```

0.4 MB çıkıyor (78 dosya). Sonra:

- **Kaggle**: Datasets → New Dataset → zip'i yükleyin → notebook'a *Add Data* ile
  ekleyin. Defter `/kaggle/input/**/subfloor-code.zip` altında kendisi bulur.
- **Colab**: Drive'a koyun. Defter `MyDrive` altında arar.

> `git archive HEAD` **çalışma ağacını değil, HEAD'i** paketler ve klon yolu da
> HEAD'i çeker. Commit edilmemiş bir düzeltme buluta gitmez — zip'i almadan
> önce `git status`'a bakın.

---

## Kaggle — adım adım

1. Yeni Notebook → Settings → **Accelerator: GPU T4 ×2**, **Internet: On**
2. `cloud/subfloor_cloud.ipynb`'yi yükleyin, veya ilk hücreyi elle koşturun:

```python
import zipfile, glob
z = glob.glob('/kaggle/input/**/subfloor-code.zip', recursive=True)[0]
zipfile.ZipFile(z).extractall('/kaggle/working/subfloor')
%cd /kaggle/working/subfloor
!pip install -q -r cloud/requirements.txt
```

3. Uçuş öncesi — **atlamayın**, aşağıda nedeni var:

```python
!python cloud/preflight.py --resume-root /kaggle/working/resume
```

4. Bir nokta:

```python
!python -u cloud/run_point.py --tile 1 --budget 1.5 \
    --resume-root /kaggle/working/resume --hours 11 \
    --calib-samples 128 --calib-seqlen 2048
```

Çıkış kodu **42** ise bütçe doldu ve checkpoint tam: aynı komutu yeniden
koşturun, kaldığı yerden devam eder.

**İkinci T4 tek kutuda kullanılamıyor.** Burada bir zamanlar iki
`CUDA_VISIBLE_DEVICES` süreci başlatan bir reçete vardı; darboğaz GPU değil host.
Model `device_map=None` ile CPU'da duruyor, yani iki süreç **2 × 13.5 GiB** RAM
ister (kutuda ~29 GB, üstüne iki aktivasyon kümesi) ve **2 × 16.06 GiB**
checkpoint yazar (`/kaggle/working`'de 20 GB). İkisi de yetmiyor. Paralellik
istiyorsanız iki ayrı oturum açın.

**Oturumlar arası kalıcılık:** `/kaggle/working` yalnız oturum içinde yaşıyor.
Devam etmek istiyorsanız notebook'u "Save Version" ile koşturun (çıktı kalıcı
olur ve bir sonraki koşuya Dataset olarak bağlanır), ya da `resume` klasörünü
bir Dataset'e yazın.

---

## Colab — adım adım

> **Ücretsiz katmanda çalışmıyor.** Model CPU'da duruyor (13.5 GiB) ve üstüne
> 2.00 GiB aktivasyon biniyor; ücretsiz Colab ~12.7 GB host RAM veriyor, yani
> süreç blok 0'dan önce OOM-kill yiyor. `preflight.py` bunu yakalayamıyor —
> yalnız VRAM'e bakıyor. Aşağıdakiler daha fazla RAM'i olan bir katman için
> duruyor; ücretsiz katmandaysanız Kaggle'a gidin.

```python
from google.colab import drive; drive.mount('/content/drive')
import zipfile, glob
z = glob.glob('/content/drive/MyDrive/**/subfloor-code.zip', recursive=True)[0]
zipfile.ZipFile(z).extractall('/content/subfloor')
%cd /content/subfloor
!pip install -q -r cloud/requirements.txt
!python cloud/preflight.py --resume-root /content/resume
```

**Ucuz noktalar için** checkpoint'i Drive'a yazmayın — yerel diske yazın ve tek
oturumda bitirin. Drive'a 12 GiB blok yazmak bloğun kendisinden uzun sürer:

```python
!python -u cloud/run_point.py --tile 16 --budget 1.5 \
    --resume-root /content/resume --hours 3.5 \
    --calib-samples 128 --calib-seqlen 2048
```

Bittiğinde yalnız sonucu Drive'a kopyalayın:

```python
!cp /content/resume/*.json /content/drive/MyDrive/subfloor/
```

---

## Bilinmesi gereken üç şey

### 1. `--calib-seqlen 2048`, 4096 değil — ve bu bir geçici çözüm

`m1_run.py`'nin varsayılanı `CALIB_SEQLEN = 4096` ve **o ayarda çalışmıyor.**
`calibrate.load_calibration_tokens` C4'ten belge örnekleyip `seqlen`'i aşanları
tutuyor; deneme bütçesi pencere başına 100. Ölçüldü (600 belge):

| eşik | C4 belgelerinin | gereken deneme |
|---|---|---|
| >1024 | %10.50 | ~10 |
| >2048 | %2.83 | ~35 |
| **>4096** | **%0.33** | **~300** |

Yani 4096'da örnekleyici **hiçbir `n_samples` değerinde** yetişemiyor —
`only found 1/4 windows` ile düşüyor. 2048'de 3× marjla çalışıyor.

Bu ana projede **düzeltilmedi**, çünkü düzeltmesi kalibrasyon protokolünü
değiştirmek demek (belgeleri akışa birleştirmek, wikitext yolunun yaptığı gibi)
ve o bir karar. Burada yalnız kaydedildi.

### 2. Sabitler yeniden ölçülmeli

Bu projedeki her zamanlama sabiti "bu makinede ölçüldü" diyor ve öyle demeyi
hak ediyor. `preflight.py` çekirdek hızlarını yeniden ölçüyor; **tile
zamanlamaları ve rotasyon tablosu ölçülmüyor** ve onlar için:

```bash
python experiments/m0_tile_timings.py
python -u experiments/m0_lever_audit.py --build --rot-sweep
```

Ayrıca `preflight.py`'nin sonunda bastığı beş **eşik** var
(`_LATTICE_MIN_ROWS`, `_ANALYTIC_MIN_ROWS`, `_ANALYTIC_DIRECT_MIN_ROWS`,
`CHUNK_TARGET_ROWS`, `DECODER_MISS_FRACTION`). Bunlar bu dizüstüne göre
ayarlandı ve başka bir kartta ölü bant başka yere düşer — §6.13 tam olarak bu
hataydı ve ızgaranın 21 hücresinin 8'ini 65,536 kodsözcüğü taratıyordu.

### 3. Kaldıraçlar

Hat şu an `rotate_kron=True`, `compensate_block=512`, `search_dtype=None`
(fp16 **kapalı**) ile koşuyor. fp16 burada 1.00× çıktığı için reddedildi (§6.17)
— ama o bilanço **belleği içermiyordu**, ve bu kartta bellek belirleyici çıktı.
Bol VRAM'li bir makinede fp16'nın yeniden fiyatlanması gerekmiyor; **dar** bir
makinede gerekebilir.

`preflight.py` koştuğu konfigürasyonu basıyor, yani her koşunun kaydında hangi
kaldıraçlarla alındığı yazılı oluyor.

---

## Dosyalar

| dosya | ne yapar |
|---|---|
| `preflight.py` | ortam + disk + hat kontrolü, çekirdek hızlarını yeniden ölçer, ölçülmemiş eşikleri **basar** |
| `run_point.py` | bir nokta; duvar saati bütçesi blok sınırında durur, devam varsayılan |
| `requirements.txt` | yalnız ek bağımlılıklar; torch **kasten** sabitlenmemiş |

`run_point.py`'nin bütçesi bloğun bittiğini **mesajdan değil** `state.json`'ın
`next_block` alanından anlıyor — o alan `Checkpoint.save_block`'un en son
yazdığı şey, yani bütçe tam olarak ortada eksiksiz bir checkpoint varken
ateşliyor. Bir cümleyi değil, durumu izliyor.
