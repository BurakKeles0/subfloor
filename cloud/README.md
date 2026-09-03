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
32 blok × 0.377 GiB = 12.06 GiB   sıkıştırılmış ağırlıklar (fp16 decoder katmanı)
inputs.pt              2.00 GiB   128 × 2048 kalibrasyonda (defterin koştuğu)
inputs.pt.tmp          2.00 GiB   atomik yazma sırasında, bir torch.save boyu
                     -----------
                      16.06 GiB   preflight 16.1'in altında reddediyor

model önbelleği      13.0  GiB   ayrı bir birimde; preflight yalnız --hf-home
                                  verilirse bakıyor, defter vermiyor
```

> **Ön-kaydın şekli diske sığmıyor.** `--calib-seqlen 4096`'da iki aktivasyon
> satırının **ikisi birden** ikiye katlanıyor: 12.06 + 4.00 + 4.00 = **20.06 GiB**,
> yani `/kaggle/working`'in 20 GB'ını aşıyor. C4 örnekleyicisi düzeltilse bile
> ön-kayıt protokolüyle bir noktayı Kaggle'da koşturmak disk tarafından
> bağlı — bu, aşağıdaki "geçici çözüm"ün ikinci ve kaydedilmemiş sebebi.

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

**A — klon (varsayılan).** Defteri kullandığınızda **yazılacak hiçbir komut
yok**: hücre 2 `REPO_URL`'i okuyup klonlamayı kendisi yapıyor ve pip'i kuruyor.

Defteri kullanmıyorsanız, bir hücreye **aynen** bu üç satır:

```python
!git clone --depth 1 https://github.com/BurakKeles0/subfloor /kaggle/working/subfloor
%cd /kaggle/working/subfloor
!pip install -q -r cloud/requirements.txt
```

> **`HF_HUB_DISABLE_XET=1` kurun.** Defter hücre 2'de kendisi kuruyor; elle
> kuşturuyorsanız siz koyun. STATUS §10'un tuzağı: Xet ile HF indirmeleri
> **0 B/s'de takılıyor**, ve belirti bir hata değil — ilerleme çubuğu olduğu
> yerde duruyor, ekranda hiçbir şey yazmıyor. Özellikle C4 parçasını
> çekerken vuruyor.

```python
import os; os.environ['HF_HUB_DISABLE_XET'] = '1'
```

Baştaki `!` şart: onsuz defter satırı **Python** sanar ve `SyntaxError` verir.
URL'nin başına `$` koymayın — `$REPO_URL` defterin içindeki değişkenin adı,
elle yazarken URL'nin kendisi geliyor. Kaggle'da oturum **Internet kapalı** başlar: Settings →
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

### 1. Oturumu kurun

Yeni Notebook → sağ panel → Settings:

- **Accelerator: GPU T4 ×2.** Kaggle tek T4 sunmuyor; ikincisini
  kullanmayacağız (aşağıda).
- **Internet: On.** Hesabın telefonla doğrulanmış olması gerekiyor, yoksa
  anahtar gri kalıyor.

> **Internet bu koşunun en sinsi hatası.** Kaggle oturumları **kapalı**
> başlıyor, ve `preflight.py` ağı hiç yoklamıyor — hat kontrolü bilerek
> sentetik (`Cal.synthetic_problem(64,128,256)`), indirme gerektirmiyor. Yani
> kapalıyken bile `READY` yazar, çekirdek hızlarını ölçer, ve koşu dakikalar
> sonra ilk HuggingFace çağrısında düşer. Belirti sebebe hiç benzemiyor.

### 2. Kodu getirin

`cloud/subfloor_cloud.ipynb`'yi **File → Import Notebook** ile yükleyin, ya da
ilk hücreyi elle koşturun:

```python
import zipfile, glob
z = glob.glob('/kaggle/input/**/subfloor-code.zip', recursive=True)[0]
zipfile.ZipFile(z).extractall('/kaggle/working/subfloor')
%cd /kaggle/working/subfloor
!pip install -q -r cloud/requirements.txt
```

### 3. Uçuş öncesi — atlamayın

```python
!python cloud/preflight.py --resume-root /kaggle/working/resume
```

Saniyeler değil dakikalar sürüyor, çünkü sonunda çekirdek hızlarını bu kartta
yeniden ölçüyor. `READY.` ya da `NOT READY:` ile bitiyor.

> **Çıkış kodunu defter yutuyor.** `!python …` IPython'da dönüş kodunu
> yükseltmiyor, yani `NOT READY` yazıp 1 döndürse bile bir sonraki hücre
> koşar. **Çıktıyı gözünüzle okuyun.**

`--hf-home <yol>` verirseniz preflight o yolda 13 GiB boş yer arıyor. Defter
vermiyor, yani model indirmesi için yer kontrolü hiç yapılmıyor. Dikkat: bu
bayrak indirmenin **yerini değiştirmiyor** — kodda hiçbir yerde `HF_HOME`
kurulmuyor. Önbelleği oturumlar arası taşımak istiyorsanız `os.environ['HF_HOME']`
kalıcı bir yola kurulmalı, yoksa 13 GiB her yeni oturumda yeniden iniyor.

### 4. Noktayı koşturun

Çıkış kodunu görmek istiyorsanız `!python` **yetmiyor** (yukarıdaki uyarı burada
da geçerli). Defterin kullandığı biçim:

```python
import subprocess, sys
cmd = [sys.executable, '-u', 'cloud/run_point.py',
       '--tile', '16', '--budget', '1.5', '--draw', '0',
       '--resume-root', '/kaggle/working/resume', '--hours', '11',
       '--calib-samples', '128', '--calib-seqlen', '2048',
       '--datasets', 'wikitext2']
rc = subprocess.run(cmd).returncode
print({0: 'BITTI', 42: 'BUTCE DOLDU -- bu hucreyi tekrar kosturun'}
      .get(rc, f'HATA (exit {rc})'))
```

İlk nokta için **T=16**: modellenen 1.14 saat, tek oturuma sığıyor, yani devam
etme döngüsünü hiç kullanmıyorsunuz. (Bu tahminin dayandığı 339 s/blok ölçümü
**dört** kalibrasyon penceresiyle alındı, oysa burada 128 geçiyoruz — istatistik
toplama pencere sayısıyla ölçekleniyor, yani ekstrapolasyon ölçülmüş değil.)

> **`--datasets wikitext2` C4'ü kaldırmıyor.** O bayrak yalnız *değerlendirme*
> setini seçiyor; kalibrasyon pencereleri her hâlükarda C4'ten geliyor
> (`m1_run.run_point`, koşulsuz `dataset="c4"`). Ama bu indirme **nokta başına
> bir kez**: devam eden oturum `inputs`'u checkpoint'ten okuyor ve C4'e hiç
> dokunmuyor.

| çıkış | anlamı | ne yapacaksınız |
|---|---|---|
| `0` | nokta bitti, JSON yazıldı | sonucu indirin |
| `42` | duvar saati doldu, checkpoint **tam** | aynı hücreyi aynı bayraklarla tekrar koşturun |
| `1` | **herhangi bir hata.** İlk satır `no CUDA device` ise runtime GPU'da değil; değilse traceback | traceback ne diyorsa |
| `2` | bayrak yanlış yazılmış (argparse) | komutu düzeltin |
| `-9` / `137` | süreç öldürüldü — host RAM | aşağıdaki tabloya bakın |

`run_point.py` bütçe dolduğunda ayrıca düz metinle söylüyor: `budget spent after
N block(s) this session; …`. Yani 42'yi kaçırsanız bile ekranda yazıyor.

> **Devam ederken bayrakları değiştirmeyin.** İki ayrı sonuç doğuruyor.
> `--calib-seqlen`'i değiştirirseniz aktivasyonlar diskten eski uzunlukta gelir,
> causal mask ve rotary yenisinde kurulur; checkpoint bunu reddetmiyor.
> `--calib-samples` sessizce yok sayılıyor, çünkü `inputs` checkpoint'ten
> okunuyor. Daha pahalısı: `--tile`, `--budget`, `--draw` ya da `--model`
> değişirse **slug değişiyor**, yani hata almazsınız — yeni ve boş bir
> checkpoint dizini açılır ve eskisinin ~16 GiB'ı 20 GB'lık diskte öylece
> kalır. `Checkpoint.load`'un "refusing to resume across configurations"
> koruması bu yüzden neredeyse hiç ateşlemiyor.

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

## Uçuş öncesi neyi görüyor, neyi görmüyor

`preflight.py` READY demeden önce altı şeye bakıyor. **Beşi sert reddediyor**
(CUDA, transformers, datasets, disk, hat); yalnız VRAM kendi geçersiz kılma
bayrağını adıyla söylüyor — bir duvar sebebi olan birini durdurur, bir uyarı ise
akıp gider ve koşu saatler sonra ölür.

| kontrol | geçme koşulu | T4'te |
|---|---|---|
| CUDA | cihaz var | ✓ |
| VRAM | ≥ 12 GiB, yoksa `--allow-small-gpu` | ✓ ~14.7 |
| transformers | major ≥ 5 | defterin 1. bölümündeki `pip install -r cloud/requirements.txt`'ten sonra ✓ |
| datasets | kurulu | ✓ |
| disk | `--resume-root`'ta ≥ **16.1 GiB** (`POINT_CHECKPOINT_GIB`) | ✓ 20 GB'de |
| hat | sentetik katman uçtan uca `run_config`, artı codebook'un kanonik olduğu | ✓ |

Tablodaki sıra çalışma sırası, ve **hat kontrolü koşullu**: yalnız diğerleri
temizse koşuyor. Yani disk ya da transformers yüzünden gelen bir `NOT READY`
raporunda hat hakkında hiçbir şey yazmıyor — yokluğundan "hat sağlam" sonucu
çıkarılamaz. `--hf-home` verirseniz yedinci bir koşul ekleniyor: o yolda 13 GiB.

**Görmedikleri.** İkisi koşuyu düşürüyor, üçüncüsü yalnız yavaşlatıyor:

- **Internet.** Hiç yoklamıyor. Hat kontrolü bilerek sentetik
  (`Cal.synthetic_problem(64,128,256)`), indirme gerektirmiyor — dolayısıyla
  kapalı bir oturumda da READY yazıyor. **Koşuyu düşürür.**
- **Host RAM.** Yalnız VRAM'e bakıyor. Colab ücretsizin blok 0'dan önce
  OOM-kill yemesinin sebebi bu. **Koşuyu düşürür.**
- **Beş eşik.** `_LATTICE_MIN_ROWS`, `_ANALYTIC_MIN_ROWS`,
  `_ANALYTIC_DIRECT_MIN_ROWS`, `CHUNK_TARGET_ROWS`, `DECODER_MISS_FRACTION` —
  yeniden ölçmüyor, yalnız **basıyor**. Yanlışlarsa **sonuç bozulmaz, koşu
  yavaşlar**: ölü bant başka yere düşer ve hücreler tam taramaya sapar.

> **Ve preflight'ın kendi hız ölçümü korumasız.** `bench_guard.require_quiet_gpu`
> yalnız `m0_tile_timings.py` ile `m0_lever_audit.py`'de çağrılıyor; preflight'ın
> `measure_rates` çağrısı meşgul bir kartta da ölçer ve sonucu
> `results/m0_rates.json`'a **önbellekler**, sonraki koşular onu yeniden okur.
> Meşgul bir kartta ölçtüyseniz o dosyayı silin.

---

## Koşarken neye bakacaksınız

Ekranda iki satır, dosyada bir alan.

### Blok süresi ve bütçe

`run_point.py` her mesajın başına geçen süreyi ekliyor, ve blok sonunda bütçenin
ne kadarının kaldığını yazıyor:

```
[  27.0 min]   block 1/32 done (27.0 min)
           checkpoint at block 1/32, 633 min of budget left
```

İlk sayıyı 32 ile çarpın. Bu, projenin sahip olmadığı sayı. Yerel 8 GiB kartta
sürücü blok başına **339 s** koşuyordu, oysa aynı yedi katman sessiz bir kartta
izole ölçümde **86 s** (§6.18; §6.16'nın ısınmış ölçümü 84 s, soğuk ayrı süreçte
142 s). Aradaki fark aritmetik değil bellek baskısıydı: §6.17 bunun 1.46×'ini
Hessian'ları erken bırakarak kapattı, kalan ~2.7× **hiç profillenmedi** ve
16 GiB'ta ne kadarının kalktığı bilinmiyor. İkinci satır ise bütçenin yetip
yetmeyeceğini doğrudan söylüyor.

### Blok 0'ın erken uyarısı

`diagnostics` alanında `ratio_to_dense` var: sıkıştırılmış katmanın hatasının,
aynı katmanın **yoğun E8P** referansına oranı. Referans blok 0'ın yedi
katmanının fazladan bir quantizasyonuyla üretiliyor.

> **`ratio_to_dense > 2.0` ise varsayım düştü.** E8P'nin kompaktlanmış survivor
> alt-matrisinde 2-bit *kalitesini* koruduğu, §3.2'nin açık varsayımı ve
> projenin en büyük tek riski. Düşerse geri dönüş yolu rotasyon + GPTQ-3bit
> (`W = 3.148`) ve bant 1.83–2.83'e kayıyor. `assumption_broken` alanı bunu
> ayrıca `true` yazıyor.

Sonucu beklemenize gerek yok: blok 0 bittiği anda `state.json`'a yazılıyor.
Ayrı bir hücreden:

```python
import json
p = '/kaggle/working/resume/Llama-2-7b-hf_b1.5_t16_d0/state.json'
print(json.load(open(p))['diagnostics'])
```

Hiçbir şey bunu ekrana basmıyor, ve nokta bitince `clear()` o dizini siliyor —
bakacaksanız koşu sürerken bakın.

---

## Sonuç dosyası

```
<resume-root>/Llama-2-7b-hf_b1.5_t16_d0.json
```

`clear()` nokta bitince checkpoint **dizinini** siliyor, ama JSON o dizinin
kardeşi, yani kalıyor. Defterin son hücresi `<resume-root>/*.json` ile buluyor.

**Yalnız temiz bitişte yazılıyor.** `exit 42`'den sonra ortada checkpoint dizini
var ama JSON **yok**; dosya ancak nokta 0 ile bittiğinde oluşuyor.

| alan | ne |
|---|---|
| `spec` | model, bütçe, tile, çekiliş — yedi nokta dosyasını ayıran şey |
| `perplexity` | dataset başına. Defter yalnız `wikitext2` koşturuyor, yani tek anahtar olacak; ön-kayıt C4'ü de istiyor |
| `records` | katman başına `rel_output_error`, artı `block`/`name`/`layer`/`n_in`/`n_out`/`n_tokens` — 224 kayıt. **SNR yok** |
| `diagnostics` | blok 0'ın yoğun E8P referansı, `ratio_to_dense`, `assumption_broken`, `dense_e8p_snr_db` |
| `levers` | `rotate_kron`, `search_dtype`, `compensate_block` |
| `seconds` | noktanın **toplam** maliyeti, oturumlar boyunca |
| `seconds_this_session` | yalnız son oturum |

SNR'nin katman kayıtlarında olmaması bir eksiklik: `run_config` hesaplıyor ama
sürücü ondan yalnız `W_hat`'i alıyor. JSON'daki tek SNR blok 0'ın referansı.

**JSON'da olmayanlar**, elle not düşün: GPU adı, torch ve transformers
sürümleri, `has_triton()`. `calib_seqlen` alan olarak yok ama her kayıttaki
`n_tokens` onu ele veriyor — 128 × 2048 → 262144, 128 × 4096 → 524288. Ayırt
edilemeyen şey **slug ve `PointSpec`**, dosyanın içeriği değil.

Oturum kapanmadan indirin — `/kaggle/working` oturumla birlikte gidiyor.

---

## Bir şey ters giderse

| gördüğünüz | sebep | ne yapacaksınız |
|---|---|---|
| `exit 42` | bütçe doldu, checkpoint tam | aynı hücreyi aynı bayraklarla tekrar koşturun |
| `klon basarisiz` + 403/404 | depo private, defter kimlik doğrulamıyor | zip yolu — zip klondan **önce** deneniyor |
| `transformers 4.x … v5 keyword` | 1. bölümün pip'i koşmadı | `!pip install -q -r cloud/requirements.txt` |
| preflight READY, sonra hub hatası | Internet kapalı | Settings → Internet: On, oturumu yeniden başlatın |
| `only found 42/128 windows of 4096 tokens` (sayı oynar) | `--calib-seqlen 4096`; 12.800 denemeden sonra pes ediyor | 2048'e alın |
| disk hatası, koşunun sonuna doğru (boş alana göre blok ~24–31) | checkpoint sığmadı; 32 bloğun **hepsi** eval için gerekli | resume-root'ta 16.1 GiB boş olmalı |
| `exit -9` / runtime restart | host RAM | Colab ücretsizdeyseniz Kaggle'a geçin; ikinci bir `run_point` süreci varsa durdurun |
| `refusing to resume across configurations` | bu dizinde başka bir spec'in state'i var | dizini silin ya da doğru bayraklarla koşturun |
| `block N is marked done but … is missing` | oturum kapandı, `/kaggle/working` gitti, state kaldı | nokta dizinini silip baştan başlayın |

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
