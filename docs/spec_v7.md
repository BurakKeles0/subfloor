# Sparsity Below the Quantization Floor — Implementation Spec v7

> Uygulama şartnamesi. Matematik, muhasebe, ön-kayıt ve protokol **bağlayıcıdır**.
>
> v6'dan farkı bir yama değil: survivor quantizer'ı GPTQ-4bit'ten kafes VQ'ya
> (QuIP# E8P) geçti, bu da bütün çapaları 2 bitin altına taşıdı. Ayrıca v6'nın
> dört aritmetik hatası düzeltildi, `incoherence processing` risk olmaktan çıkıp
> çekirdek mekanizmaya döndü, ve Kapı A provası çerçeveyi daralttı.
>
> Gerekçelerin tamamı `docs/audit.md`'de. Değişiklik listesi Ek A'da.
> Bu belgedeki her sayı `accounting.py`'den üretilmiştir; hiçbiri elle yazılmadı.

---

## 0. Ana soru, mekanizma, hedef rejim

### 0.1 Hedef rejim — bit bütçesi bağlam uzunluğudur

Llama-2-70B, 24 GiB kart. Bütçeden bağımsız üç kalem:

```
embed_tokens + lm_head = 2 · 32000 · 8192 = 524.3M param, FP16 → 0.977 GiB
KV cache (GQA: 2·8·128·80·2 bayt)        = 327.7 kB/token
aktivasyon + framework yükü              ≈ 0.7 GiB
lineer katmanlar: 68.45B param           → 7.9686 GiB / bit
```

> **v6 burada hata yapıyordu.** "4k bağlam, gerçek eşik ~2.2–2.3 bit" diyordu;
> kendi kalemleriyle 4k'da eşik **2.46–2.65**. Yani PTQ tabanı (~2.0) rahatça
> yetiyordu ve "tabanın altına inmeliyiz" gerekçesi çöküyordu.

Doğru çerçeve tek bir eşik değil, **bütçe ile bağlamın takası**. 22.5 GiB
kullanılabilir varsayımıyla:

| bit/pozisyon | sığan bağlam |
|---|---|
| 2.5 | 2.9k |
| 2.2 | 10.5k |
| **2.0** (PTQ tabanı) | **15.6k** |
| 1.8 | 20.7k |
| 1.6 | 25.8k |
| **1.5** | **28.4k** |
| 1.4 | 30.9k |

**Motivasyon budur:** yoğun PTQ'nun tabanı seni ~16k bağlamda durduruyor.
1.5 bite inmek bağlamı neredeyse ikiye katlıyor. Bu, tek bir eşik iddiasından
hem daha dürüst hem daha güçlü — ve doğrulanabilir.

> ⚠️ **Limitations:** deneyler 7B/13B'de, motivasyon 70B'de. Büyük modeller daha
> toleranslı → 7B sonuçları **muhafazakâr**. M5'te 13B doğrulama noktası zorunlu.

### 0.2 Soru

PTQ'nun pratik bir tabanı var (~2.0 bit; QuIP#, QTIP, AQLM). Altında yoğun
quantization'ın gidecek yeri yok — QuaRot-GPTQ 2-bit'te 22.07, QuIP# 6.66.
Seyrekliğin tabanı ise **kullanılan indeks formatına bağlıdır** (§3.2).

> **Birincil soru:** 2 bitin altındaki bütçelerde, seyreklik ailesi *içinde*
> granülerlik nasıl seçilmeli?
>
> **İkincil soru:** o bütçelerde seyrek şemalar yoğun PTQ tabanının altına
> inebiliyor mu?

> **Sıralama Kapı A provasının sonucudur** (`docs/gate_a_dry_run.md`).
> GPTQ-4bit survivor'larla, literatürün konuşabildiği her yerde seyrek şemalar
> yoğun düşük-bite kaybediyor. v6'nın dağıtım odaklı çerçevesi bu haliyle
> savunulamazdı. E8P survivor'lar tabloyu değiştiriyor ama iddia yine de
> ikincil tutulur ve ölçümle kazanılır.

### 0.3 Mekanizma

`W` = survivor başına bit, `L = log₂(n_idx)`, `index(T) = min(1, d·L)/T`.

Bitmap rejiminde (`d·L ≥ 1`), `d(T) = (B − 1/T)/W` ve:

$$\boxed{\;d(T) - d(1) = \frac{1 - 1/T}{W}\;}\qquad\qquad \boxed{\;\frac{d(T)-d(1)}{d(\infty)-d(1)} = 1 - \frac{1}{T}\;}$$

Birincisi `B`'den bağımsız, ikincisi hem `B`'den hem `W`'den bağımsız.

| survivor quantizer | `W` | `T=16` avantajı |
|---|---|---|
| GPTQ 4-bit | 4.156250 | 0.2255639 |
| GPTQ 3-bit | 3.148438 | 0.2977667 |
| GPTQ 2-bit | 2.140625 | 0.4379562 |
| **QuIP# E8P** | **2.000000** | **0.4687500** |

**Mutlak avantaj sabittir.** Oranın büyümesi tamamen paydanın küçülmesinden —
manşeti "oran büyüyor" diye yazmak yanlıştır (§7, tuzak 2).

**Doğru mekanizma:** sabit mutlak yoğunluk kazancı × kalite-yoğunluk eğrisinin
dikleşen kısmı. Bağımsız test edilebilir (§5.1'in `τ` yüzeyi).

### 0.3.1 `B*` duvarı — avantajın erime noktası

Liste rejiminde unstructured için `B = d(W+L)`. Geçiş noktası:

$$B^*(T) = \frac{1}{T} + \frac{W}{L}$$

`n_idx = 11008` (`L = 13.4262648`): 4-bit `B* = 1.3095612`, E8P `B* = 1.1489618`.
İki dal orada tam birleşir (`d* = 1/L = 0.0744809`).

`B < B*`'ta unstructured'ın indeksi ucuzladığı için tile avantajı **erir**.
`B*` avantajın tam korunduğu en düşük bütçedir — kapalı formda bilinen bir sınır.

**Kapsam notu:** E8P bandının tamamı (`1.40–1.80`) `B*`'ın üzerinde, ve tile
hiçbir bütçede liste rejimine girmez. `min(bitmap, liste)` bir doğruluk
düzeltmesi ve bir bulgudur, deney tasarımı değişikliği değil.

`B < B*` bölgesi **perplexity ölçülmeden**, yalnızca muhasebeden türetilmiş bir
eğri olarak raporlanır: orada `d < 0.12`, kalite zaten ölçülemez.

### 0.3.2 `W × T` — ön-kaydedilen belirsizlik

Birinci koşul:

$$\left[|Q'(d)| + |\partial_d\tau|\right]\cdot \frac{1}{T^2 W} = \partial_T \tau$$

`W` küçüldüğünde sol taraf büyür → `T*` **büyümeli**. Ters yönde çalışan tek şey:
düşük `W`'de `d` yüksektir (eğrinin düz kısmı, `|Q'|` küçük). **İşaret belirsizdir.**

Ayrıca `is_live` (§3.5) ile birleşince birincil bantta yalnızca dar bir `W`
aralığı canlı kalır. Log aralıklı `T` ızgarasında `T*`'ın kayması için etkinin
en az 2× olması gerekir.

> **Kayıt:** `W × T` ön-kayıtta birincil öngörü **değildir**. `Δ(T)` eğrileri
> `W` başına raporlanır, `argmin` kayması iddia edilmez.

### 0.4 Delta cümlesi

> Guo et al. (SC 2020) tile-wise sparsity'yi CNN/BERT için, yeniden eğitimle
> tanımladı. VENOM (SC 2023) N:M ailesini donanım desteğiyle genişletti.
> **OBR (arXiv:2509.11177)** ortak quantization+seyrekleştirmeyi, rotasyon
> uyumsuzluğunu *onararak* çözdü. Bu çalışma farklı bir soru soruyor:
> **verili bir bit bütçesi altında `(survivor quantizer, granularity, density)`
> üçlüsünü birlikte optimize etmek** — özellikle bütçe PTQ tabanının altına
> indiği rejimde.
>
> **OBR'den farkımız:** OBR verili bir sırayı onarır; biz **sırayı değiştiririz**
> (§4.6) ve granülerliği serbest değişken yaparız.
>
> ⚠️ **VENOM'dan farkımız dar, ve dürüstçe yazılmalı.** V:N:M formülü M0'da
> doğrulandı ve yapısal bir şey ortaya çıkardı: VENOM `R×K` matrisi `V×M`
> bloklara böler ve **her bloğun `V` satırı ortak bir sütun seçimi paylaşır**
> (`column-loc`, §3.2). Bu tam olarak Axis B'nin `T=V` hali. Yani indeksin
> satır grubuna amortize edilmesi — manşet özdeşliğimizin arkasındaki mekanizma
> — **yeni değildir**; VENOM'da zaten var ve maliyeti `1/V` ile düşüyor.
>
> Geriye kalan gerçek farklar:
> ① VENOM'un sütun seçimi **blok-yereldir** (her `M`'den 4 tane); bizimki satır
> boyunca serbest — bizimki bir üst küme.
> ② VENOM'un tile-içi yoğunluğu donanım tarafından `N/4`'e sabitlenmiştir;
> bizimki serbest, yani aile `d`'de de sürekli.
> ③ VENOM `V`'yi donanıma göre seçer (32/64/128); biz `V`'yi **optimize edilen
> serbest değişken** yapıyoruz ve bir bit bütçesine karşı ölçüyoruz.
>
> Yani katkı *"indeksi amortize etmek"* değil, **`(T, d)` düzleminin tamamını
> verili bir bütçe altında taramak** ve `T`'nin iç optimumunu aramaktır.
>
> Katkının çekirdeği: `1 − 1/T` özdeşliği ve `B*` erime duvarının kapalı formu,
> granülerlik vergisi yüzeyi `τ(T,d)`, ve **maske-önce sırasının kafes VQ'yu
> seyrek matriste mümkün kılması**.

### 0.5 Incoherence processing — risk değil, çekirdek mekanizma

> **v6 bunu tersinden okuyordu.** "Seyrek matris global rotasyondan sonra
> yoğunlaşır" diyerek QuIP#/QTIP'i toptan eledi. İddia *maskeden önce* uygulanan
> rotasyon için doğru; toptan eleme olarak yanlış.

**Belgelenen başarısızlık gerçek ve serttir.** OBR Tablo 1, Llama2-7B, W-A-KV
4-bit: QuaRot + Wanda %50 seyreklik → **5868.24 ppl**. Ve arXiv:2603.18426:
*"Rotation amplifies pruning effects."* Mekanizma: rotasyon büyüklük dağılımını
düzleştirir, budama ise yoğunlaşmış enerjiden beslenir.

**Ama bu bir sıra problemidir.** Maske dondurulduktan sonra rotasyon maskeyi
bozamaz — seçim çoktan yapılmıştır. Doğru sıra (§4.6):

```
skorla → maske seç (döndürülmemiş bazda) → dondur
       → kompaktla → döndür → kafes VQ → telafi
```

Belgelenen başarısızlık modu bu sırada **oluşamaz**.

**Bedeli `T`'ye bağlıdır.** QuaRot'un rotasyonları komşu katmanlara fold edilir,
bedavadır. Kompaktlanmış survivor'a uygulanan rotasyon fold edilemez — her tile
farklı sütun kümesine sahip, gather'dan *sonra* uygulanmalı:

| T | rotasyon ek yükü (GEMV'e oran, `log₂(k)/T`) |
|---|---|
| 1 | ~11.5× — imkânsız |
| 16 | ~0.75× — pahalı ama mümkün |
| 64 | ~0.19× — makul |
| max | ~0 — bedava |

**Rotasyonun faydası nereden geliyor:** ağırlık dağılımının izotropik
olmamasından. Ölçüldü (16×64 blok, korelasyonlu Hessian):

| blok | rotasyon, düz yuvarlama | rotasyon, LDLQ |
|---|---|---|
| Gaussian | +17.5% (zarar) | +4.8% (zarar) |
| kalın kuyruklu | **−61.7%** | **−39.0%** |

Survivor'lar tanım gereği dağılımın kalın kuyruğudur. Rotasyon bu hatta yerini
bu yüzden hak ediyor — ve maliyeti `log₂(k)/T` olduğu için **ancak büyük `T`'de
karşılanabilir.** Granülerlik ekseninin ikinci kuvveti budur.

> ⚠️ **Açık varsayım:** E8P'nin kompaktlanmış survivor alt-matrisinde
> `vq_bits ≈ 2.0` kalitesini koruduğu **varsayılıyor, doğrulanmadı**.
> Erken uyarı kuralı ve geri dönüş yolu `preregistration.md` §9.1'de.

### 0.6 Roofline

Batch=1 decode bant genişliği sınırlı: `süre ≈ taşınan bayt / bant genişliği`.
`accounting.roofline_bytes` her şema için bayt sayısını verir. Manşet:
*"tile-sparse'ın roofline'ı X bayt daha düşük"* — **alt sınır** olarak sunulur.

⚠️ Kendi şemamız için kernel yok (§8), ve rotasyon roofline ile gerçek latency
arasındaki açığı büyütür. **Hız iddiası yapılmaz.**

---

## 1. Ortam

`python>=3.10, torch>=2.4, transformers, datasets, accelerate, lm-eval, numpy, scipy, pytest`

Tek A100 40GB yeterli (7B/13B).

---

## 2. Repo yapısı

```
tilesparse/
├── accounting.py     bit bütçeleri, özdeşlik, B*, canlı bant
├── scoring.py        §4.3'ün saliency'leri
├── tiling.py         tile bölümlemesi, dondurulmuş maske
├── prune.py          maske seçimi + ileriye telafi  (H1 assert'i burada)
├── compact.py        survivor'ları yoğun bloklara topla
├── rotation.py       maske-koruyan rotasyon
├── quantize.py       E8P codebook + LDLQ
├── calibrate.py      sıralı kalibrasyon, LayerProblem
├── hf_llama.py       HuggingFace adaptörü (blok 0 girdilerini yakalar)
├── eval/perplexity.py  ppl + protokol koruması
├── experiments/m1_gates.py
├── preregistration.md   ← M1'den ÖNCE dondurulur
├── docs/{audit,gate_a_dry_run,spec_v7}.md
└── tests/            322 test
```

---

## 3. Muhasebe

### 3.1 Şema başına `q_overhead`

```python
Q_OVERHEAD_SCALES_WITH_DENSITY = {
    'dense': True, 'unstructured': True, 'tile': True, 'structured': True,
    'nm': False, 'vnm': False,
    # 'vq' YOK: VQ dalı q_over kullanmaz, vq_bits kullanır (§3.2)
}
```

`q_over = (16 + wb)/128` — grup başına FP16 ölçek + `wb`-bit sıfır noktası.

Bu konvansiyon M1'in işaretini çevirir: `True` olsaydı 2:4 @ 4-bit = 3.078125
(dense 3-bit'in **altında**), `False` ile 3.156250 (**üstünde**), offset +0.248%.
**Her tabloda yazılır.**

### 3.2 İndeks modeli

| Şema | practical | not |
|---|---|---|
| `dense`, `structured` | 0 | |
| `unstructured` | `min(1, d·log₂(n_idx))` | bitmap / sabit-genişlikli liste kaskadı |
| `tile` (T) | `min(1, d·log₂(n_idx))/T` | |
| `nm` | `min(⌈log₂M⌉·N/M, log₂C(M,N)/M)` | **ikisi de pratiktir** |
| `vnm` | `2N/M + 4⌈log₂M⌉/(V·M)` | VENOM. **`V` bir row-tile** — §0.4 |

> **v6 burada fazla iddialıydı.** "Rastgele erişim `H(d)`'yi yasaklar, entropi
> kodlaması kullanılamaz" diyordu. Doğru değil: **blok-yerel sabit-sayılı
> kodlama** hem rastgele erişimlidir (blok başına O(1) çözülür, önceki blokları
> okumak gerekmez) hem de `H(d)`'ye yaklaşır. `2:8` kombinatoryal kodlaması
> `0.600919` bit — bitmap'in 1.0'ından ucuz.

**Makaleye girecek doğru paragraf:**

> Rastgele erişim entropi sınırını yasaklamaz; maske üzerinde blok yapısı
> dayatarak ona yaklaşılabilir — ama o yapı kendi kalite bedelini doğurur.
> İndeks ucuzluğu ile maske özgürlüğü arasındaki takas kaçınılmazdır, ve `T`
> bu takasın tek ekseni değildir. Bu çalışma tile eksenini ölçer; N:M ailesi
> aynı takasın ayrık bir kafesidir.

**Kapsam:** `d·log₂(M) < 1` ancak `d < 1/log₂(M)` iken. Birincil bantta
(`d = 0.25–0.88`) bitmap zaten optimaldir, yani **bu düzeltme birincil bant
sonuçlarını değiştirmez.** `B*`'ı ve delta cümlesini etkiler.

**`n_idx` katman-bağımlıdır:** Axis B'de `d_in`, Axis A'da `n_out`. Bitmap
rejiminde indeks `1/T` olduğundan katmandan bağımsızdır; katman-bağımlılık
yalnızca `B < B*`'ta devreye girer.

**VQ:**

```
vq_bits = idx_bits/dim + codebook_amortization
QuIP# E8P : 16/8 + 0     = 2.000   ← yapılandırılmış codebook, 256 girdi (~1 KiB)
AQLM 1x16 : 16/8 + 0.186 = 2.186   ← modele özel eğitilir, amortize edilir
```

E8P doğrulandı (`quantize.py`): 227 negatif-olmayan yarı-tam-sayı örüntü
(`‖s‖² ≤ 10`) + 29 dolgu (`‖s‖² = 12`) = 256 kaynak tablosu; kodsözcüğü
8 bit indeks + 7 bit işaret + 1 bit ±¼ kayma; `256 · 2⁷ · 2 = 2¹⁶` → **tam 2
bit/ağırlık**. Yarı-tam-sayı seçimi tesadüf değil: koordinatlar asla sıfır
olmadığından 128 işaret çevirmesinin hepsi ayrı vektör verir.

> Checkpoint'in **gerçek dosya boyutundan** doğrulanmalı (katman ölçekleri ve
> Hadamard seed'leri dahil).

### 3.3 API

`accounting.py`: `bits_per_position`, `anchor_budget_to`, `density_for_budget`,
`scheme_floor`, `b_star`, `d_star`, `in_bitmap_regime`, `tile_density_advantage`,
`live_band`, `live_diagnostics`, `is_live`, `budget_matched_grid`, `roofline_bytes`.

**Kural 1:** Her bütçe bir yoğun baseline'ın **tam** maliyetine çapalanır.
Yuvarlak sayıya asla çapalanmaz.

**Kural 2:** Density'si sabit şemalar (`nm`, `vnm`, `vq_dense`, `dense`) kendi
doğal maliyetlerinde raporlanır; tabloda **işaretli offset sütunu**;
`|offset| > %1` → **bayraklanır**. `density_for_budget` onlar için `ValueError`.

**Kural 3:** Gerçekleşen yoğunluk raporlanır, istenen değil. Tile başına
yuvarlama ve `align=8` hizalaması `d`'yi kaydırır; bit sayısı **gerçekleşenden**
yeniden hesaplanır.

### 3.4 Kabul testleri

```
W:  4-bit = 532/128 = 4.156250 · 3-bit = 403/128 = 3.148438 · 2-bit = 274/128 = 2.140625
    E8P   = 2.000000 (amortizasyon 0)

log2(11008) = 13.4262648          ← v6: 13.4262102 (YANLIŞ)

ÖZDEŞLİK (geçerlilik alanı assert'li: B ≥ B*(T)):
  d(tile,T) − d(unstr) == (1−1/T)/W   ± 1e-12
  T=16: 4-bit 0.2255639 · 3-bit 0.2977667 · 2-bit 0.4379562 · E8P 0.4687500

B* SÜREKLİLİĞİ:
  b_star(4, 11008)      == 1.3095612     ← v6: 1.3095620
  b_star(4, 4096)       == 1.3463542
  b_star(E8P, 11008)    == 1.1489618
  B*'ta iki dal aynı avantajı verir      ± 1e-9

ÇAPA 1 = 3.148438 (v6'nın tablosu, İKİ HÜCRE DÜZELTİLDİ):
  T=1   0.516917        T=4   0.697368   ← v6: 0.696992 (YANLIŞ)
  T=16  0.742481        T=max 0.757519   ← v6: 0.757954 (YANLIŞ)

E8P BANDI, B = 1.50:
  T=1 0.250000 · T=2 0.500000 · T=4 0.625000 · T=8 0.687500
  T=16 0.718750 · T=32 0.734375 · T=max 0.750000        (hepsi dyadic)

nm(2,4) 4-bit == 3.156250 (offset +0.248%) · 2-bit == 2.140625 (offset 0)

İNDEKS:
  index('unstructured', d=0.50, n=11008) == 1.0
  d*(11008) == 0.0744809
  nm_index(2,8,'combinatorial') == 0.600919   < bitmap 1.0

TABAN (v6'dan düzeltildi):
  scheme_floor('unstructured', 4, 11008) == 0.0    (1.0 DEĞİL)
  scheme_floor('tile', 4, 11008, T=16)   == 0.0
  density_for_budget('unstructured', 0.60, 4, 11008) == 0.0341248
                                                        ← v6: 0.0341271 (YANLIŞ)

RAISE: density_for_budget('nm'|'vnm'|'vq_dense'|'dense', ...) → ValueError
       index_bits('vnm', ...) → NotImplementedError
```

> **Golden sabitler elle yazılmaz.** `tests/golden.py` bunları tam kesirli
> aritmetikle (`Fraction`) bağımsız olarak türetir; `accounting.py` genel
> dispatch ile hesaplar. İki bağımsız yol, tek cevap. v6'nın hata sınıfı
> (elle yazılmış ondalık) yapısal olarak imkânsız.

### 3.5 Canlı bant filtresi

**Canlı tanımı:** `d(T=1) > 0.2` **ve** `d(T=max) < 0.9`.

| survivor `W` | canlı bant |
|---|---|
| GPTQ 4-bit | 1.831250 – 3.740625 |
| GPTQ 3-bit | 1.629687 – 2.833594 |
| GPTQ 2-bit | 1.428125 – 1.926562 |
| **E8P** | **1.400000 – 1.800000** |

> **Yapısal sonuç:** E8P ve GPTQ-4bit aileleri canlı bütçelerde **örtüşmez.**
> Aynı canlı hücrede bütçe-eşleştirilemezler — survivor quantizer'ı bütçe
> rejimini belirler. M2'nin grid'i buna göre kurulur.

> **`is_live` iki soruyu ayırır** (v6 karıştırıyordu). "Bu hücre bir
> **granülerlik** probu mu?" ile "bu satır raporlanabilir mi?" farklı sorular.
> Coarse ucu yoğun olan bir konfigürasyon granülerlik probu değildir ama
> geçerli bir Kapı A satırı olabilir. `live_diagnostics` gerekçeyi verir;
> satırlar sessizce düşürülmez.

### 3.6 Modellenmeyen kalite maliyetleri — ZORUNLU ABLASYONLAR

1. **`q_over` grup konvansiyonu.** `g=128 hayatta kalanlar` vs `g=128 orijinal
   pozisyonlar`. ⚠️ İki konvansiyon **farklı bit maliyetine** sahiptir, saf
   kalite ablasyonu değildir → **iki koşulda** koş: (a) aynı `d`, farklı bütçe;
   (b) aynı bütçe, farklı `d`.
2. **Quantization hatası ↔ maske hatası ayrımı.** Düşük `d`'de survivor'lar
   dağılımın büyük ucu; post-prune quantization hatası **ayrı raporlanır**.
   §0.5'in açık varsayımını izleyen ölçüm budur.
3. **Tensor-core hizalaması.** `align=8` (LDLQ zaten gerektiriyor). `n_idx=11008`'de
   %0.07, küçük katmanlarda kaba — offset her zaman raporlanır.

---

## 4. Yöntem

### 4.1 İki eksen

**Axis B — row-tile (ÖNCE BU).** Satırlar `n/T` tile'a bölünür; tile bir sütun
kümesi seçer. `T=1` → unstructured, `T=n` → girdi-kanalı budama.

**Axis A — column-tile.** Transpozu: sütunlar tile'a bölünür, tile satır kümesi seçer.

### 4.2 `T=max` — uygunluk ve koordinasyon

| Katman | Girdi | Çıktı | Axis A (`T=d_out`) | Axis B (`T=n`) |
|---|---|---|---|---|
| q/k/v_proj | residual ✗ | head dims ✓ | ✓ | ✗ |
| o_proj | head dims ✓ | residual ✗ | ✗ | ✓ |
| gate/up_proj | residual ✗ | FFN inter ✓ | ✓ | ✗ |
| down_proj | FFN inter ✓ | residual ✗ | ✗ | ✓ |

$$\varepsilon_{\text{FFN}}(k) = \frac{\varepsilon_{\text{gate}}(k)}{c_{\text{gate}}} + \frac{\varepsilon_{\text{up}}(k)}{c_{\text{up}}} + \frac{\varepsilon_{\text{down}}(k)}{c_{\text{down}}}$$

> ⚠️ Üç terim farklı Hessian'lardan gelir → `c_ℓ` normalizasyonu **zorunlu**.
> ⚠️ **SwiGLU:** kanal katkısı `SiLU(gate_k)·up_k` — çarpımsal. Toplam bir
> **yaklaşımdır**; makalede böyle yazılır.
> ⚠️ **Attention koordinasyonu formülleştirilmeli:** `v_proj` çıkışı ↔ `o_proj`
> girişi eşleşmesi; GQA altında bir KV head boyutunu budamak gruptaki *tüm*
> query head'lerini etkiler; q/k çıkış boyutları **RoPE çiftleri halinde**
> budanmalıdır. v6 bunları yalnızca ima ediyordu.
> ⚠️ `T=max` **ayrı kod yolu** ve **yetkin structured baseline** gerektirir.

### 4.3 Saliency

Dört formül aslında **iki metriktir**; eksenler yalnızca toplama yönünü değiştirir:

```
ağırlık-başı, Wanda : (|w_ij| · ‖X_j‖)²
ağırlık-başı, OBS   : w_ij² / [H⁻¹]_jj

Axis B: tile'ın SATIRLARI üzerinden topla  → [n_tiles, n_in]
Axis A: tile'ın SÜTUNLARI üzerinden topla  → [n_tiles, n_out]
```

> **v6 burada kendi kuralını çiğniyordu.** "Eksen karşılaştırmasında saliency
> sabit" diyor, ama Axis A'ya tam grup-OBS, Axis B'ye köşegen yaklaşım
> veriyordu — farklı doğruluk seviyeleri. O karşılaştırma granülerliği değil,
> saliency doğruluğunu ölçerdi.
>
> **v7'de eksenler metriği paylaşmak zorundadır** (yapısal, `scoring.py`).
> Tam grup-OBS `½·tr(W_S [(H⁻¹)_SS]⁻¹ W_Sᵀ)` **ablasyon olarak** kalır.

`(H⁻¹)_SS` ≠ `(H_SS)⁻¹` (§7, tuzak 16) · Wanda ve OBS aynı ölçekte değil, tek
`λ` altında karıştırılmaz · Wanda'nın karesi bilinçli bir seçimdir (grup
toplamını OBS'in kuadratiğiyle hizalar) ve L1'e karşı ablasyonlanır.

### 4.4–4.6 Tahsis, sıra, telafi

`allocate(mode='per_tile_uniform', freeze='per_tile_local', block_size=128, align=8)`.
Dengeli kümeleme: **Sinkhorn / min-cost flow** — Hungarian değil, düz k-means değil.
Permütasyon yalnızca FFN ara boyutu ve head-içi; RoPE çiftleri korunur.

**Hattın sırası — tasarım değişmezi:**

```
skorla → maske seç (döndürülmemiş bazda) → dondur
       → kompaktla → döndür → LDLQ → telafi
```

Maske **her zaman** döndürülmemiş bazda seçilir. `prune()` bunu assert eder,
konvansiyona bırakmaz. Telafi yalnızca **ileriye**. Sıralı kalibrasyon zorunlu,
ve istatistikler **sıkıştırılmış** modelden gelir (§7, tuzak 20).

> **Sıra ablasyonu M2'ye çekildi.** arXiv:2603.18426 zayıf pertürbasyonun önce
> gelmesini öneriyor (quantize-then-prune), spec'in varsayılanı tersi. Ölçüldü:
> 2-bit quantizer, OBS telafisinin kazancını geri alıyor (bazı `T`'lerde işaret
> çeviriyor) — LDLQ bunu düzeltiyor ama sıra sorusu yine de ampirik.

---

## 5. Milestone'lar

### 5.1 M0 — Muhasebe, `τ` yüzeyi, uçuş öncesi

- [x] `accounting.py` tam; §3.4'ün tümü geçiyor
- [x] Kapı A kâğıt üstü provası (`docs/gate_a_dry_run.md`)
- [x] E8P codebook doğrulandı (227+29, 2¹⁶, 2.0 bit)
- [x] HuggingFace adaptörü + katman-akışlı eval — 7B, 8 GB kartta, 4 dk/eval
- [x] `vnm` formülü VENOM'dan doğrulandı — ve §0.4'ü daralttı
- [ ] VQ checkpoint maliyetleri **dosya boyutundan** ölçüldü
- [x] **Protokol kimliği** (2026-08-21): 2048 → **5.4675**, 4096 → **5.1143**;
      ikisi de yayımlanmıştan <0.006 sapıyor. **seqlen 4096 birincil** donduruldu.
- [ ] `tau_sweep.py`: `Q(d)` 3 seed, `τ(T,d)` **eşleştirilmiş** 1 seed
- [ ] **Transfer pilotu** — `τ`'yu bir noktada quantization'lı/suz ölç,
      ön-kayıt toleransını oradan türet (~2 GPU-saat)
- [ ] Minimum saptanabilir fark (Kapı B'nin gücü)
- [ ] **`preregistration.md` donduruldu ve commit edildi**

Maliyet: `Q` 5×3 + `τ` 25×1 + pilot ≈ **25 GPU-saat** (C4 eval'i dahil).

### 5.2 M1 — İKİ KAPI ⭐

**Model:** Llama-2-7B. **Eksen:** B. **Survivor:** E8P.
**Bütçeler:** 1.75 / 1.60 / 1.50 — üçü de canlı ve **2 bitin altında**.
**`T` ızgarası:** {1, 2, 4, 8, 16, 32, max}.

Yoğunluklar `preregistration.md` §2'de dondurulmuştur.

**Kapı A:** test edilen seyrek konfigürasyonların en iyisi, PTQ tabanı
referansını (QTIP 2-bit) geçiyor mu? Bütçe-eşleşmiş **değildir** ve bu kasıtlı:
karşılaştırma *"2 bitin altında, 2 bite karşı"*. A fortiori geçerlidir.

**Kapı B:** optimum `T` içeride mi, uçta mı? **Argmin değildir.** Üç koruma:
≥5 çekiliş, iç adaylar üzerinden Bonferroni, farklar üzerinde eşleştirilmiş
bootstrap. A fortiori **geçerli değildir** — hem Wanda hem SparseGPT ile koşulur.

> Karar tablosu ve okuma kuralları `preregistration.md` §6–§8'de.
> **Kapı B düşerse proje durmaz**; Kapı A'nın bağımsız değeri vardır.

### 5.3 M2 — Bütçe süpürmesi

Bütçeler `{1.80, 1.75, 1.65, 1.60, 1.50, 1.45}` — canlı bandın içi.
`T` ızgarası tam. `axis ∈ {A, B}` (saliency **her iki eksende de köşegen**).
Baseline: yoğun 2/3/4-bit (GPTQ + VQ), unstructured, 2:4, 4:8, V:N:M, magnitude,
structured, **OBR**. Sıra ablasyonu (prune-then-quant vs joint) burada.

**Ana metrik:** *"dense FP16'nın %X'i içinde kalmak için gereken minimum
bit/pozisyon"*, X ∈ {5,10,25}. Bootstrap ile belirsizlik yayılır;
**"aralıkta kesişim yok"** dalı tanımlıdır; `inf` yazılmaz.

**"Fark bütçe düştükçe büyüyor"** şartı burada test edilir — M1'de değil.

`B < B*` bölgesi: muhasebe-türevi avantaj eğrisi, perplexity ölçülmez.

### 5.4 M3 — Tam OBS
Hessian/Cholesky, `block_size` lazy-batch, blockwise (SparseGPT) maske seçimi,
`obs_B` → `obs_A`, `c_ℓ` + `global_lambda`, §3.6'nın üç ablasyonu.

### 5.5 M4 — Healing
`<1.5`'te tek-atışlık PTQ geçersizleşir. Kısa LoRA (~100M token) **açık/kapalı
ablasyonu** zorunlu.

### 5.6 M5 — Ablasyonlar ve ölçek
Tile oluşturma · telafi · `c_ℓ` · `freeze` · katman tipi kırılımı ·
**7B → en az bir 13B** · Llama-2 *ve* Llama-3.x.

---

## 6. Protokol

**Protokol ailesi kuralı — v7'nin en sert kuralı.** Llama-2-7B için literatürde
iki uyumsuz aile var (dense **5.12**: Wanda, QTIP · dense **5.47**: QuIP#,
SliceGPT). Aynı yöntemin sayısı 0.47 ppl değişiyor — Kapı B'nin çözmeye
çalıştığı etkiden büyük.

> **ÖLÇÜLDÜ (2026-08-21).** Ayrımın sebebi dizi uzunluğuymuş:
>
> | seqlen | bizim | yayımlanan | fark |
> |---|---|---|---|
> | 2048 | **5.4675** | 5.47 | −0.0025 |
> | 4096 | **5.1143** | 5.12 | −0.0057 |
>
> İki aile de yeniden üretilebiliyor, yani kural "birini seç" değil
> **"pencereyi sabitle"**. Yayımlanmış bir sayı ancak bizim **aynı seqlen'de**
> aldığımız bir sayının yanına konabilir; `eval.perplexity.compare` uyuşmazsa
> **hata fırlatır**.
>
> **seqlen 4096 birincil** (ön-kayıt §9). Gerekçe: `dense-5.12` ailesi hem budama
> baseline'larını hem QTIP/QuIP#'i taşıyor — Kapı A'nın rakibi orada. 2048
> ikincil; SliceGPT ve QuaRot yalnızca orada.

Her tabloda: **bit/pozisyon**, **`q_over` konvansiyonu**, **çapa**, **offset**,
**`n_idx`**, **protokol ailesi**, **seqlen**, **convention**. Eş-sparsity yasak.

WikiText-2 + C4 · zero-shot 5 görev (kapılara girmez) · GSM8K/MMLU ≥7B ·
C4 128×2048 kalibrasyon, **seed = çekiliş** · Llama-3.x yayımlanmış sayılarla
kıyaslanmaz.

Her run: git hash, config, seed, metrikler, bit/poz, konvansiyon, çapa, offset,
`n_idx`, gerçekleşen `d` → `results/*.json`.

---

## 7. Tuzaklar — YAPMA

1. **Özdeşliği geçerlilik alanı olmadan kullanmak.** Yalnızca `B ≥ B*(T)`.
2. **Manşeti "oran büyüyor" diye yazmak.** Mutlak avantaj sabit.
3. **Karşı kuvveti atlamak.** Düşük `d`'de kısıt bedeli artar.
4. **`practical` indeksi gerekçesiz bırakmak** veya **"entropi kodlaması
   kullanılamaz" demek.** Blok-yerel kodlama hem pratik hem ucuz (§3.2).
5. **"Unstructured 1.0'ın altına inemez" demek.** *Bitmap* inemez.
6. **Bütçeyi yuvarlak sayıya çapalamak** · **offset'i gizlemek** ·
   **istenen yoğunluğu raporlamak.**
7. **Kapı B'yi "tile > unstructured" diye tanımlamak.** Trivial.
8. **"Fark büyüyor"u M1'de test etmek.**
9. **Eşiği ölçülen etkiden türetmek.** Döngüsel. `τ` yüzeyinden tahmin et.
10. **"Tez yanlış" ile "model yanlış"ı karıştırmak.**
11. **Kapı B düşünce projeyi durdurmak.**
12. **A fortiori'yi Kapı B'ye uygulamak.**
13. **Kapı B'yi argmin ile karara bağlamak** veya **3 seed ile karar vermek.**
14. **Dejenere hücreleri granülerlik verisi saymak** — ama `is_live`'ı
    raporlanabilirlik filtresi sanmak da yanlış (§3.5).
15. **`W × T` öngörüsünü kaydetmek.** İşaret belirsiz.
16. **`(H_SS)⁻¹`** · **telafiyi geriye uygulamak.**
17. **Üç FFN ölçeğini ham toplamak** · SwiGLU'nun çarpımsal olduğunu yazmamak ·
    **attention koordinasyonunu formülleştirmemek.**
18. **Düz k-means / Hungarian** · **Wanda+OBS'yi tek `λ`'da.**
19. ⭐ **Rotasyondan sonra budamak.** QuaRot+Wanda %50 → 5868 ppl.
    Maske **her zaman** döndürülmemiş bazda seçilir.
20. **Kalibrasyonu yoğun modelden** · **residual'da permütasyon** ·
    **structured baseline'sız `T=max`.**
21. **RTN** (smoke test hariç) · **sabit-density şemalarda `density_for_budget`.**
22. **Ön-kaydı sonradan değiştirmek** · **bootstrap'sız kesişim.**
23. ⭐ **Protokoller arası sayı alıntılamak.** 0.47 ppl fark, ölçtüğümüz
    etkiden büyük.
24. ⭐ **Rotasyonu Hessian-farkında yuvarlama olmadan kullanmak.** Maliyeti
    öder, faydasını toplamaz (§0.5).

---

## 8. Kapsam dışı

Kendi şemamız için kernel · tam yeniden eğitim (M4 hariç) · dinamik sparsity ·
aktivasyon sparsity · KV cache sıkıştırma · blok-bazlı yeniden yapılandırma.
*(Roofline alt sınırı + hazır implementasyon ölçümü kapsam içi; hız iddiası değil.)*

---

## 9. Referanslar

**PTQ tabanı:** QuIP# arXiv:2402.04396 · QTIP arXiv:2406.11235 · AQLM
arXiv:2401.06118 · QuaRot arXiv:2404.00456 · Kuzmin et al. NeurIPS 2023

**Ortak prune+quant:** **OBR arXiv:2509.11177** · sıra: arXiv:2603.18426 ·
SLiM arXiv:2410.09615 · SparseGPT birleşik modu

**Structured:** SliceGPT arXiv:2401.15024 · LLM-Pruner · ShortGPT · Sheared LLaMA

**Doğrudan öncüller:** Guo et al. SC 2020 arXiv:2008.13006 ·
VENOM/Spatha SC 2023 arXiv:2310.02065 · UnionSparse arXiv:2608.09291

**Yöntem temeli:** SparseGPT · GPTQ · Wanda arXiv:2306.11695 · OBS (1993) ·
Pool & Yu NeurIPS 2021 · RIA ICLR 2024

**Uyarlanabilir tahsis:** OWL · PALS arXiv:2607.07557 · AlphaPruning · BESA

---

## Ek A — v6'dan değişenler

| # | Değişiklik | Sebep |
|---|---|---|
| 1 | **Survivor quantizer GPTQ-4bit → E8P** (`W`: 4.156 → 2.000) | Kapı A provası: GPTQ-4bit survivor'larla her yerde kaybediliyor. `B=1.5`'te `T=16` seyrekliği %65 → %28 |
| 2 | **Bant 3.148/2.141 → 1.75/1.60/1.50** | E8P'nin canlı bandı 1.40–1.80. Çalışma kendiliğinden 2 bitin altına kaydı — motivasyonun tuttuğu yere |
| 3 | §0.1 **bağlam fonksiyonu** olarak yeniden yazıldı | v6'nın "eşik 2.2–2.3" iddiası kendi aritmetiğinden çıkmıyordu (4k'da 2.46–2.65) |
| 4 | **§0.5 tersine döndü:** risk → çekirdek mekanizma | Rotasyon maskeden *sonra* uygulanabilir. Başarısızlık bir sıra problemi (QuaRot+Wanda 5868 ppl) |
| 5 | **H1 değişmezi eklendi** ve koda assert edildi | Yukarıdakinin doğrudan sonucu |
| 6 | **LDLQ zorunlu hale geldi** (tuzak 24) | Rotasyon Hessian-farkında yuvarlama olmadan maliyeti öder, faydasını toplamaz |
| 7 | Çapa 1 tablosunda **iki hücre düzeltildi** | 0.696992 → 0.697368 · 0.757954 → 0.757519 |
| 8 | **`log2(11008)` düzeltildi** (13.4262102 → 13.4262648) | `b_star`, `B<B*` avantajları, `d*` zincirleme etkilendi |
| 9 | `density_for_budget(0.60)` 0.0341271 → **0.0341248** | Aynı kök |
| 10 | Golden sabitler **türetiliyor**, elle yazılmıyor | v6'nın hata sınıfı yapısal olarak imkânsız hale geldi |
| 11 | §3.2 **"entropi kodlaması kullanılamaz" iddiası geri çekildi** | Blok-yerel sabit-sayılı kodlama hem rastgele erişimli hem `H(d)`'ye yakın |
| 12 | `nm`'in kombinatoryal indeksi **`practical` sütununa** taşındı | Blok başına O(1) çözülür |
| 13 | **`is_live` iki amaca ayrıldı** | Granülerlik probu ≠ raporlanabilirlik |
| 14 | §4.3 **tek metrik, iki toplama yönü** | v6 eksenlere farklı doğrulukta saliency veriyordu; karşılaştırma granülerliği ölçmezdi |
| 15 | **Kapı B: ≥5 çekiliş + Bonferroni + eşleştirilmiş bootstrap** | 3 seed ile `gate_b` saf gürültüde "interior" verdi. §6'nın "seed ≥ 3"ü bu kapı için yetersiz |
| 16 | Ön-kayıt **ayrı belgeye** taşındı ve dondurma listesi eklendi | Tolerans ölçüm gerektiriyor; dondurma görünür bir olay olmalı |
| 17 | **Protokol ailesi kuralı** (tuzak 23) | Aynı yöntem iki ailede 0.47 ppl fark ediyor |
| 18 | **Sıra ablasyonu M3 → M2** | 2-bit quantizer telafinin kazancını geri alıyor |
| 19 | **OBR öncül olarak eklendi**, §0.4 güncellendi | Doğrudan komşu iş; farkımız sırayı değiştirmek |
| 20 | `align=8` (tensor-core + LDLQ) | v6'nın muhasebesinde hizalama yoktu |
| 21 | Attention koordinasyonu **formülleştirilmeli** olarak işaretlendi | v6 yalnızca ima ediyordu |
| 22 | `d` notasyon çakışması giderildi (`d_out`) | Kod ajanında hata üretiyordu |
| 23 | M0 maliyeti 15–17 → **~25 GPU-saat** | C4 eval'i ve transfer pilotu dahil |
| 24 | **`vnm` formülü dolduruldu**, §0.4 daraltıldı | VENOM'un `V`'si bir row-tile: indeks amortizasyonu yeni değil |
