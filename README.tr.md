# subfloor

*[English](README.md) · Türkçe*

**Sparsity Below the Quantization Floor** — bit bütçesi PTQ tabanının altına
indiğinde `(survivor quantizer, granularity, density)` üçlüsünü birlikte
optimize etmek.

> **Durum.** Hat uçtan uca çalışıyor ve gerçek Llama-2-7B'ye bağlı: dense
> perplexity yayımlanmış değerin 0.006 içinde yeniden üretiliyor, tam model
> sürücüsü gerçek ağırlıkları sıkıştırıyor, kesilen bir koşu kesilmeyenin
> cevabına varıyor. **Sıkıştırılmış model artık ölçüldü — iki noktada:**
> 1.5 bit / d=0.719 → **47.64** ppl, 2.0625 bit / d=1.000 → **13.97**
> (WikiText-2, seqlen 4096, T=16, çekiliş 0). İkincisi hattın kendi ~2-bit
> referansı: orada QuaRot-GPTQ 2-bit'in 1.6× önünde, QuIP#'in 2.1× gerisinde —
> yani hat bozuk değil. Ama **Kapı A bu noktada düşüyor**: 0.5625 bitlik
> tasarruf hattın kendi referansına karşı **3.4× ppl** ödetiyor. Bütçe
> ekseninin ortası hâlâ boş.
> Ayrıntı: [`docs/STATUS.md`](docs/STATUS.md).

---

## Soru

Yoğun post-training quantization'ın pratik bir tabanı var, ~2 bit civarı. Altına
inildiğinde çöküyor: Llama-2-7B'de QuaRot-GPTQ 2-bit **22.07** ppl, QuIP# 2-bit
**6.66** (dense 5.47). Seyrekliğin tabanı ise kullanılan **indeks formatına**
bağlı — bitmap 1 bit/pozisyonun altına inemez, ama tile başına paylaşılan bir
indeks `1/T`'ye iner.

Bu neden önemli: bit bütçesi doğrudan bağlam uzunluğudur. Llama-2-70B, 24 GiB
kart, 22.5 GiB kullanılabilir:

| bit/pozisyon | sığan bağlam |
|---|---|
| 2.0 (PTQ tabanı) | ~15.6k |
| 1.5 | ~28.4k |

Yani 2 bitin altı, bağlamı ikiye katlamak demek.

## Çekirdek özdeşlik

`W` survivor başına bit, `d` yoğunluk, `T` tile boyutu. Bitmap rejiminde:

```
d(T) − d(1) = (1 − 1/T) / W          ← bütçeden bağımsız, SABİT
[d(T) − d(1)] / [d(∞) − d(1)] = 1 − 1/T   ← W'den de bağımsız
```

Mutlak avantaj sabittir. Oranın büyümesi paydanın küçülmesindendir — bunu "oran
büyüyor" diye yazmak yaygın ve yanlış bir okuma.

Kaldıraç `W` küçüldükçe büyür: GPTQ-4bit'te `0.2256`, kafes VQ'da (`W=2.0`)
**`0.4688`**. Survivor quantizer'ı seçimi bu yüzden bütçe rejimini belirliyor.

## Tasarım değişmezi

```
skorla → maske seç (döndürülmemiş bazda) → dondur
       → kompaktla → döndür → LDLQ → telafi
```

**Maske her zaman döndürülmemiş bazda seçilir.** Rotasyon büyüklük dağılımını
düzleştirir, budama ise yoğunlaşmış enerjiden beslenir; rotasyonlu bazda budama
modeli yok ediyor (QuaRot+Wanda %50 seyreklik → **5868 ppl**, OBR Tablo 1).

Ama bu bir *sıra* problemi. Maske dondurulduktan sonra rotasyon onu bozamaz —
seçim çoktan yapılmıştır. `prune()` yanlış sırayla çağrılırsa **hata fırlatır**;
konvansiyona bırakılmadı.

---

## Kurulum ve çalıştırma

`transformers >= 5` zorunlu: `hf_llama.load_llama`,
`from_pretrained(dtype=...)` çağırıyor ve bu bir v5 anahtar sözcüğü.

```bash
pip install "torch" "transformers>=5" datasets numpy pytest
python -m pytest -q                    # 689 test
```

Sentetik smoke test — hattın tamamı, iki kapı dahil, GPU gerekmez:

```bash
python experiments/m1_gates.py --synthetic --n-out 64 --n-in 128 --draws 3 --budgets 1.5
```

Gerçek model, tek ızgara noktası (kalibre et → sıkıştır → ölç, blok
granülerliğinde checkpoint):

```bash
python -u experiments/m1_run.py --budget 1.5 --tile 16 --calib-seqlen 2048
```

Bu makinenin sabitlerini yeniden ölç — **başka bir kartta zorunlu**, çünkü
buradaki her zamanlama sabiti "bu makinede ölçüldü" diyor:

```bash
python experiments/m0_cost_model.py
python -u experiments/m0_lever_audit.py --build --rot-sweep
```

Ücretsiz bir bulut GPU'sunda koşmak için: [`cloud/README.md`](cloud/README.md).

---

## Yapı

| Modül | İş |
|---|---|
| `accounting.py` | bit bütçeleri, `1−1/T` özdeşliği, `B*` duvarı, canlı bant filtresi |
| `scoring.py` | saliency — iki ağırlık-başı metrik, iki toplama yönü |
| `tiling.py` | tile bölümlemesi, dondurulmuş maske (`T=1` unstructured, `T=max` structured) |
| `prune.py` | maske seçimi + ileriye telafi; tasarım değişmezinin assert'i |
| `compact.py` | survivor'ları tile başına yoğun bloklara topla |
| `rotation.py` | maske-koruyan rotasyon, `kron(RHT(2^a), orthogonal(m))` |
| `quantize.py` | QuIP# E8P codebook + LDLQ (Hessian-farkında yuvarlama) |
| `calibrate.py` | sıralı kalibrasyon; istatistikler **sıkıştırılmış** modelden |
| `eval/perplexity.py` | ppl + protokol koruması |
| `eval/streamed.py` | GPU'ya sığmayan model için katman-akışlı ppl |
| `hf_llama.py` | HuggingFace adaptörü — blok 0'ın girdilerini yakalar |
| `experiments/m1_gates.py` | M1'in iki kapısı; hattın kendisi (`run_config`) |
| `experiments/m1_run.py` | **tam model sürücüsü** — blok granülerliğinde checkpoint ve devam |
| `experiments/m0_cost_model.py` | ölçülen eğrilerden gerçek koşu maliyeti |
| `experiments/bench_guard.py` | kartın ölçülecek kadar boş olduğunu **fırlatarak** doğrular |
| `cloud/` | ücretsiz bir bulut oturumunda bir nokta koşturmak; hat kodu ek almıyor |

`experiments/m0_*.py` dosyalarının geri kalanı tek tek ölçüm: rotasyonun
değeri, ölçek uydurma, hassasiyet kaldıraçları, tile zamanlamaları, kaldıraç
denetimi. Her birinin docstring'i neyi neden ölçtüğünü ve yanıldığı yerleri
taşıyor.

**Belgeler:**

- [`docs/STATUS.md`](docs/STATUS.md) — **buradan başla.** Ne doğrulandı, ne varsayıldı,
  hangi karar neden alındı, sırada ne var, hangi ortam tuzakları saatlere mal oldu
- [`docs/spec_v7.md`](docs/spec_v7.md) — şartname. Matematik, muhasebe ve protokol bağlayıcı
- [`preregistration.md`](preregistration.md) — M1'in ön-kaydı. **Henüz dondurulmadı**; §9 eksikleri listeliyor
- [`docs/audit.md`](docs/audit.md) — v6'nın M0 öncesi denetimi. Kararların hangi bilgiyle alındığının kaydı, olduğu gibi korunuyor
- [`docs/gate_a_dry_run.md`](docs/gate_a_dry_run.md) — GPU harcamadan önce literatürden yapılan Kapı A provası

---

## Ne doğrulandı, ne varsayıldı

Bu ayrımı açıkça yapmak gerekiyor.

**Doğrulanmış (test edilmiş):**

- Muhasebenin tamamı. Golden sabitler tam kesirli aritmetikle bağımsız
  türetiliyor; `accounting.py` genel dispatch ile hesaplıyor. İki yol, tek cevap
- E8P codebook kurgusu — 227+29 kaynak örüntüsü enumerasyonla, 2¹⁶ ayrık
  kodsözcüğü, kafes üyeliği, **tam 2 bit/ağırlık**
- **`vq_bits = 2.0`'ın maliyet tarafı, gerçek checkpoint'ten** — QuIP# E8P ve
  QTIP releases'lerinde kodsözcüğü yükü tam 2.000000; katman-başı yan bilgiyle
  2.005204 / 2.006740. Manifest aritmetiği ve toplam dosya boyutu birbirini tam
  tutturuyor (`experiments/m0_vq_bits.py`)
- Rotasyonun maskeyi koruduğu (her iki eksende, her `T`'de)
- Telafinin ileriye-only olduğu, ve kazancının kanal korelasyonundan geldiği
- Kalibrasyonun sıkıştırılmış modeli okuduğu — sentetik bloklarda **ve gerçek bir Llama'da**
- Adaptörün modelin kendi hesabını birebir yeniden ürettiği (elle sürülen
  bloklar → modelin logit'leri, 1e-5)
- Kapı B'nin gürültüde "interior" **demediği**
- **`Δ = Q + τ` transfer sapması** — tahmin edici kurulup gerçek hattın yanında
  koşuldu. `T=1` kimlik kontrolü tam sıfır; sapma çekiliş gürültüsünü 12.3×
  aşıyor, yani ön-kaydın toleransı seed varyansından türetilemez
- **Kapı B'nin gücü** — 5 çekiliş 2.29 σ saptıyor, ölçülen etki 6.7 σ.
  Ama komşu tile'lar (0.31 σ) ayrılmıyor, o yüzden `T*` nokta değil **küme**
  olarak raporlanıyor (`experiments/m0_gate_b_power.py`)
- **Rotasyonun değeri gerçek katmanda** — sentetik veride %3 okunuyordu, gerçek
  `o_proj`'ta **−%70**. Sentetik ölçüm iki mertebe yanılmış; sebebi rotasyonun
  işinin kalın kuyruğu yaymak olması ve sentetik verinin kuyruğunun olmaması
  (`experiments/m0_rotation_value.py`)
- **Tam model sürücüsü ve devam etme** — gerçek Llama-2-7B blokları sıkıştırıldı,
  checkpoint yazıldı, ve testi "resume çalışıyor" değil **"resume cevapta
  görünmez"**: kesilen koşu kesilmeyenin ppl'ine varıyor. Aradaki fark, blok 17'yi
  hiçbir model sürümünün üretmediği aktivasyonlara karşı kalibre etmek olurdu —
  ve o **patlamaz**, sessizce yanlış olur
- **İki hassasiyet kaldıracı gerçek blokta denetlendi** — her biri iki kez
  ölçüldü (terim tek başına, ve terim hattın içinde) ve altı karşılaştırmanın
  altısında yerinde tasarruf izolenin %95–107'si. Üçüncü kaldıraç (fp16) aynı
  denetimden **geçemedi** ve reddedildi (`experiments/m0_lever_audit.py`)

**Varsayım — doğrulanmadı:**

> E8P'nin **kompaktlanmış survivor alt-matrisinde** 2 bit kalitesini koruduğu.
> Survivor'lar tanım gereği dağılımın kalın kuyruğu; kafes quantizer Gauss'a
> yakın girdi ister. Rotasyon bunu düzeltmeli ama gösterilmedi.
>
> Ölçüm **maliyeti** kapattı, kaliteyi değil: 2 bit ödendiği kesin, o 2 bitin
> karşılığında 2 bitlik *kalite* alınıp alınmadığı açık.

Bu varsayımı sınayacak ucuz deney bilinçli olarak atlandı. Erken uyarı kuralı ve
geri dönüş yolu `preregistration.md` §9.1'de tanımlı.

**İlk gerçek ölçüm yapıldı (2026-08-21).** Llama-2-7B dense perplexity,
WikiText-2, 8 GB kartta katman-akışlı:

| seqlen | ölçtüğümüz | yayımlanan | fark |
|---|---|---|---|
| 2048 | **5.4675** | 5.47 | −0.0025 |
| 4096 | **5.1143** | 5.12 | −0.0057 |

Bu iki şeyi birden veriyor: hat yayımlanmış sayıları yeniden üretiyor, **ve**
literatürdeki 5.12/5.47 ayrımının sebebi dizi uzunluğu olarak doğrulandı — ölçüm
öncesinde kaydedilmiş bir hipotezdi.

**Sıkıştırma kalitesi artık ölçüldü — ama yalnız iki noktada**
(`docs/STATUS.md` §5.11 ve §5.14). Dense baseline ve o iki nokta dışındaki her
sayı ya katman düzeyinde `tr(E H Eᵀ)` proxy'si ya da sentetik veri. Sentetik
smoke testte hata eğrisi U şeklinde çıkıyor ve Kapı A geçiyor — ama **veriyi biz
ürettik, bu tez lehine kanıt değil**, ve gerçek modelde Kapı A, o iki nokta
karşılaştırıldığında **düşüyor**.

---

## Maliyet — ve maliyet modelinin kendisi bir sonuç

M1 ızgarası (3 bütçe × 7 tile × 5 çekiliş) bu makinede önce **120 gün**
çıkmıştı. Şu an **11.7 gün**, ve aradaki farkın çoğu daha hızlı kod değil, daha
doğru bir model:

| | ne oldu |
|---|---|
| 120 → 12 gün | Cholesky hızı k=2048'de ölçülüp her genişliğe uygulanmıştı; gerçek widthlerde **9.4× fazla** yazıyordu |
| 12 → ~40 gün | Modelin **hiç bilmediği iki terim** bulundu: kalibrasyon ve ileri telafi. Sayı bir kez **yukarı** gitti |
| ~40 → 15 gün | O iki terim düzeltildi (Hessian'lar bloğun kendi cihazında: **25×**) |
| 15 → 11.7 gün | Model, hattın **koştuğu aritmetiği** fiyatlamaya başladı |

Sonuncusu şu sınıftandı: `rotation_seconds`'ın Kronecker yolu **yoktu** ve
hattın 08-25'ten beri koşmadığı yoğun formu fiyatlıyordu. **Kötümser** olduğu
için hiçbir şey şikâyet etmemişti.

Model dokuz kez yanıldı ve **yedisi oran hatası değil, eksik terimdi.** Yani bu
modelde sorulacak soru "oran doğru mu" değil, **"listede ne yok"**.

**Ölçüm hijyeni bu projenin asıl çıktısı** ve `docs/STATUS.md` §14'te duruyor.
Tekrar tekrar vuran tuzaklar: bir rejimde ölçülüp hepsine uygulanan sabit,
cevabı sınayıp yolu sınamayan test, hiç koşulmamış bir kompozisyon, ve
değiştirdiği yoldan geçmeyen ölçüm. Kural olarak: **bir test cevabı değil yolu
izlemeli**, ve **yeni bir test eski koda karşı kırmızı olduğu gösterilmeden
kabul edilmemeli**.

---

## Eksikler

- **Bütçe ekseninin ortası.** Elde iki nokta var, arası yok: 1.5 bit → 47.64,
  2.0625 bit → 13.97. Eğri düz mü, yoksa bir dizde mi kırılıyor bilinmiyor, ve
  Kapı A'yı tek bir uç noktayla kapatmak §5.11'in `dense_wall`'la yaptığı
  hatanın aynısı olur. Sıradaki nokta **B≈1.75 (d≈0.85)** — bu makinede ~4 saat
- **Kapı B'nin hiç gerçek verisi yok.** Verdikt için ≥5 çekiliş gerekiyor, elde
  bir tane var; yedi tile boyutunun altısı hiç koşulmadı. Kapı B, Kapı A'nın
  düşmesinden **bağımsız**: çerçeve daralır, proje durmaz
- **E8P varsayımı** (yukarıda) — `docs/STATUS.md` §9 onu hâlâ projenin en büyük
  tek riski olarak taşıyor, ama beklediği kontrol artık alındı: aynı hat
  `d≈1.0`'da. Budamayı tamamen kaldırmak katman hatasını yalnız %8–32 alıyor ve
  `o_proj` %1 **kötüleşiyor** — yani quantizer'ın çöktüğü yer kompaktlanmış
  survivor alt-matrisi değil. §3.2 varsayımı yine de doğrulanmamış sayıyor,
  §5.13'ün pahalı yoldan öğrendiği sebeple: katman hatası **teşhis**, karar
  ppl'den okunur, ve orada aynı değişiklik **3.4×**
- **Ön-kaydın dondurulması.** `Δ(T)` tahmin eğrisi ve `T*_tahmin` kutuları açık;
  ikisi de `τ` süpürmesine bağlı ve süpürme betiği henüz yazılmadı
- **Ölçülen iki nokta da bir protokol kusuru taşıyor.** Kalibrasyon seqlen
  2048'de, değerlendirme 4096'da — projenin kendi "pencereyi sabitle" kuralına
  aykırı. İki koşu kusuru **birebir aynı** taşıdığı için aralarındaki
  karşılaştırma temiz; ama iki sayı da literatürün yanına bu not olmadan
  konmamalı
- **Sürücünün bağlam bedeli — ve burada duran eski madde yarı yanlıştı.**
  Aritmetik gerçek katmanda 1.03× doğrulukla fiyatlanıyor, ve kalan açık bellek
  tavanına bağlanmıştı. Ölçüldü: 8 GiB karttan 14.6 GiB'a çıkmak açıklanmamış
  2.7×'in ~1.4×'ini aldı, yani **1.95× hâlâ açıklanmamış**. Blok başına modelin
  aritmetiği 84 s, T4 sürücüsü 168 s, dizüstü sürücüsü 339 s
- **Axis A için LDLQ** — şu an `NotImplementedError`

---

## Lisans

[Apache License 2.0](LICENSE). MIT yerine bunun seçilme sebebi **açık patent
lisansı**: bu depodaki katkı bir *yöntem*, ve MIT patent konusunda sessiz — bu da
üçüncü bir taraf komşu bir şeyi patentlerse kodu kullananı korumasız bırakıyor.

Hiçbir üçüncü taraf kodu kopyalanmadı. QuIP#, QTIP, QuaRot, SparseGPT, Wanda,
GPTQ, VENOM ve OBR makalelerinden yeniden uygulandı — E8P codebook'u
enumerasyonla, muhasebe özdeşlikten — yani devralınan bir lisans yükümlülüğü
yok.

---

## Notlar

Test kodu koruduğu modüllerin **1.6 katı**: `tests/` altında **7742** satır,
yukarıdaki Yerleşim tablosunun on bir modülünde **4885**. `pytest -q` **689**
test topluyor; 682'si geçiyor, 7'si GPU olmadan atlanıyor. Bu bilinçli: kodun çoğu
doğrulanması gereken matematiksel iddia taşıyor, ve bu projede en pahalı hata
sınıfı sessizce yanlış bir sayı üretmek. Testlerin çoğu davranış değil,
**iddia** sınıyor — özdeşliğin geçerlilik alanı, codebook'un kurgusu,
rotasyonun maskeyi koruduğu, kapının gürültüde geçmediği.

`tests/golden.py` özel bir dosya — `accounting.py`'yi **import etmez**. Golden
değerleri çağıran bir test hiçbir şey kanıtlamaz; bu yüzden aynı sayılara iki
bağımsız yoldan varılıyor.

Bu çalışmanın bazı bölümleri yapay zekâ desteğiyle geliştirildi.
