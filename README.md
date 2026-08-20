# tilesparse

**Sparsity Below the Quantization Floor** — bit bütçesi PTQ tabanının altına
indiğinde `(survivor quantizer, granularity, density)` üçlüsünü birlikte
optimize etmek.

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

```bash
pip install torch transformers datasets accelerate lm-eval numpy scipy pytest
```

```bash
python -m pytest tests/ -q
```

Sentetik smoke test — hattın tamamı, iki kapı dahil:

```bash
python experiments/m1_gates.py --synthetic --n-out 64 --n-in 128 --seeds 3 --budgets 1.5
```

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
| `experiments/m1_gates.py` | M1'in iki kapısı |

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

**Ama sıkıştırma kalitesi hâlâ ölçülmedi.** Dense baseline dışındaki her sayı
ya katman düzeyinde `tr(E H Eᵀ)` proxy'si ya da sentetik veri. Sentetik smoke testte hata eğrisi
U şeklinde çıkıyor ve Kapı A geçiyor — ama **veriyi biz ürettik, bu tez lehine
kanıt değil.**

---

## Eksikler

- **Ön-kayıt toleransı.** Transfer pilotundan türetilmeli, seed varyansından
  değil — yoksa prereg "tutmadı" dalına kilitlenir
- **Kapı B'nin istatistiksel gücü.** 5 çekiliş `T=4` ile `T=16`'yı ayırmaya
  yetiyor mu, M1'den *önce* ölçülmeli
- Axis A için LDLQ

---

## Notlar

Test kodu üretim kodunun ~%85'i (2900 / 3397 satır) ve bu bilinçli: kodun çoğu
doğrulanması gereken matematiksel iddia taşıyor, ve bu projede en pahalı hata
sınıfı sessizce yanlış bir sayı üretmek. Testlerin çoğu davranış değil,
**iddia** sınıyor — özdeşliğin geçerlilik alanı, codebook'un kurgusu,
rotasyonun maskeyi koruduğu, kapının gürültüde geçmediği.

`tests/golden.py` özel bir dosya — `accounting.py`'yi **import etmez**. Golden
değerleri çağıran bir test hiçbir şey kanıtlamaz; bu yüzden aynı sayılara iki
bağımsız yoldan varılıyor.
