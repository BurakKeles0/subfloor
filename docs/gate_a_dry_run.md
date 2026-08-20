# Kapı A — Kâğıt Üstü Prova

> Plan §F madde 6. M1'e GPU harcamadan önce, her satırın literatürdeki en yakın
> komşusuna bakarak Kapı A'nın yönünü belirlemek. Maliyeti sıfır, bilgi değeri
> yüksek: sonuç beklendiği gibi çıkarsa §0.2/§0.4'ün çerçevesi **ön-kayıttan önce**
> daraltılmalı.

Tarih: 2026-08-20. Model: Llama-2-7B, WikiText-2.

---

## 0. Bulunan ilk şey: iki uyumsuz protokol

Literatürde Llama-2-7B için **iki farklı dense baseline** dolaşıyor ve sayılar
birbirine karıştırılamaz:

| Protokol | dense ppl | Kullananlar |
|---|---|---|
| **A** | **5.12** | Wanda makalesi, QTIP makalesi |
| **B** | **5.47** | QuIP# makalesi, SliceGPT |

Aynı yöntemin aynı modeldeki sayısı protokole göre değişiyor — QuIP# 2-bit, kendi
makalesinde **6.66**, QTIP'in tablosunda **6.19**. Fark 0.47 ppl, yani Kapı A'da
ölçmeye çalıştığımız etkiden büyük.

**DOĞRULANDI (2026-08-21).** Sebep dizi uzunluğuymuş. Kendi ölçümümüz,
Llama-2-7B fp16, WikiText-2 test:

| seqlen | ölçtüğümüz | yayımlanan | fark |
|---|---|---|---|
| 2048 | **5.4675** | 5.47 | −0.0025 |
| 4096 | **5.1143** | 5.12 | −0.0057 |

Yani iki aile de yanlış değil — aynı model, farklı pencere. Sonuç "birini seç"
değil, **"pencereyi sabitle"**: yayımlanmış bir sayı ancak bizim aynı `seqlen`'de
aldığımız bir sayının yanına konabilir. Bu, aşağıdaki bütün karşılaştırmaları
geçerli kılıyor — her satır tek bir aile içinde kalıyor.

Üçüncü bir protokol daha var: PALS aynı modele dense **5.33** ve %50
Wanda için **12.92** diyor — bu, Wanda'nın kendi 6.42'siyle uzlaşmıyor ve
kullanılmadı.

> **Protokol §6'ya eklenecek kural:** her karşılaştırma tek bir protokol ailesi
> içinde kalmalı ve tabloya dense baseline değeri yazılmalı. Protokoller arası
> alıntı yapılmayacak. Bu, spec'in "Llama-3.x yayımlanmış sayılarla kıyaslanmaz"
> kuralının Llama-2 için de geçerli olduğu anlamına geliyor.

---

## 1. Referans sayılar

### Protokol A (dense = 5.12)

| Kaynak | Konfig | ppl |
|---|---|---|
| Wanda Tablo 3 | dense | 5.12 |
| Wanda Tablo 3 | Wanda %50 unstructured | **6.42** |
| Wanda Tablo 3 | SparseGPT %50 unstructured | 6.51 |
| Wanda Tablo 3 | Magnitude %50 | 14.89 |
| Wanda Tablo 3 | Wanda 4:8 | 7.97 |
| Wanda Tablo 3 | SparseGPT 4:8 | 8.12 |
| Wanda Tablo 3 | SparseGPT 2:4 | 10.17 |
| Wanda Tablo 3 | Wanda 2:4 | 11.02 |
| QTIP Tablo 5 | QTIP 4-bit | 5.17 |
| QTIP Tablo 5 | QuIP# 4-bit | 5.19 |
| QTIP Tablo 5 | QTIP 3-bit | **5.28** |
| QTIP Tablo 5 | QuIP# 3-bit | 5.41 |
| QTIP Tablo 5 | QTIP 2-bit | **5.86** |
| QTIP Tablo 5 | QuIP# 2-bit | 6.19 |

13B (Wanda Tablo 3): dense 4.57 · Wanda %50 5.56 · SparseGPT %50 5.63 ·
Wanda 4:8 6.55 · Wanda 2:4 8.27. M5'in 13B doğrulama noktası için.

### Protokol B (dense = 5.47)

| Kaynak | Konfig | ppl |
|---|---|---|
| QuIP# Tablo 2 | dense | 5.47 |
| QuIP# Tablo 2 | QuIP# 4-bit | 5.56 |
| QuIP# Tablo 2 | QuIP# 3-bit | **5.79** |
| QuIP# Tablo 2 | QuIP# 2-bit | 6.66 |
| SliceGPT Tablo 1 | SliceGPT %10 slicing | 5.89 |
| SliceGPT Tablo 1 | SliceGPT %20 slicing | 6.64 |
| SliceGPT Tablo 1 | SliceGPT %25 slicing | **7.24** |
| SliceGPT Tablo 1 | SliceGPT %30 slicing | 8.12 |
| SliceGPT Tablo 1 | SparseGPT 2:4 | 8.69 |

### Doğrulanmamış / eksik

- SparseGPT %60 ≈ 9.58, Wanda %60 ≈ 9.71 — yalnızca arama özetinden, birincil
  kaynak okunmadı. **M0'da doğrula.**
- **Dense 3-bit GPTQ, Llama-2-7B** — çapa 1'in spec'te yazılı baseline'ı. Bulunan
  GPTQ sayıları LLaMA-**1**-7B'ye ait (4-bit g128 5.85, 3-bit g128 6.61, dense 5.68).
  Llama-2 için ölçülmeli.
- **Unstructured %20–40 bandı, Llama-2-7B.** Tek bir karşılaştırılabilir tabloda yok.
  Aşağıda görüleceği gibi Kapı A'nın kaderi tam olarak buraya bağlı.

---

## 2. Çıkarım kuralı (a fortiori)

Bir tile/structured konfigürasyonu, **aynı yoğunlukta unstructured seyreklikten daha
iyi olamaz** (maske özgürlüğü alt kümesi), ve quantization yalnızca hata ekler.
Dolayısıyla:

> aynı `d`'de yayımlanmış **unstructured, FP16** ppl'i, bizim konfigürasyonumuz için
> bir **alt sınırdır** (iyimser tarafta).

Alt sınır bile rakibe kaybediyorsa, o satır için Kapı A ölçüm yapılmadan düşer.

---

## 3. Çapa 2 = 2.140625 — Kapı A **düşüyor**, her satırda

Rakip: **QTIP 2-bit = 5.86** (protokol A).

| # | Konfig | `d` | seyreklik | iyimser alt sınır | vs 5.86 |
|---|---|---|---|---|---|
| 2 | 4-bit + unstructured | 0.274 | %72.6 | ≫ 6.42 (%60 zaten ~9.6) | ✗ |
| 3 | 4-bit + tile-4 | 0.455 | %54.5 | > 6.42 | ✗ |
| 4 | 4-bit + tile-16 | 0.500 | %50.0 | **6.42** (Wanda %50) | ✗ |
| 5 | 4-bit + T=max | 0.515 | %48.5 | ≫ 7.24 (SliceGPT %25 bile 7.24) | ✗ |
| 7 | AQLM-survivor + unstr | 0.522 | %47.8 | ~6.42 | ✗ |
| 8 | AQLM-survivor + tile-16 | 0.951 | %4.9 | ~AQLM 2-bit ≈ 6.2–6.6 | ✗ |

En iyi satır (#4) alt sınırı 6.42; rakip 5.86. **Fark 0.56 ppl ve alt sınır iyimser.**
Ölçüm bunu tersine çeviremez.

Satır #8 kendi başına bir bulgu olarak duruyor: `d = 0.951`'de seyrekliğin kaldıracı
yok, konfigürasyon fiilen dense AQLM 2-bit.

---

## 4. Çapa 1 = 3.1484375 — iki satır düşüyor, iki satır **belirsiz**

Rakip: **QTIP 3-bit = 5.28** / QuIP# 3-bit 5.41 (protokol A), ya da protokol B'de
QuIP# 3-bit 5.79.

> Not: spec çapayı "dense 3-bit **GPTQ**"ya bağlıyor. GPTQ 3-bit, 3-bit sınıfının en
> zayıf üyesi. §6 "yetkin baseline" şart koştuğuna göre gerçek rakip QTIP/QuIP#.
> Çapa GPTQ'da kalırsa Kapı A kolay geçer ama sonuç savunulamaz.

| # | Konfig | `d` | seyreklik | iyimser alt sınır | vs 5.28 |
|---|---|---|---|---|---|
| 2 | 4-bit + unstructured | 0.517 | %48.3 | **6.42** | ✗ |
| 3 | 4-bit + tile-4 | 0.697 | %30.3 | **yayımlanmış komşu yok** | ? |
| 4 | 4-bit + tile-16 | 0.742 | %25.8 | **yayımlanmış komşu yok** | ? |
| 5 | 4-bit + T=max | 0.758 | %24.2 | 7.24 (SliceGPT %25, protokol-eşleşmiş) | ✗ |

Satır #5, protokol içinde temiz bir karşılaştırma: SliceGPT %25 slicing = **7.24** ile
QuIP# 3-bit = **5.79**, ikisi de protokol B, ikisi de `d ≈ 0.75`. Structured kenar
açık farkla kaybediyor.

Satır #3–4 gerçekten belirsiz: %26–30 seyreklik çok hafif bir rejim, ve tile-16
unstructured ile structured arasında bir yerde. Bu iki hücre M1'in ölçmesi gereken
tek şey.

---

## 5. Verdikt

**Kapı A, literatürün konuşabildiği her yerde düşüyor.** Ayakta kalan tek hücre, çapa
1'de tile-4/tile-16.

Ve burada rahatsız edici bir simetri var:

> Kapı A'nın hâlâ canlı olduğu tek hücre, tezin motivasyonunun geçerli olmadığı
> bütçede. Çapa 1 = 3.15 bit, PTQ tabanının (~2.0–2.2) **üstünde**. Motivasyonun
> tuttuğu yerde (çapa 2) Kapı A her satırda düşüyor.

Karar tablosunun `✗/✓` dalı gerçekleşiyor: **proje durmuyor, çerçeve daralıyor.**

### Bunun anlamı

1. **§0.2 / §0.4 şimdi daraltılmalı.** Birincil soru "seyreklik ailesi *içinde*
   granülerlik nasıl seçilir" olmalı; dense-lowbit karşılaştırması ikincil bağlam.
   Ön-kayıttan sonra daraltmak pahalıya patlar.

2. **§C (RHT-on-compacted-survivors) artık opsiyonel değil.** Prova şunu gösteriyor:
   `sparse + GPTQ-4bit`, `dense + VQ-2bit`'e her yerde kaybediyor. Sayıları
   oynatabilecek tek şey, survivor'ların da VQ sınıfı bir quantizer alması. O da
   kompaktlanmış survivor matrisine RHT uygulamayı gerektiriyor, o da büyük `T`
   gerektiriyor. **Kapı A'nın geçebileceği tek yol bu** — ve granülerlik tezini
   destekleyen bir mekanizma, çünkü `T`'yi yukarı itiyor.

3. **Kapı B etkilenmiyor.** Granülerlik sorusu, dense-lowbit'e kaybedilse bile
   anlamlı; iki kapıyı ayırmanın amacı buydu.

4. **M1'in ölçüm önceliği değişiyor.** Çapa 1'de `T ∈ {4, 16}` tek belirsiz hücre
   olduğuna göre, `T=4` ile `T=16` arasındaki ayrımın istatistiksel gücü (plan §B5)
   kritik hale geliyor.

---

## 6. M0'a düşen doğrulama işleri

- [ ] Protokol farkının kaynağını sabitle (seqlen 2048 vs 4096) ve §6'ya "protokoller
      arası alıntı yok" kuralını ekle
- [ ] Dense 3-bit GPTQ, Llama-2-7B ölç — çapa 1'in yazılı baseline'ı elde yok
- [ ] Çapa kararı: GPTQ mı, QTIP/QuIP# mı? (Öneri: yetkin baseline, yani QTIP)
- [ ] Unstructured %25 / %30 / %40 ölç — çapa 1'in belirsiz hücresinin alt sınırı
- [ ] SparseGPT/Wanda %60 sayılarını birincil kaynaktan doğrula
- [ ] PALS'ın 5.33 / 12.92 tutarsızlığını çöz ya da o kaynağı devre dışı bırak

## Kaynaklar

- Wanda — arXiv:2306.11695
- QTIP — arXiv:2406.11235
- QuIP# — arXiv:2402.04396
- SliceGPT — arXiv:2401.15024
- AQLM — arXiv:2401.06118
- PALS — arXiv:2607.07557 (kullanılmadı, protokol uzlaşmıyor)
