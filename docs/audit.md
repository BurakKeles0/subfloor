<!-- STATUS HEADER — prepended when the audit was moved into the repo. -->

> ## Bu belge hakkında
>
> **Spec v6'nın M0 öncesi denetimi, 2026-08-20.** Kod yazılmadan önce yazıldı ve
> sonrasında uygulamadan çıkan bulgularla genişletildi (§G, §H, §I).
>
> **Olduğu gibi korunuyor.** Bazı bölümler artık geçmiş zamandan bahsediyor —
> §C "planın kaçırdığı en büyük fırsat" diyor, oysa o fırsat artık `rotation.py`
> ve LDLQ olarak uygulandı. Bu bir kusur değil: belgenin değeri, kararların
> **hangi bilgiyle** alındığının kaydı olması. Sonradan düzeltmek o kaydı yok eder.
>
> Neyin nereye vardığı:
>
> | Bölüm | Durum |
> |---|---|
> | §A aritmetik hatalar | Kodda düzeltildi. `tests/golden.py:SPEC_V6_ERRATA` ve `test_spec_v6_errata_are_not_reproduced` nöbet tutuyor. **Spec belgesinin kendisi hâlâ düzeltilmedi.** |
> | §B1 indeks modeli | `quantize`/`accounting` içinde alternatif kodlama mevcut; varsayılan v6'nın dondurduğu haliyle bırakıldı |
> | §B2 Kapı A provası | Yapıldı → `docs/gate_a_dry_run.md`. Sonuç: Kapı A literatürün konuşabildiği her yerde düşüyor |
> | §B3 tolerans | **Açık.** `preregistration.md` §9, transfer pilotu bekliyor |
> | §B4 eşleştirilmiş seed | `preregistration.md` §5.2 |
> | §B5 Kapı B formu | `experiments/m1_gates.gate_b` + `preregistration.md` §7. §I1 bunu ampirik olarak doğruladı |
> | §B6 `wb × T` | Birincil öngörüden çıkarıldı → `preregistration.md` §5.3 |
> | §C RHT | Uygulandı (`rotation.py`, `quantize.ldlq_quantize`). **Sonucu §I3'te düzeltildi** — mekanizma burada anlatılandan farklı çıktı |
> | §D1 `is_live` ayrımı | `accounting.live_diagnostics` |
> | §D2 eksen saliency'si | `scoring.py`'de yapısal olarak çözüldü — eksenler metriği paylaşmak zorunda |
> | §D3 ablasyon kurgusu | **Açık.** Metot kararı |
> | §E küçük notlar | E3 (tensor-core hizalaması) `align=8` olarak geldi; gerisi açık |
> | §F yapılacaklar | Büyük ölçüde tamamlandı; `preregistration.md` §9 güncel liste |
> | §G survivor quantizer | E8P kararına götürdü → `quantize.py` |
> | §H uygulama planı | Uygulandı; bant `B ∈ {1.75, 1.60, 1.50}` |
> | §I uygulama bulguları | I2 ve I3 LDLQ ile çözüldü; I1 ön-kayda girdi |
>
> Hâlâ açık olanlar: **§B3** (tolerans), **§D3** (ablasyon kurgusu), ve
> spec belgesinin §A'ya göre düzeltilmesi.

---

# Spec v6 Değerlendirmesi — "Sparsity Below the Quantization Floor"

## Context

Kullanıcı, henüz kodu yazılmamış bir araştırma projesinin uygulama şartnamesini (v6)
değerlendirilmek üzere verdi. `tilesparse/` dizini mevcut değil — bu bir kod incelemesi
değil, **M0'a başlanmadan önce yapılan bir spec denetimi**. Amaç: matematiğin doğru olup
olmadığını, deney tasarımının kendi sorusunu gerçekten yanıtlayıp yanıtlamadığını ve
GPU saati harcanmadan önce düzeltilmesi gereken şeyleri saptamak.

Aşağıdaki tüm sayısal iddialar bağımsız olarak yeniden hesaplandı.

---

## Genel hüküm

**Plan sağlam ve alışılmadık derecede disiplinli.** Çekirdek matematik doğru: `1 − 1/T`
özdeşliği, `B*` duvarı, `is_live` filtresi ve ön-kaydın döngüsellikten arındırılması —
hepsi doğrulandı. v5→v6 düzeltmelerinin çoğu gerçek hataları kapatıyor.

Ama **başlamadan önce düzeltilmesi gereken 4 aritmetik hata, 3 iç tutarsızlık ve
1 çerçeveleme riski var.** Ayrıca planın kaçırdığı, tezi belirgin biçimde güçlendirebilecek
bir mekanizma mevcut (§C).

---

## A. Doğrulanmış aritmetik hatalar — M0'dan ÖNCE düzelt

Bu değerler §3.4'te `1e-12` toleransla assert edilecek. Yanlış golden değerlerle
yazılan test, doğru kodu reddeder.

### A1. §5.2 Çapa 1 tablosunda iki hücre yanlış

| Konfig | Spec | **Doğru** |
|---|---|---|
| 4-bit + tile-4 | 0.696992 | **0.697368** (= 371/532) |
| 4-bit + `T=max` | 0.757954 | **0.757519** (= 403/532) |

Doğrulama, özdeşliğin kendisiyle: `d(4) = 0.516917 + 0.75/4.15625 = 0.697368`.
Çapa 2 tablosunun **tamamı doğru** (0.274436 / 0.454887 / 0.500000 / 0.515038) — hata
yalnızca çapa 1'de.

### A2. `density_for_budget('unstructured', 0.60, 4, 11008)`

Spec: `0.0341271` → **Doğru: `0.0341248`**
(`0.60 / (4.15625 + 13.4262648) = 0.0341248`)

### A3. `log2(11008)` hassasiyeti zincirleme yayılıyor

Spec `13.4262102` kullanıyor; gerçek değer **`13.4262648`**. Bundan türeyenler:

| Sabit | Spec | Doğru |
|---|---|---|
| `b_star(4, 11008)` | 1.3095620 | **1.3095612** |
| `B=1.20` avantajı | 0.205428 | **0.205435** |
| `B=1.00` avantajı | 0.168684 | **0.168689** |

`b_star(4, 4096) = 1.3463542` ✓ doğru. `d*(11008) = 0.0744812` ✓ doğru.

> **Aksiyon:** golden sabitleri elle yazmayı bırak. `tests/golden.py` içinde `log2()`'den
> türet, tabloları o modülden üret. Aksi halde bu sınıf hata tekrar eder.

### A4. §0.1'in eşiği kendi aritmetiğinden çıkmıyor

Spec 4k bağlam varsayıp "gerçek eşik ~2.2–2.3 bit" diyor. Kendi kalemleriyle:

| Kapasite | 4k | 8k | 16k |
|---|---|---|---|
| 24.0 GiB | 2.645 | 2.488 | 2.174 |
| 23.0 GiB | 2.519 | 2.362 | 2.048 |
| 22.5 GiB | **2.456** | **2.299** | 1.986 |

**4k'da eşik 2.46–2.65 bit'tir, 2.2–2.3 değil.** 2.2–2.3 ancak 8k+ bağlamda çıkıyor.

Bu kozmetik değil — motivasyonun tamamı buna dayanıyor. 4k'da eşik 2.46 ise, PTQ tabanı
(~2.0–2.2) **rahatça yetiyor** ve "tabanın altına inmek zorundayız" gerekçesi çöküyor.
**Düzeltme kolay ve tezin lehine:** eşiği bağlam uzunluğunun fonksiyonu olarak sun ve
motivasyonu 8k–16k'ya taşı. 16k'da eşik 1.99–2.17 → tam olarak istenen rejim.

---

## B. Yapısal sorunlar

### B1. ⚠️ En önemli: `practical` indeks modeli unstructured'a haksız derecede pahalı

Spec, rastgele erişimli en ucuz kodlamanın `min(1, d·log₂(n_idx))` olduğunu varsayıyor ve
§3.2'de "`H(d)` erişilebilir değildir" paragrafını makaleye koymayı planlıyor.

**Bu doğru değil.** Blok-yerel sabit-sayılı kodlama (yani N:M'in genelleştirilmiş hali)
hem rastgele erişimlidir hem de `d·log₂(n_idx)`'ten çok daha ucuzdur: blok boyu `M` için
maliyet `d·log₂(M)`, ve `M ≈ 1/d` seçilirse `≈ d·log₂(1/d) ≈ H(d)`. Blok düzeyinde O(1)
erişim korunur — çözmek için önceki blokları okumaya gerek yok.

Somut: `d = 0.25`'te bitmap 1.0 bit, ama 1:4 kodlaması 0.5 bit, 2:8 kodlaması
`log₂C(8,2)/8 = 0.601` bit. **Manşet avantajın tamamı "unstructured indeksi 1.0 bit"
varsayımına yaslanıyor.** Gerçekçi unstructured indeksi 0.75 olsa, tile-16 avantajı
`(1−1/16)/W = 0.2256` yerine `(0.75−1/16)/W = 0.1654` — **%27 daha küçük.**

**İyi haber, kapsamı dar:** `d·log₂(M) < 1` ancak `d < 1/log₂(M)` iken; birincil bantta
(`d = 0.27–0.76`) bitmap zaten optimal. Yani **M1/M2'nin birincil bant sonuçları güvende.**
Etkilenenler:
- §0.3.1'in `B*` duvarı (blok-yerel kodlamayla `B*` kayar)
- §3.2'nin "H(d) erişilemez" paragrafı — **bu haliyle yazılırsa hakem yakalar**
- §0.4'ün "N:M ayrık kafes, tile sürekli" delta iddiası zayıflıyor

**Ayrıca bir iç tutarsızlık:** `nm` satırında `log2(C(M,N))/M` "info_theoretic" (erişilemez)
sütununda. Ama blok-yerel kombinatoryal kodlama **pratiktir** — blok başına O(1) çözülür.
Bu sütun ataması yanlış.

> **Düzeltme (ucuz):** §3.2'nin paragrafını yeniden yaz. Doğru iddia şudur: *"rastgele
> erişim entropi sınırını yasaklamaz; maske üzerinde blok yapısı dayatarak yaklaşılabilir —
> ama o yapı kendi kalite bedelini doğurur. Yani indeks ucuzluğu ile maske özgürlüğü
> arasındaki takas kaçınılmazdır ve `T` bu takasın tek ekseni değildir."* Bu, tezi
> zayıflatmıyor — **genelleştiriyor** ve N:M/V:N:M'i rakip değil aynı ailenin üyesi yapıyor.
> §0.4'ün delta cümlesi de bu haliyle daha savunulabilir.

### B2. Kapı A muhtemelen her iki çapada da düşecek — GPU harcamadan önce kâğıt üstünde test et

Yayımlanmış sayılara dair hatırladıklarım (**doğrulanması gerekiyor, bellekten**):
Llama-2-7B WikiText-2, dense FP16 ≈ 5.47; SparseGPT/Wanda %50 unstructured ≈ 6.9–7.0;
%70 ≈ 10+; AQLM 2-bit ≈ 6.6; QuIP# 2-bit ≈ 6.2; QTIP 2-bit ≈ 5.9; GPTQ 3-bit g128 ≈ 6.3.

Bunlar doğruysa:
- **Çapa 2:** 4-bit + tile-16 → %50 *tile* seyreklik (unstructured'dan kötü) ≈ 7.3+.
  Rakip QTIP 2-bit ≈ 5.9. Kapı A **açık ara** düşer.
- **Çapa 1:** 4-bit + unstructured `d=0.517` ≈ 7.0. Rakip GPTQ 3-bit ≈ 6.3. Yine düşer.

Bu, Kuzmin et al. (NeurIPS 2023)'ün "quantization pruning'i yener" sonucuyla da tutarlı —
ve spec o makaleyi zaten referans veriyor.

**Bu projeyi öldürmez** (karar tablosu `✗/✓` dalını doğru şekilde tanımlamış). Ama §0.2 ve
§0.4'ün mevcut çerçevesi *dağıtım odaklı* ve o çerçeve düşme riski yüksek.

> **Aksiyon — M0'a 1 günlük kalem işi ekle:** M1'in 14 konfigürasyonunun her biri için
> literatürden en yakın komşu ppl'i tabloya dök. Maliyeti sıfır, bilgi değeri devasa.
> Sonuç beklendiği gibi çıkarsa çerçeveyi **şimdi** daralt: birincil soru "seyreklik ailesi
> *içinde* granülerlik nasıl seçilir" olsun, dense-lowbit karşılaştırması ikincil bağlam
> olarak kalsın. Sonradan daraltmak, ön-kayıttan sonra daraltmak demektir — ve bu maliyetlidir.

### B3. Ön-kayıt tolerans kuralı, kapının düşmesini garantiliyor

§5.2 toleransı "M0'ın seed varyansından türet" diyor. Ama `Δ = Q + τ` tahmininin hatası
**seed gürültüsü değil, ayrılabilirlik varsayımının yanlılığıdır**: `τ` eş-yoğunlukta ve
quantization'sız ölçülüyor, sonra bütçe-eşleşmiş + quantize edilmiş ayara taşınıyor.
Seed varyansından türetilen tolerans neredeyse kesin aşılır → prereg "tahmin tutmadı"
dalına kilitlenir ve `T*` yorumlanamaz hale gelir. v5'in döngüselliği gitti ama yerine
**ters yönde bir yapısal kusur** geldi.

> **Düzeltme:** M0'a 4–5 koşuluk bir *transfer pilotu* ekle — `τ`'yu tek bir `(T,d)`
> noktasında hem quantization'sız hem 4-bit ile ölç. Toleransı bu iki ölçümün farkından
> türet. ~2 GPU-saat, ön-kaydı kurtarır.

### B4. `τ` tek seed'in gürültü-sadeleşmesi ancak eşleştirilmiş çekilişte geçerli

"τ bir *fark* ölçüyor, ortak kalibrasyon gürültüsü sadeleşiyor" doğru — **ama yalnızca
`ppl(T,d)` ve `ppl(1,d)` aynı kalibrasyon çekilişinden geliyorsa.** Spec `Q` için 3 seed,
`τ` için 1 seed diyor ama eşleştirmeyi şart koşmuyor.

> **Düzeltme:** ön-kayda tek satır — *"`τ`'nun seed'i, `Q`'nun üç seed'inden biriyle
> aynı çekiliştir; `τ` o çekiliş üzerinde eşleştirilmiş fark olarak hesaplanır."*

### B5. Kapı B'nin argmin formu istatistiksel olarak kırılgan

`T ∈ {1,4,16,max}` üzerinde 3 seed ile ham `argmin` almak seçim yanlılığı taşır; gürültülü
4 noktada argmin, gerçek `T*` uçta olsa bile içeride çıkabilir — yani **Kapı B yanlış
pozitif verebilir.**

> **Düzeltme:** Kapı B'yi argmin yerine şu şekilde tanımla: *"`Δ(T*)`, hem `Δ(1)`'den hem
> `Δ(max)`'tan bootstrap CI'ları örtüşmeyecek şekilde düşük mü?"* Ve ön-kayda M0'ın seed
> varyansından türetilmiş **minimum saptanabilir fark**ı yaz. 3 seed bu ayrımı yapmaya
> yetmiyorsa şimdi öğren, M1'den sonra değil.

### B6. §0.3.2'nin `wb × T` öngörüsü muhtemelen saptanamaz

Türetme doğru düzeltilmiş (birinci koşul gerçekten `T*`'ın *büyümesini* söylüyor, işaret
belirsizliği de doğru gerekçelendirilmiş). Ama `is_live` (§3.5) ile birleştirince:
birincil bantta `wb=2` ölü, `wb=3` çapa 1'de ölü. Geriye `W ∈ {3.148, 4.156}` kalıyor —
**yalnızca 1.32× aralık.** `T` ızgarası log aralıklı olduğuna göre `T*`'ın kayması için
etkinin en az 2× olması gerekir.

> **Düzeltme:** `wb × T`'yi ön-kayıtta *birincil öngörü* olmaktan çıkar; `Δ(T)` eğrilerini
> `wb` başına raporla, `argmin` kaymasını iddia etme.

---

## C. Planın kaçırdığı en büyük fırsat: RHT'nin maliyeti `T`'ye bağlı

§0.5 "seyreklik ile incoherence processing bir arada olamaz" diyor ve QuIP#/QTIP'i tamamen
dışlıyor. §7.19 da "tile-yerel ortogonal rotasyon `ε_S`'i değiştirmez" diyor — **bu doğru**
(Axis B'de `Σ_{i∈R_t} w_i,S M w_i,Sᵀ = tr(W_S M W_Sᵀ)`, satırların sol ortogonal
dönüşümü altında değişmez).

Ama bir üçüncü seçenek atlanmış: **maske sabitlendikten sonra, kompaktlanmış survivor
matrisine RHT uygulanabilir.** Maske zaten dondurulmuştur; rotasyon onu bozamaz. Alt-Hessian
`H_{S_t S_t}` tam olarak hesaplanabilir.

Bedeli tam olarak `T`'ye bağlı: her tile kendi sütun kümesini kullandığından tile başına
ayrı bir dönüşüm gerekir. Aktivasyon tarafı maliyeti / GEMV maliyeti oranı:

```
(n/T) · (d·n) · log₂(d·n)  /  (d · n²)  =  log₂(d·n) / T
```

`d·n ≈ 3000`, `log₂ ≈ 11.5`:

| T | RHT ek yükü (GEMV'e oran) |
|---|---|
| 1 | 11.5× — imkânsız |
| 16 | 0.72× — pahalı ama mümkün |
| 64 | 0.18× — makul |
| max | ~0 — bedava |

**Bu, `T`'yi yukarı iten, `Δ = Q + τ` modelinde bulunmayan ikinci bir kuvvettir.**
Ve tezin en güçlü versiyonunu veriyor:

> *Tile granülerliği yalnızca indeksi ucuzlatmaz — seyrek bir matriste incoherence
> processing'i ilk kez karşılanabilir kılan şeydir.*

Bu, Guo et al. ve VENOM'dan net biçimde ayrışır ve Kapı A'nın düşme olasılığını
ciddi biçimde azaltır: "tile-16 + RHT + VQ-on-survivors", "tile-16 + GPTQ-4bit"ten
kategorik olarak daha güçlü bir şemadır.

> **Maliyeti düşük:** M0'da bir muhasebe formülü (rotasyon `q_over`'a girmiyor, sadece
> latency modeline), M2'de bir konfigürasyon. Kernel gerekmez — offline uygulanır.
> **Yüksek getiri, düşük risk. Planın en değerli tek eklentisi bu.**

---

## D. İç tutarsızlıklar

### D1. `is_live`, spec'in kendi "en güçlü adayı"nı eliyor

AQLM-survivor (`W = 2.186`), çapa 2'de: `d(T=max) = 2.140625/2.186 = 0.9792 > 0.9`
→ **`is_live` başarısız.** Yani §5.2'nin "Kapı A'nın en güçlü adayı" dediği #7–8 satırları,
§3.5'in filtresi tutarlı uygulanırsa dışarıda kalır. Spec bunu satır #8 için sezgisel
olarak fark etmiş ("seyrekliğin kaldıracı kalmıyor") ama filtreyle bağlamamış.

> **Çözüm:** `is_live`'ı iki amaç için ayır. #7 (`d=0.522`) meşru bir **Kapı A** girdisidir
> ve kalmalı. #8 (`d=0.951`) meşru bir **Kapı B / granülerlik** girdisi değildir.
> Filtreyi "granülerlik hücresi mi" testi olarak tanımla, "raporlanabilir mi" testi olarak değil.

### D2. Eksen karşılaştırmasında saliency sabit tutulamıyor

§4.3 "eksen karşılaştırmasında saliency sabit" diyor. Ama aynı bölüm Axis A için **tam
grup-OBS** (`½ w_S [(H⁻¹)_SS]⁻¹ w_Sᵀ`), Axis B için **köşegen yaklaşım**
(`Σ w²/[H⁻¹]_jj`) tanımlıyor. Bu zorunlu bir asimetri — Axis B'de `|S| = d·n` olduğundan
tam form hesaplanamaz. Ama o zaman A-vs-B karşılaştırması **granülerlik eksenini değil,
saliency doğruluğunu ölçer.**

> **Çözüm:** eksen karşılaştırmasını **her iki eksende de köşegen formla** yap.
> Tam grup-OBS Axis A'yı ayrı bir ablasyon olarak raporla.

### D3. §3.6'nın 1. ablasyonu bütçe-eşleşmiş değil

`g=128 orijinal pozisyon` konvansiyonunda, `d=0.2744`'te grup başına ~35 survivor kalır →
survivor başına `q_over = 20/35.1 = 0.570` bit (0.156 değil) → `W = 4.570` → aynı bütçede
`d = 0.2496`. Yani iki konvansiyon **farklı bit maliyetine** sahip; saf kalite ablasyonu değil.

> **Çözüm:** ablasyonu iki koşulda çalıştır — (a) aynı `d`, farklı bütçe; (b) aynı bütçe,
> farklı `d`. Tek koşullu haliyle sonuç yorumlanamaz.

---

## E. Küçük ama gerçek notlar

1. **Notasyon çakışması (§4.2):** `d` hem yoğunluk hem boyut olarak kullanılıyor
   (`Axis A T=d`). Bir kod ajanına verilecek şartnamede bu hata üretir → `d_out` yaz.
2. **Attention koordinasyonu formülsüz.** FFN için `ε_FFN(k)` verilmiş ama `v_proj` çıkışı ↔
   `o_proj` girişi eşleşmesi, GQA altında bir KV head boyutunu budamanın gruptaki *tüm* query
   head'lerini etkilemesi, ve q/k çıkış boyutlarının **RoPE çiftleri halinde** budanma zorunluluğu
   yalnızca tabloda ima ediliyor. `T=max` için bunlar sert kısıt — formülleştirilmeli.
3. **Tensor-core hizalaması muhasebede yok.** Tile başına survivor sayısı 8/16 katına
   yuvarlanmalı; bu `d`'yi yukarı yuvarlar ve küçük `T`'de göreli maliyeti büyür — yani
   `T`'yi yukarı iten üçüncü bir kuvvet. En azından bir satır not.
4. **Wanda'nın kareli toplaması gerekçesiz.** Orijinal Wanda `|w|·‖X‖` (karesiz). Grup
   toplamında kare almak sıralamayı değiştirir. Bilinçli bir seçim (OBS'in kuadratiğiyle
   uyumlu) ama yazılmalı, tercihen L1 toplamaya karşı ablasyonlanmalı.
5. **Protokol §6 C4 ppl'i de istiyor** — M0 maliyet tahmini (15–17 GPU-saat) yalnızca
   WikiText-2 varsayıyor. C4 eval'i ekleyince ~25 GPU-saat.
6. **Takvim iyimser.** İki eksen + `T=max` ayrı kod yolu + Sinkhorn dengeli kümeleme +
   tam OBS telafisi + 7 baseline + AQLM survivor **yeniden kalibrasyonu**. Sonuncusu tek
   başına haftalar sürebilir (AQLM codebook eğitimi katman başına saatler). Toplam tahmin
   8–10 hafta; gerçekçi olarak 2–3×. **AQLM-survivor en büyük takvim riski** — M2'de
   opsiyonel işaretle, kritik yola koyma.

---

## F. M0'a başlamadan önce yapılacaklar

**Zorunlu (yarım gün):**
1. A1–A3'ün aritmetik düzeltmeleri; golden sabitleri elle yazmak yerine `log2()`'den türet.
2. §0.1'i bağlam uzunluğunun fonksiyonu olarak yeniden yaz; motivasyonu 8k–16k'ya taşı (A4).
3. §3.2'nin "H(d) erişilemez" paragrafını B1'e göre yeniden yaz; `nm`'in
   `log2(C(M,N))/M` girdisini `practical` sütununa taşı.
4. D1 (`is_live`'ı iki amaca ayır), D2 (eksen karşılaştırmasında köşegen saliency),
   D3 (ablasyonu iki koşullu yap) düzeltmeleri.
5. Ön-kayda B4 (eşleştirilmiş seed) ve B5 (argmin yerine CI-tabanlı Kapı B + minimum
   saptanabilir fark) satırlarını ekle; B6'yı birincil öngörüden çıkar.

**Yüksek getirili (1–2 gün):**
6. **Kâğıt üstü Kapı A provası** (B2): M1'in 14 konfigürasyonu için literatürden en yakın
   komşu ppl tablosu. Sonuca göre §0.2/§0.4 çerçevesini **şimdi** ayarla.
7. **Transfer pilotu** (B3): `τ`'yu bir noktada quantization'lı/suz ölç, ön-kayıt
   toleransını oradan türet. ~2 GPU-saat.

**Stratejik (karar gerektirir):**
8. **RHT-on-compacted-survivors** kolunu (§C) kapsama al: M0'da latency modeli formülü,
   M2'de bir konfigürasyon. Planın en yüksek getirili tek eklentisi.

---

## Doğrulama

Bu bir spec denetimi olduğundan "çalıştırılacak kod" yok. Denetimin kendisi şöyle doğrulanır:

- **A1–A3:** `python -c` ile yeniden hesaplandı; sonuçlar yukarıda. Düzeltilmiş değerler
  §3.4'ün assert'lerine girdiğinde `pytest` yeşil olmalı.
- **A4:** `(kapasite − sabit_yük)/7.9686` üç kapasite × üç bağlam için hesaplandı.
- **§C'nin maliyet oranı:** `log₂(d·n)/T` — `d·n ≈ 3000` için tabloda.
- **B2:** M0'ın literatür tablosu tamamlandığında Kapı A'nın beklenen yönü doğrulanır.
- **D1:** `is_live(AQLM-survivor, çapa 2, T=max)` → `d = 0.979243 > 0.9` → `False`.
  Bu, `accounting.py` yazıldığında birebir test edilebilir.

---

# §G. Survivor quantizer'ı: neden VQ kullanmıyoruz ve neden kullanmalıyız

*(Kapı A provası sonrası eklendi. Provanın bulgusu: `sparse + GPTQ-4bit`,
`dense + VQ-2bit`'e her yerde kaybediyor.)*

## G1. Şu anki durumun sebebi

Spec aslında **bir** VQ kolu planlıyor: AQLM-survivor (M1 satır #7–8, M2 grid).
Ama bu VQ ailesinin en zayıf ve en pahalı üyesi:

- QTIP 2-bit = **5.86**, AQLM 2-bit ≈ **6.2–6.6** — AQLM baştan geride.
- `vq_bits = 2.186` (codebook amortizasyonu 0.186), E8P'nin **2.000**'ine karşı.
- Spec'in kendi uyarısı (§0.5): survivor'lar kompaktlandığında 8-boyutlu vektörler
  artık bitişik kanallardan gelmiyor, AQLM'in Hessian kalibrasyonu bozuluyor.
- Codebook yeniden kalibrasyonu katman başına saatler — takvimin en büyük riski.

Güçlü VQ yöntemleri (QuIP#, QTIP) ise §0.5'te **toptan** eleniyor: "seyrek matris
global rotasyondan sonra yoğunlaşır."

## G2. O eleme fazla geniş

İddia *maskeden önce uygulanan global* rotasyon için doğru. Ama maske
dondurulduktan sonra rotasyon maskeyi bozamaz — seçim çoktan yapılmıştır.
Maskeyi koruyan **iki** rotasyon var ve spec ikisini de kaçırıyor:

**(a) Satır ekseni (çıkış) rotasyonu — blok boyu `T`.**
Spec'in kendi §7.19'u bunun `ε_S`'i değiştirmediğini söylüyor
(`Σ_{i∈R_t} w_i,S M w_i,Sᵀ = tr(W_S M W_Sᵀ)`, sol ortogonal dönüşüm altında
değişmez) ve **"etkisiz"** diye kapatıyor. Bu sonuç *maske seçimi* için doğru,
*quantization* için yanlış: maskenin değişmemesi tam da istediğimiz şey.
Çıkarım maliyeti `log₂(T)/(d·n_in)` — `T=16, d=0.5, n_in=11008` için **0.0007**.
Fiilen bedava.

**(b) Sütun ekseni rotasyonu — kompaktlanmış survivor matrisi üzerinde.**
Maske dondurulduktan sonra uygulanır, alt-Hessian `H_{S_t S_t}` tam hesaplanabilir.
Tam güçte (blok boyu `d·n`), ama her tile kendi sütun kümesine sahip olduğundan
tile başına ayrı dönüşüm gerekir: maliyet `log₂(d·n)/T` ≈ **0.7–0.8** (T=16),
**11.5×** (T=1), **~0** (T=max).

İkisi de `T`'yi yukarı itiyor. Yani granülerlik ekseninin **üç** kuvveti oluyor
(indeks `1/T`, satır-RHT gücü, sütun-RHT karşılanabilirliği) ve tek bir aşağı
kuvveti (maske özgürlüğü, `τ`).

## G3. Neden bu şimdi belirleyici

GPTQ-4bit survivor başına **4.156** bit istiyor. Bu, her ilgi çekici bütçede
acımasız seyreklik dayatıyor. E8P **2.000** istiyor:

| B | quantizer | T=16'da `d` | seyreklik |
|---|---|---|---|
| 1.50 | GPTQ 4-bit | 0.3459 | **%65.4** — umutsuz |
| 1.50 | QuIP# E8P | 0.7188 | **%28.1** — makul |
| 1.25 | GPTQ 4-bit | 0.2857 | %71.4 |
| 1.25 | QuIP# E8P | 0.5938 | %40.6 |

Ve kaldıraç `(1−1/T)/W` ikiye katlanıyor: `T=16`'da 4-bit'te `0.2256`, E8P'de
**`0.4688`**.

Dahası, E8P ile çapa 2 (2.140625) **fazla cömert** kalıyor — `T=16`'da `d > 1`,
ulaşılamıyor. Yani çalışma kendiliğinden **2 bit altına** kayıyor; tam olarak
motivasyonun tuttuğu ve dense PTQ'nun cevabının olmadığı rejime.

## G4. Literatür ne diyor — araştırma sonucu

### G4.1 Rotasyon tek başına yetmiyor; belirleyici olan kafes codebook'u

QuaRot, weight-only, Llama-2-7B (protokol B, dense 5.47):

| Yöntem | W4 | W3 | W2 |
|---|---|---|---|
| QuaRot-RTN | 6.76 | **Inf** | **Inf** |
| QuaRot-GPTQ | 5.60 | 6.09 | **22.07** |
| QuIP# (aynı protokol) | 5.56 | **5.79** | **6.66** |

**Rotasyon + GPTQ her bit genişliğinde QuIP#'in altında ve 2-bit'te tamamen
çöküyor.** 2 biti mümkün kılan şey rotasyon değil, **kafes codebook'u**.
Bu, "önce ucuz satır-RHT + GPTQ" basamağının değerini büyük ölçüde siliyor:
o yol survivor'ı 2 bite indiremiyor, ki §G3'ün bütün kazancı oradaydı.

### G4.2 Rotasyonlu bazda budama felaket — ve bu belgelenmiş

OBR (arXiv:2509.11177) Tablo 1, Llama2-7B, W-A-KV hepsi 4-bit (dense 5.47):

| Konfig | Seyreklik | Wiki2 |
|---|---|---|
| QuaRot (yalnız quant), 3-4-4 | %0 | 132.97 |
| **QuaRot + Wanda** | %50 | **5868.24** |
| SparseGPT + GPTQ | %50 | 12.94 |
| OBR_RTN | %50 | 9.23 |
| OBR_GPTQ | %50 | 8.40 |

Ve arXiv:2603.18426: *"Rotation amplifies pruning effects, underscoring the
necessity of designing rotation-aware pruning methods."*

Mekanizma benim hipotezimle aynı ama daha keskin: **rotasyon büyüklük
dağılımını düzleştirir, budama ise yoğunlaşmış enerjiden beslenir.** Rotasyonlu
bazda alınan budama kararları yanlış oluyor.

> ⚠️ Bu sayılar W4A4KV4 (aktivasyon ve KV de 4-bit), bizim weight-only ayarımız
> değil. Nicel olarak taşınmazlar; taşınan şey **işaret ve büyüklük sırası**.

### G4.3 Ama bu, önerilen sırayı çürütmüyor — tam tersine

QuaRot+Wanda **önce döndürüp sonra buduyor**. Öneri bunun tersi:

```
buda (orijinal bazda, saliency anlamlıyken) → maskeyi dondur
    → kompaktla → döndür → kafes VQ
```

Belgelenen başarısızlık modu (rotasyonlu bazda yanlış budama kararı) bu sırada
**oluşamaz**, çünkü maske rotasyondan önce ve dokunulmamış bazda seçiliyor.

**Tasarım değişmezi — kod yazılırken korunmalı:** *maske her zaman döndürülmemiş
bazda seçilir; rotasyon yalnızca dondurulmuş maskeden sonra, kompaktlanmış
survivor matrisine uygulanır.*

Bunun neden daha önce yapılmadığına dair makul açıklama: mevcut pipeline'ların
hepsi (QuaRot, QuIP#, QTIP) rotasyonu en başa koyuyor, çünkü quantization için
tasarlanmışlar. Seyreklik sonradan eklenince sıra yanlış kalıyor.

### G4.4 Bedeli: rotasyon artık katlanamıyor

QuaRot'un rotasyonları komşu katmanlara **fold** ediliyor, o yüzden bedava.
Kompaktlanmış survivor'a uygulanan rotasyon fold edilemez — tile başına farklı
sütun kümesi var, gather'dan *sonra* uygulanmalı. Yani `log₂(d·n)/T` çevrimiçi
maliyet gerçek ve kaçınılmaz. `T`'yi yukarı iten kuvvet buradan geliyor.

### G4.5 Sıra sorusu spec'e ters

arXiv:2603.18426'nın "Progressive Intensity Hypothesis"i: *zayıf pertürbasyon
önce*. Bizim rejimde baskın pertürbasyon seyreklik, quantization hafif →
**quantize-then-prune** öneriyor. Spec'in varsayılanı `prune_then_quantize`.
M3'ün `joint` modu zaten planlı; bu ablasyon artık **zorunlu**.

### G4.6 OBR yeni bir öncül — §0.4 güncellenmeli

OBR, ortak quantization+seyrekleştirmeyi rotasyon uyumsuzluğunu çözerek yapan
doğrudan öncül. Spec onu "⚠️ doğrulanmadı" diye listelemiş; artık doğrulandı ve
delta cümlesinde adlandırılmalı. **Bizim farkımız:** OBR verili bir sırayı
onarıyor, biz sırayı değiştiriyoruz ve granülerliği serbest değişken yapıyoruz.

## G5. Revize merdiven

| # | Seçenek | `vq_bits` | B=1.5, T=16 seyreklik | Not |
|---|---|---|---|---|
| 1 | GPTQ 4-bit *(mevcut)* | 4.156 | %65 | Prova: her yerde kaybediyor |
| 2 | Rotasyon + GPTQ 3-bit | 3.148 | %54 | Yeni codebook yok; 3-bit'te çalışıyor (6.09), 2-bit'te çöküyor |
| 3 | **Kompakt-survivor rotasyonu + kafes VQ (E8P)** | **2.000** | **%28** | Tek gerçek yol; amortizasyon 0 |
| 4 | AQLM-survivor *(spec'in planı)* | 2.186 | %31 | En zayıf VQ + en pahalı + kompaktlama riski |

**Önerilen: 3 — ama önce onu tek başına karara bağlayan ucuz deneyi koş (§G6).**

## G6. Kararı veren ucuz deney

Haftalar taahhüt etmeden önce cevaplanması gereken tek soru:

> Kafes VQ (E8P), **kompaktlanmış bir survivor alt-matrisi** üzerinde çalışıyor mu?

Survivor'lar tanım gereği dağılımın büyük ucu — kalın kuyruklu, kafes
quantizer'ın varsaydığı Gauss'tan uzak. Rotasyon bunu düzeltmeli, ama blok boyu
`T` (16) QuIP#'in kullandığından çok küçük; 8-boyutlu E8P için yeterli
Gaussianizasyon sağlayıp sağlamadığı **açık ampirik soru**.

Deney: tek bir katman, tek bir `(T, d)` noktası. Wanda ile maske → kompaktla →
blok-`T` rotasyon → E8P → katman-çıkışı MSE'sini şunlarla karşılaştır:
(a) rotasyonsuz E8P, (b) GPTQ-4bit, (c) dense E8P.
Maliyet: birkaç saat. Çıktı: `vq_bits ≈ 2.0` varsayımı ayakta mı, değil mi.

> **KARAR (kullanıcı, 2026-08-20):** Bu deney **atlanıyor**, doğrudan §G5 satır 3'e
> giriliyor. Kalın-kuyruk riski §H5'te açık varsayım olarak taşınıyor.

---

# §H. Uygulama planı — E8P-survivor hattı

## H1. Tasarım değişmezi

Bütün hattın doğruluğu tek bir kurala bağlı:

> **Maske her zaman döndürülmemiş bazda seçilir. Rotasyon yalnızca dondurulmuş
> maskeden sonra, kompaktlanmış survivor matrisine uygulanır.**

Sıra: `skorla → maske seç → dondur → kompaktla → döndür → E8P → telafi`.
Bunun tersi (QuaRot+Wanda) Llama-2-7B'de 5868 ppl veriyor. Bu kural koda
assert olarak girmeli, yoruma bırakılmamalı.

## H2. Bütçe yapısı yeniden kuruluyor

`W = 2.0` bütün çapaları kaydırıyor. Canlı bant (§3.5 filtresi):

| Survivor quantizer | `W` | canlı bant |
|---|---|---|
| GPTQ 4-bit | 4.156 | 1.83 – 3.74 |
| **E8P** | **2.000** | **1.40 – 1.80** |

> **Yapısal sonuç:** iki aile canlı bütçelerde **örtüşmüyor**. E8P-survivor ile
> GPTQ4-survivor aynı canlı hücrede bütçe-eşleştirilemez. M2'nin `weight_bits`
> ekseni büyük ölçüde işlevsizleşiyor — survivor quantizer'ı bütçe rejimini
> belirliyor. Grid buna göre yeniden kurulmalı.

Yeni yapı:

- **Referans duvarı (çapa değil):** QTIP 2-bit'in ölçülen maliyeti (~2.0 bit),
  ppl 5.86 (protokol A). Altına indiğimiz PTQ tabanı budur.
- **Birincil bant:** `B ∈ {1.75, 1.60, 1.50}` — üçü de E8P için canlı.
- **Yüksek çapa (aile karşılaştırması için korunur):** QTIP 3-bit'in ölçülen
  maliyeti; orada GPTQ-4bit ailesi canlı ve Kapı A provasıyla süreklilik kalır.

`B = 1.5`'te ızgara (çok temiz sayılar):

| T | 1 | 2 | 4 | 8 | 16 | 32 | max |
|---|---|---|---|---|---|---|---|
| `d` | 0.2500 | 0.5000 | 0.6250 | 0.6875 | 0.7188 | 0.7344 | 0.7500 |

Ve kaldıraç ikiye katlanıyor: `T=16` avantajı **0.46875** (4-bit'te 0.2256).

**Bu bandın tamamı 2 bitin altında** — dense PTQ'nun cevabının olmadığı yer
(QuIP# 2-bit 6.66, QuaRot-GPTQ 2-bit 22.07). Tezin motivasyonu ilk kez
deney bandıyla çakışıyor.

## H3. Modüller

| Dosya | İş |
|---|---|
| `accounting.py` | `vq_bits_from_spec` E8P vakası (amortizasyon 0) zaten var; rotasyon latency terimi ve yeni çapa yardımcıları eklenecek |
| `tests/golden.py` · `test_accounting.py` | E8P bandı için golden tablo (yukarıdaki `B=1.5` satırı), canlı bant 1.4–1.8, `B*=1.148962` |
| `compact.py` *(yeni)* | Tile başına survivor'ları yoğun bloklara topla; `S_t` indeksleri ve ters eşleme |
| `rotation.py` *(yeni)* | Kompakt survivor üzerinde blok-diyagonal RHT; maske-koruma testi |
| `quantize.py` | E8P kafes quantizer; `H_{S_t S_t}` alt-Hessian ile kalibrasyon |
| `scoring.py` · `tiling.py` · `prune.py` | Maske seçimi — **döndürülmemiş bazda**, H1'in assert'i burada |
| `experiments/m1_gates.py` | Grid yeni bantta yeniden kurulur |

Mevcut `accounting.py` API'si (`density_for_budget`, `is_live`, `live_band`,
`budget_matched_grid`, `Config`) olduğu gibi kullanılabilir — `vq_bits=2.0`
geçmek yeterli. Yeniden yazılmasına gerek yok.

## H4. Sıra ablasyonu artık zorunlu

arXiv:2603.18426 quantize-then-prune öneriyor, spec `prune_then_quantize`
varsayıyor. M3'ün `joint` modu planlıydı; artık **M2'ye çekilmeli** çünkü
hangi sıranın kullanıldığı bütün sayıları etkiliyor.

## H5. Taşınan açık varsayım

> **E8P'nin kompaktlanmış survivor alt-matrisinde `vq_bits ≈ 2.0` kalitesini
> koruduğu VARSAYILIYOR; doğrulanmadı.**

Gerekçesi: survivor'lar tanım gereği dağılımın büyük ucu (kalın kuyruklu),
kafes quantizer ise Gauss varsayıyor. Rotasyon bunu düzeltmeli ama blok boyu
`T=16`, QuIP#'in kullandığından küçük — 8-boyutlu E8P için yeterli
Gaussianizasyon sağladığı gösterilmedi.

**Erken uyarı:** ilk katman E8P'den geçtiğinde katman-çıkışı MSE'sini dense
E8P ile karşılaştır. 2× ve üzeri bir fark varsayımın düştüğünü söyler; o
noktada §G5 satır 2'ye (rotasyon + GPTQ 3-bit, `W=3.148`) düşülür ve bant
1.83–2.83'e kayar. Bu geri dönüş yolu kodda hazır tutulmalı.

## H6. Doğrulama

- `pytest` yeşil; yeni golden tablo `B=1.5` satırını birebir üretiyor
- **Maske-koruma testi:** rastgele blok-diyagonal ortogonal `Q` için
  `support(Q @ W_compact) == support(W_compact)` ve grup-OBS saliency'si
  değişmiyor (±1e-10) — §7.19'un iddiasının kod içi kanıtı
- **Sıra assert'i:** rotasyon uygulanmış bir matris üzerinde maske seçimi
  çağrılırsa hata fırlat (H1)
- Tek katman uçtan uca: Llama-2-7B `mlp.down_proj`, `B=1.5`, `T=16` →
  katman-çıkışı MSE'si dense E8P ve GPTQ-4bit ile karşılaştırılır
- `budget_matched_grid(1.5, vq_bits_grid=(2.0,))` yalnızca canlı hücreleri
  döndürüyor ve hepsinin offset'i 0

## H7. Riskler — makalede yorumda belirtilmeli

- **Survivor dağılımı kalın kuyruklu.** Survivor'lar tanım gereği büyük ağırlıklar;
  bu RHT'yi *daha* değerli yapar (outlier rejimi tam da RHT'nin hedefi), ama aynı
  bitte dense matristen daha fazla bit gerektirebilir. §3.6'nın 2. ablasyonu
  (quantization hatası ↔ maske hatası ayrımı) bunu zaten ölçüyor. Ana risk §H5.
- **Kernel yok (§8).** RHT, roofline ile gerçek latency arasındaki açığı büyütür.
  Roofline alt sınır olarak sunulmalı, hız iddiası yapılmamalı.
- **E8P implementasyonu gerçek iş.** QuIP# kodu açık ama survivor alt-Hessian'ı
  (`H_{S_t S_t}`) ile yeniden kalibrasyon gerekiyor.
- **`vq_bits = 2.0` ölçülmeli.** Spec §3.2 yapılandırılmış codebook için
  amortizasyonu 0 alıyor, ama QuIP#'in katman-başı ölçekleri ve Hadamard
  seed'leri var. Checkpoint dosya boyutundan doğrula.
- **Ek todo:** `golden.rht_overhead_ratio` şu an yalnızca sütun ekseni için;
  satır ekseni için `log2(T)/(d*n_in)` formülü eklenecek.

---

# §I. Uygulamadan çıkan bulgular (2026-08-20)

Hat yazılırken ortaya çıkan, plan ve spec'i değiştiren üç şey.

## I1. Kapı B, spec'in 3 seed'i ile karara bağlanamaz

`gate_b`'yi ilk yazdığımda **gürültüde "interior" verdi** — yani hiçbir etki
içermeyen veride tez lehine sonuç üretti. İki ayrı sebep:

1. **Seçim yanlılığı.** `T*` argmin ile seçilip *aynı* çekilişlerle test ediliyor.
   Çoklu karşılaştırma düzeltmesi şart → aday iç `T` sayısına göre Bonferroni.
2. **Çekiliş sayısı.** 3 çekilişle percentile bootstrap 3 sayıyı yeniden
   örnekliyor; %95 aralığının %95 kapsaması yok.

`gate_b` artık `min_seeds=5` altında **"undetermined"** döndürüyor ve alpha'yı
Bonferroni ile düzeltiyor. Altı ayrı gürültü çekilişinde test edildi.

> **Spec çelişkisi:** §6 "seed ≥ 3" diyor. Üç seed ortalama raporlamaya yeter,
> **bu kapıyı karara bağlamaya yetmez.** Ön-kayda Kapı B için ayrı ve daha
> yüksek bir seed sayısı yazılmalı.

## I2. Telafinin kazancını 2-bit quantizer geri alıyor

`B=1.5`, sentetik katman, göreli çıkış hatasında telafinin etkisi:

| | quantization yok | 2-bit E8P ile |
|---|---|---|
| T=4 | **−19.1%** | +2.5% |
| T=8 | **−20.7%** | +4.4% |
| T=16 | **−16.1%** | −7.6% |

Mekanizma: OBS telafisi budanan ağırlığın işini hayatta kalanlara yüklüyor,
onlar **büyüyor ve yayılıyor**, 2 bitlik quantizer de onları daha çok eziyor.
Ardışık uygulandığında ikinci adım birincinin kazandığını geri alıyor.

Bu, §G4.5'te literatürden aktardığım "Progressive Intensity Hypothesis"in
(arXiv:2603.18426) kendi hattımızdaki somut kanıtı. Çözüm ardışıklık değil,
**quantization-farkında telafi** (SparseGPT joint modu, OBR). §H4'ün sıra
ablasyonunu M2'ye çekme kararı artık kanıtlı.

> ✅ **ÇÖZÜLDÜ (LDLQ ile).** Telafinin etkisi, `B=1.5`, rotasyon açık:
>
> | T | düz yuvarlama | LDLQ ile |
> |---|---|---|
> | 4 | +2.5% | **−5.8%** |
> | 8 | +7.5% | **−13.9%** |
> | 16 | −4.8% | −8.2% |
> | 32 | −2.4% | −9.1% |
>
> LDLQ quantization hatasını Hessian'ın ucuz yönlerine ittiği için telafinin
> şişirdiği survivor'lar artık ezilmiyor; iki adım kavga etmeyi bırakıyor.
> Sıra ablasyonu yine de yapılmalı — ama artık "hattı düzelten" değil,
> "ne kadar daha iyi olur" sorusu.

## I3. Rotasyon, aktivasyon-ağırlıklı hedefte işe yaramıyor — henüz

`test_quantize` rotasyonun kafes uyumunu **ağırlık uzayında** ~2.2 dB
iyileştirdiğini gösteriyor. Ama gerçek hedef aktivasyon-ağırlıklı ve orada
rotasyon kazandırmıyor (bu fixture'da hafifçe kötüleştiriyor).

Sebep yapısal: RHT quantization hatasını **izotropik** yapar. İzotropik hata
ancak Hessian da izotropikse optimaldir; değilse doğru olan hatayı düşük-eğrilik
yönlerine itmektir — OBS'in yaptığı tam da bu, rotasyon da tam da bunu bozuyor.

QuIP# bu tuzağa düşmüyor çünkü yuvarlamayı **döndürülmüş uzayın içinde**,
döndürülmüş Hessian `V H Vᵀ` ile yapıyor (LDLQ). Bizim quantizer şu an düz
en-yakın-komşu, yani rotasyonun bedelini ödeyip faydasını toplamıyoruz.

> ✅ **ÇÖZÜLDÜ — ama mekanizma ilk teşhisimden farklı çıktı.**
>
> `quantize.py`'ye LDLQ eklendi (`ldlq_quantize`): tile başına alt-Hessian
> `H[S_t,S_t]`, blokla aynı baza döndürülüp (`Q H Qᵀ`) sekizli gruplarda
> ileri-besleme ile yuvarlama. Hedef tam korunuyor:
> `tr((E Qᵀ)(Q H Qᵀ)(E Qᵀ)ᵀ) = tr(E H Eᵀ)`.
>
> **Hat ölçümü** (`B=1.5`, telafi açık) — rotasyonun etkisi:
>
> | T | düz yuvarlama | LDLQ ile |
> |---|---|---|
> | 4 | +2.6% | **−29.5%** |
> | 8 | +4.6% | **−23.2%** |
> | 16 | +0.1% | **−31.0%** |
> | 32 | +3.9% | **−27.0%** |
>
> **İzole ölçüm daha dar bir iddiayı destekliyor.** 16×64 bloklarda:
>
> | blok | rotasyon, düz | rotasyon, LDLQ |
> |---|---|---|
> | Gaussian | +17.5% | +4.8% |
> | kalın kuyruklu | **−61.7%** | **−39.0%** |
>
> Yani rotasyonun değeri **LDLQ'dan değil, ağırlık dağılımının izotropik
> olmamasından** geliyor. RHT outlier yaymak için var; Gaussian blokta
> yayılacak bir şey yok, rotasyon sadece maliyet. Survivor'lar tanım gereği
> kalın kuyruklu olduğu için rotasyon bu hatta yerini hak ediyor.
>
> **Makaleye girecek doğru cümle:** *"rotasyon `T`'yi yukarı iten bir kuvvettir"*
> değil de *"rotasyon, survivor dağılımı kalın kuyruklu olduğu için kazandırır;
> maliyeti `log₂(k)/T` olduğundan bu kazanç ancak büyük `T`'de karşılanabilir."*
> İkinci cümle hem doğru hem daha güçlü — çünkü kazancın kaynağını da açıklıyor.
