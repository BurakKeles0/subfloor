# Durum ve Devir Belgesi

> **Bu belge, bağlam kaybolduğunda projeye kaldığı yerden devam edebilmek için var.**
> Kod ne yaptığını söyler; bu belge **neden öyle olduğunu** söyler.
> Son güncelleme: 2026-08-21.

---

## 1. Nerede duruyoruz — üç cümle

Hat uçtan uca çalışıyor ve **gerçek Llama-2-7B üzerinde ilk ölçüm alındı**:
dense perplexity, yayımlanmış değerlerin 0.006 içinde yeniden üretildi. Bu, hem
protokol sorusunu çözdü hem de bütün makinenin gerçek ağırlıklara doğru
bağlandığını kanıtladı. **Ama hiçbir sıkıştırma kalitesi henüz gerçek modelde
ölçülmedi** — dense baseline dışındaki her sayı ya katman düzeyi proxy'si ya
sentetik.

---

## 2. Proje 60 saniyede

**Soru.** Yoğun PTQ'nun pratik tabanı ~2 bit. Altında çöküyor (QuaRot-GPTQ
2-bit → 22.07 ppl). Seyrekliğin tabanı ise **indeks formatına** bağlı: bitmap
1 bit/pozisyonun altına inemez, ama `T` satırın paylaştığı bir indeks `1/T`'ye
iner. 2 bitin altındaki bütçelerde `(survivor quantizer, granularity, density)`
üçlüsü nasıl seçilmeli?

**Neden önemli.** Bit bütçesi doğrudan bağlam uzunluğudur. Llama-2-70B, 24 GiB
kart: 2.0 bit → ~15.6k bağlam, 1.5 bit → ~28.4k. Yani 2 bitin altı bağlamı ikiye
katlamak demek.

**Çekirdek özdeşlik.** Bitmap rejiminde
`d(T) − d(1) = (1 − 1/T)/W` — **bütçeden bağımsız sabit**. Oranın büyümesi
paydanın küçülmesinden; "oran büyüyor" diye yazmak yaygın ve yanlış okuma.
Kaldıraç `W` küçüldükçe büyür: GPTQ-4bit `0.2256`, E8P (`W=2.0`) **`0.4688`**.

**Tasarım değişmezi.**
```
skorla → maske seç (döndürülmemiş bazda) → dondur
       → kompaktla → döndür → LDLQ → telafi
```
Maske **her zaman** döndürülmemiş bazda seçilir. Rotasyonlu bazda budama modeli
yok ediyor (QuaRot+Wanda %50 → **5868 ppl**, OBR Tablo 1). `prune()` yanlış
sırayla çağrılırsa hata fırlatır.

---

## 3. Ne doğrulandı, ne varsayıldı — en önemli tablo

### Doğrulanmış

| Ne | Nasıl |
|---|---|
| Muhasebenin tamamı | `tests/golden.py` tam kesirli aritmetikle bağımsız türetiyor, `accounting.py` genel dispatch ile hesaplıyor. İki yol, tek cevap |
| E8P codebook | 227+29 kaynak örüntüsü enumerasyonla, 2¹⁶ ayrık kodsözcüğü, kafes üyeliği, **tam 2 bit/ağırlık** |
| Rotasyonun maskeyi koruduğu | Her iki eksende, her `T`'de, destek birebir |
| Telafinin ileriye-only olduğu | Son sütunu budayınca öndekiler değişmiyor |
| Telafinin kanal korelasyonundan beslendiği | Bağımsız kanallarda oran 0.7+, korelasyonlu 0.13 |
| Kalibrasyonun **sıkıştırılmış** modeli okuduğu | Sentetik bloklarda ve gerçek Llama'da |
| Adaptörün modelin hesabını yeniden ürettiği | Elle sürülen bloklar → modelin logit'leri, 1e-5 |
| Akışlı eval == tam model eval | 1e-6 |
| Kapı B'nin gürültüde geçmediği | 6 ayrı gürültü çekilişi |
| **Dense ppl (gerçek model)** | 2048 → 5.4675, 4096 → 5.1143; yayımlanmıştan <0.006 |
| **Transfer sapması** | `Δ = Q + τ` tahmin edicisi kurulup gerçek hattın yanında koşuldu. `T=1` kimlik kontrolü **tam sıfır**; sapma gürültüyü **12.3×** aşıyor |
| **Kapı B'nin gücü** | 600 denemelik simülasyon, **gerçek `gate_b` çağrılarak**. 5 çekiliş → 2.29 σ saptıyor; ölçülen etki 6.7 σ. Tip-I her yerde %5'in altında |
| **`vq_bits = 2.0` (maliyet)** | QuIP# E8P ve QTIP releases'lerinin manifest'i; kodsözcüğü yükü **tam 2.000000**, yan bilgiyle 2.005204 / 2.006740. Manifest ve dosya boyutu iki bağımsız yol, tam aynı sayı |

### Varsayım — doğrulanmadı

> **E8P'nin kompaktlanmış survivor alt-matrisinde 2 bit kalitesini koruduğu.**
> Survivor'lar tanım gereği dağılımın kalın kuyruğu; kafes quantizer Gauss'a
> yakın girdi ister. Rotasyon bunu düzeltmeli ama **gerçek modelde gösterilmedi.**

Bu varsayımı sınayacak ucuz deney **bilinçli olarak atlandı** (kullanıcı kararı,
2026-08-20). Erken uyarı kuralı: ilk katman E8P'den geçtiğinde katman-çıkışı
MSE'si dense E8P referansının 2 katını aşarsa varsayım düşmüş sayılır; geri
dönüş yolu rotasyon + GPTQ-3bit (`W=3.148`), bant 1.83–2.83'e kayar.

### Henüz hiç ölçülmemiş

Sıkıştırılmış modelin perplexity'si. Kapı A ve Kapı B'nin **hiçbir gerçek
verisi yok**. Sentetik smoke testte hata eğrisi U şeklinde çıkıyor ve Kapı A
geçiyor — **ama veriyi biz ürettik, bu tez lehine kanıt değil.**

---

## 4. Alınan kararlar ve gerekçeleri

| Karar | Tarih | Gerekçe |
|---|---|---|
| Survivor quantizer **GPTQ-4bit → QuIP# E8P** | 08-20 | Kapı A provası: GPTQ-4bit survivor'larla literatürün konuşabildiği her yerde kaybediliyor. `W` 4.156 → 2.000, `B=1.5`'te `T=16` seyrekliği %65 → %28 |
| Ucuz doğrulama deneyi **atlandı** | 08-20 | Kullanıcı kararı; risk §3'te açık varsayım olarak taşınıyor |
| Bant **1.75 / 1.60 / 1.50** | 08-20 | E8P'nin canlı bandı 1.40–1.80; çalışma kendiliğinden 2 bitin altına kaydı |
| Çapa **QTIP/QuIP#**, GPTQ değil | 08-20 | GPTQ 3-bit sınıfın en zayıfı; ona çapalanırsa Kapı A kolay geçer ama savunulamaz |
| **LDLQ eklendi** | 08-20 | Rotasyon Hessian-farkında yuvarlama olmadan maliyeti ödeyip faydasını toplamıyordu |
| Kapı B için **≥5 çekiliş** | 08-20 | 3 seed ile `gate_b` saf gürültüde "interior" verdi. Spec §6'nın "seed ≥ 3"ü bu kapı için yetersiz |
| Checkpoint: **NousResearch aynası** | 08-21 | Resmi repo kapılı, lisans onayı gerekiyor; ayna kapısız ve dense ppl ölçümü ağırlıkların doğruluğunu zaten teyit etti |
| **seqlen 4096 birincil** | 08-21 | `dense-5.12` ailesi hem budama baseline'larını hem QTIP/QuIP#'i taşıyor; Kapı A'nın rakibi orada |

---

## 5. Planı değiştiren bulgular

**§0.5 tersine döndü.** v6 incoherence processing'i en büyük risk sayıp
QuIP#/QTIP'i toptan eliyordu. Eleme fazla genişti: maske dondurulduktan sonra
rotasyon onu bozamaz. Belgelenen çöküş bir **sıra** problemi.

**Rotasyonun değeri LDLQ'dan değil, dağılımdan geliyor.** Önce "rotasyonun
faydası tamamen Hessian-farkındalığa bağlı" dedim, fazla genelmiş. İzole ölçüm:

| blok | rotasyon, düz | rotasyon, LDLQ |
|---|---|---|
| Gaussian | +17.5% (zarar) | +4.8% (zarar) |
| kalın kuyruklu | **−61.7%** | **−39.0%** |

Rotasyon, ağırlık dağılımı izotropik olmadığında kazandırıyor. Survivor'lar
tanım gereği kalın kuyruklu — bu yüzden yerini hak ediyor.

**LDLQ, telafi/quantization çakışmasını çözdü.** `B=1.5`, telafinin etkisi:
düz yuvarlamada T=4'te **+2.5%**, T=8'de **+7.5%** (zararlı); LDLQ ile
**−5.8%** ve **−13.9%** (faydalı).

**VENOM'un `V`'si bizim `T`'miz.** V:N:M formülü M0'da dolduruldu ve yapısal
bir şey ortaya çıktı: VENOM `V` satırın paylaştığı bir sütun seçimi kullanıyor,
yani indeksi `1/V` ile amortize ediyor. **İndeks amortizasyonu yeni değil.**
Katkı "amortize etmek" değil, `(T, d)` düzlemini bir bütçe altında taramak.

**SU ve SV aynı şey değil — ve bu hattın maliyetini kurtarıyor.** QuIP#'in yan
bilgisini ölçerken çıktı: `SU` (girdi ekseni) ince ayarın ±1'den zar zor
kıpırdattığı bir işaret vektörü (11008 girdide 26 ayrık değer, hepsi 1'in %0.4
yakınında); `SV` (çıktı ekseni) gerçek kanal-başı ölçek. QTIP'te aynı asimetri
daha da keskin (SU 8 ayrık değer; SV ~0.018 civarında saf ölçek).

Önemi şu: her tile kendi sütun kümesine sahip olduğu için, tile başına
**öğrenilmiş** bir sütun vektörü tutmak `16/T` bit (T=16'da 1.0 — bant bunu
kaldırmaz), paketlenmiş işaret olarak tutmak `1/T` bit (0.0625) demekti.
İkincisi bile indeksin üstüne aynı `1/T` şeklinde binen bir terim. Ölçüm bunu
ödemek zorunda olmadığımızı gösteriyor: köşegen gather ile yer değiştirdiği
için **girdi ekseni köşegeni global tutulabilir**, tile başına kalan tek şey
rotasyonun kendisi, o da seed'den üretilirse yük taşımıyor. Ayrıştırılmış
tasarımda maliyet **0.0077 bit/survivor** ve `T` ile neredeyse sabit.
`accounting.rotation_side_bits`.

**Ayrılabilirlik varsayımı büyük `T`'ye karşı önyargılı.** Transfer pilotu
`τ`'nun quantization'sız ölçüldüğünde sistematik olarak **büyük** çıktığını
gösterdi — `T=2` dışında her yerde, ve fark `T` ile büyüyor (T=max'ta 0.172'ye
karşı 0.135). Mekanizma makul: 2 bitte quantization hatası maske kalitesi
farkının bir kısmını zaten örtüyor. Sonuç: **model tile'ların maliyetini
olduğundan pahalı gösteriyor.** M1'de büyük `T` tahminden iyi çıkarsa bu
beklenen bir şeydir, tez lehine kanıt değil — ön-kayıt §5.1'e yazıldı.

İyi haber: sapma işaret değiştirdiği halde `T*` kaymadı, tahmin de ölçüm de
`T*=4` verdi. Garanti değildi; her koşuda `argmin_agreement` olarak raporlanıyor.

**Toleransın seed varyansından türetilmesi 12.3 kat küçük olurdu.** Denetimin B3
uyarısı ampirik olarak doğrulandı: en kötü sapma 0.0368, ortalama çekiliş
gürültüsü 0.0030. Kural artık `1.5 × max_T |sapma(T)|` ve ön-kayıtta.

**Kapı B'nin verdikti güvenli, `T*` değil.** Güç analizi ikiye ayrıldı ve iki
farklı cevap verdi. Verdikt (içeride mi, uçta mı) 5 çekilişle rahat kararlanıyor
— bağlayıcı uç `T=max` ve o da eşiğin üç katı uzakta. Ama **komşu tile'lar
ayrılmıyor**: sentetik katmanda `T=4` ile `T=8` arası yalnızca **0.31 σ**,
%90 güvenilir bir argmin için ~53 çekiliş gerekiyor. Düz bir iç bölgede 20
çekilişle verdikt %77 doğru, argmin %41.

Sonuç `m1_gates.t_star_set`: argmin değil, argmin'den **ayrılamayanların
kümesi**. Duman testinde Kapı B "interior" derken küme `{2, 4, 8, 16}` çıkıyor —
yani dürüst manşet "T=16 optimal" değil, "optimum içeride, yeri 2–16 arasında".

**Çekilişler yanlış eksende üretiliyordu.** `GateRun` rotasyon seed'ini
değiştiriyordu, ön-kayıt ise kalibrasyon çekilişi diyor. Ölçüldü: kalibrasyon
gürültüsü rotasyon gürültüsünün **1.95 katı**. Öyle koşulsa Kapı B iki kat fazla
kendinden emin çıkacaktı. `run()` artık `LayerProblem` listesi alıyor; tek
problem verilirse eski davranışa düşüyor ama çıktıyı `draw_axis` ile etiketliyor.

**Protokol ayrımı dizi uzunluğuymuş.** Ölçümden önce hipotez olarak kaydedildi,
ölçümde tuttu. Kural "birini seç" değil "pencereyi sabitle".

---

## 5.5 ⏸️ Yarım kalan iş — buradan devam et

**Rotasyonun gerçek katmanda değeri ölçülüyordu; koşu durduruldu (2026-08-23).**
Kurulum tarafı çalışıyor ve doğrulandı:

```bash
HF_HUB_DISABLE_XET=1 python experiments/m0_rotation_value.py --tiles 4 16 max --seqs 16 --rows 512
```

Model yükleniyor, 32,768 gerçek kalibrasyon token'ı alınıyor, blok 0 girdileri
yakalanıyor, `o_proj`'un Hessian'ı birikiyor (4 s). Kalan altı LDLQ kolu
(T=4/16/max × rotasyonlu/düz) **~2.5–3.5 saat** sürüyor. Hiç `rel.err` satırı
alınamadı — **sonuç yok.**

**Neden önemli:** rotasyonun ölçülmüş bütün faydası (`−29.5%` … `−31.0%`)
**sentetik**. Maliyeti ise gerçek: tile başına rotasyon → tile başına baz →
tile başına Cholesky → M1 76 gün. Sentetik bir kazanç için yapısal bir bedel
ödüyoruz ve bu deney tam olarak o boşluğu kapatacaktı.

**Ölçülen tile maliyetleri** (tam çekirdek, `o_proj`, 512 satır):

| kol | tile başına | tile | kol süresi |
|---|---|---|---|
| T=4 | 7.85 s | 128 | ~17 dk |
| T=16 | 54.26 s | 32 | ~29 dk |
| T=max | ~30 dk *(ekstrapole, alt sınır)* | 1 | ~30 dk |

İş 4.6 katına çıkarken süre 6.9 katına çıkıyor — birim maliyet büyük tile'larda
artıyor, çünkü codebook araması bellek-bağlı.

**Üç ucuzlatma seçeneği var, hiçbiri seçilmedi:**

| seçenek | süre | bedeli |
|---|---|---|
| `T=max`'ı at | ~1.5 sa | Rotasyonun çıkarımda bedava olduğu uç gözlemsiz kalır |
| `--rows 256` | ~1.2 sa | Tile başına problem aynı, istatistik zayıflar |
| Ölçeği katman başına oturt | ~25 dk | Hattı değiştirmek olur; iki kolu eşit etkilediği için karşılaştırmayı bozmaz ama açıkça kaydedilmeli |

---

## 6. Sırada ne var

### Hemen yapılabilir (model yerelde, hat çalışıyor)

1. ⛔ **`tau_sweep.py` — BLOKE.** Maliyet modeli çıkarıldı
   (`experiments/m0_cost_model.py`, 2026-08-21) ve süpürme bu makinede
   koşulamıyor: spec'in 25 GPU-saat tahmini yerine **14–43 gün**, ve M1'in
   kendisi **61–170 gün**. Ayrıntı ve seçenekler §7'de. Süpürmeden önce
   hattın maliyeti düşürülmeli.

~~2. Transfer pilotu~~ — **yapıldı, 2026-08-21.** Tolerans kuralı
   `1.5 × max_T |sapma(T)|`; sentetik katmanda 0.0552 (hata seviyesinin %18.8'i).
   Mutlak sayı M1'in ilk bütçesinden donacak. `experiments/m0_transfer_pilot.py`
   (`--reuse` ile türetilmiş sayılar hattı yeniden koşmadan güncellenir).

~~3. Kapı B'nin minimum saptanabilir farkı~~ — **yapıldı, 2026-08-21.**
   Cevap ikiye ayrılıyor: verdikt için 5 çekiliş yeterli (2.29 σ saptıyor,
   etki 6.7 σ); `T=4` ↔ `T=16` de ayrılıyor (2.69 σ). Ama `T=4` ↔ `T=8`
   **ayrılmıyor** (0.31 σ) — bu yüzden `T*` artık küme olarak raporlanıyor.
   Ön-kayıt §7.1–7.4. `experiments/m0_gate_b_power.py`.

~~4. `vq_bits = 2.0` doğrulaması~~ — **yapıldı, 2026-08-21.** Yük tam 2.000000;
   yan bilgi dahil 2.005204 (QTIP 2.006740). Izgara `vq_bits = 2.0`'da
   donduruluyor, düzeltme raporlanan bütçeye geri ekleniyor — gerekçe
   `preregistration.md` §2. `experiments/m0_vq_bits.py`, ağdan yalnızca
   safetensors başlığını çekiyor.

### Sonra

5. **Ön-kaydı dondur ve commit et** (§9 listesi dolunca) — M1'den ÖNCE.
6. **M1** — iki kapı, `B ∈ {1.75, 1.60, 1.50}`, `T ∈ {1,2,4,8,16,32,max}`.
7. **M2** — bütçe süpürmesi, sıra ablasyonu (prune-then-quant vs joint).

### Açık kalan kod işleri

- **Axis A için LDLQ** — şu an `NotImplementedError`; Axis B'de indeks ekseni
  girdi kanalları olduğu için Hessian doğrudan uygulanıyor, Axis A'da sweep
  tile'ın sütunları boyunca olmalı
- **Blockwise (tam SparseGPT) maske seçimi** — M3 teslimatı, şu an `upfront`
- **§3.6'nın üç ablasyonu** — grup konvansiyonu (iki koşullu olmalı),
  quantization/maske hatası ayrımı, hizalama
- **Attention koordinasyonu formülü** — `v_proj`↔`o_proj`, GQA, RoPE çiftleri;
  `T=max` için sert kısıt, hâlâ yalnızca ima edilmiş durumda

---

## 7. Açık riskler

**⛔ EN BÜYÜĞÜ ARTIK BU: hat gerçek katman boyutunda koşulamıyor.**
Maliyet modeli (bu makinede ölçülen sabitlerle) üç ayrı duvar buluyor:

| Duvar | Büyüklük | Sebep | Zorluk |
|---|---|---|---|
| **Bellek** | `T=2`'de **462 GiB** tek tensör | `tile_hessians` `[n_tiles, k, k]`'yı bir kerede ayırıyor | **Kolay** — tile'ları akıtmak 231 MiB'a düşürüyor |
| **Cholesky** | Model başına 10¹⁶ flop; `T=16`'da 15 saat, `T=1`'de 10 saat | Her tile kendi sütun kümesine sahip → tile başına `k³` faktorizasyon | **Yapısal** |
| **Codebook araması** | `T`'den bağımsız 12 saat (CPU f64) | 2¹⁶ kodsözcüğü üzerinde kaba kuvvet en yakın komşu | **Orta** — E8 kafesine doğrudan yuvarlama bunu ~0'a indirir |
| **`fit_scale`** | Yukarıdakinin **6 katı** | LDLQ her tile için 24 aday ölçeği tarıyor, her taramada tüm tile'ı arıyor — ölçüldü, LDLQ'nun **%83'ü** | **En kolay** — QuIP# ölçeği katman başına oturtuyor |

> ⚠️ **Maliyet modelinin ilk sürümü `fit_scale`'i hiç saymıyordu ve altı kat
> yanılıyordu.** Doğrusu `SCALE_FIT_MULTIPLIER = 6.0` olarak modelde, anahtarlı
> ve varsayılanı açık — çünkü kodun bugün yaptığı bu. M1 tahmini **61 → 76 gün**.
>
> ✅ **Bellek duvarı kapatıldı:** `tile_hessian_stream` ile 119 GiB → 239 MiB,
> ve yığılmış yolla **bit-birebir** aynı sonucu verdiği testle sabit.

Cholesky duvarı **kaba kuvvet değil, tasarımdan geliyor**: her tile'ın kendi
sütun kümesi + kendi rotasyonu var, dolayısıyla kendi alt-Hessian'ını
faktorize etmesi gerekiyor. `T=1`'de bu **satır başına** bir faktorizasyon
demek — yani SparseGPT'nin var oluş sebebi olan "row Hessian challenge"ı
birebir yeniden üretiyoruz ve tam bedelini ödüyoruz.

Maliyet `(n_out/T)·k³` olduğu için **ızgaranın ince ucunda yoğunlaşıyor**
(3 bütçe × 5 çekiliş, cuda_f32): `T=2` 419 saat, `T=4` 384, `T=8` 254,
`T=16` 150, `T=32` 86, `T=max` **15**. Yani granülerlik tezinin en çok
veriye ihtiyaç duyduğu bölge, en pahalı bölge.

**Kapı A'nın düşme olasılığı yüksek.** Prova (`gate_a_dry_run.md`)
GPTQ-4bit survivor'larla her satırın düştüğünü gösterdi. E8P aritmetiği
değiştiriyor ama **gösterilmedi**. Karar tablosunun `✗/✓` dalı hazır: proje
durmaz, çerçeve daralır.

**E8P varsayımı** (§3). Düşerse bant 1.83–2.83'e kayar ve tezin "2 bitin altı"
motivasyonu zayıflar.

**Kapı B'nin istatistiksel gücü** — verdikt tarafında **çözüldü** (5 çekiliş
yeterli, §7.1). Kalan risk `T*`'ın kendisi: eğri iç bölgede düzse küme büyük
çıkar ve *hangi* granülerlik sorusu cevapsız kalır. Bu bir başarısızlık değil,
ama manşeti zayıflatır. Ölçülen σ **sentetik**; gerçeği ilk M1 bütçesinden
gelecek ve §7.4'ün uyarlanabilir kontrolü onun için var.

**Takvim.** Spec'in M0–M5 tahmini 8–10 hafta; gerçekçi olarak 2–3 katı.
8 GB kart yerel geliştirme için yeterli ama `τ` süpürmesi gibi çok noktalı
işler için dar.

---

## 8. Ortam tuzakları — saatlere mal oldu, tekrar etmesin

| Sorun | Çözüm |
|---|---|
| **HF indirmeleri takılıyor** (0 B/s, tekrar tekrar) | `HF_HUB_DISABLE_XET=1`. Xet arka ucu bu ağda çalışmıyor; klasik HTTP ile 10–14 MiB/s |
| **Kimliksiz HF istekleri sert kısıtlanıyor** | Token şart. `hf auth login` (diske yazar, her süreç görür). `$env:HF_TOKEN=...` yalnızca o pencerede geçerli, alt süreçler görmez |
| **`snapshot_download` oturumlar arası devam ETMİYOR** | Her yeniden başlatma büyük parçaları sıfırdan indiriyor (her seferinde farklı sonekli `.incomplete`). **Bir kez başlat, kesme.** |
| **Arka plan görev bildirimleri güvenilmez** | "exit 127 ile düştü" dediği halde süreç yaşamaya devam ediyor. Yeniden başlatmadan **önce süreç listesine bak**; yoksa iki indirici aynı repoya yazıp birbirini engelliyor |
| **`torchvision` ABI uyumsuzluğu transformers'ı komple kırıyor** | torch'u yükseltirken torchvision'ı da eşleştir, ya da kaldır (bu projede görüntü yolu yok; transformers `is_vision_available()` ile koruyor) |
| **`load_dataset("wikitext", ...)` reddediliyor** | `Salesforce/wikitext` — çıplak ad artık geçersiz, `namespace/name` gerekiyor |
| **Süreç sayarken kendi ölçüm sürecini sayma** | PowerShell filtresini `python -c` içinden çağırınca komut satırında aranan string geçiyor ve kendini yakalıyor |

**Donanım:** RTX 5060 Laptop, 8 GB VRAM, sm_120 (Blackwell → cu128+ gerekiyor,
`torch 2.12.0+cu130` kurulu). 23.7 GiB RAM, 24 mantıksal çekirdek.
7B fp16 (13.5 GB) GPU'ya sığmıyor → **katman-akışlı zorunlu**, ~2.8 GB tepe.

---

## 9. Repo haritası

| Modül | İş |
|---|---|
| `accounting.py` | bit bütçeleri, `1−1/T`, `B*`, canlı bant, V:N:M |
| `scoring.py` | saliency — iki ağırlık-başı metrik, iki toplama yönü |
| `tiling.py` | tile bölümlemesi, dondurulmuş maske, `align` |
| `prune.py` | maske seçimi + ileriye telafi; **H1 assert'i burada** |
| `compact.py` | survivor'ları tile başına yoğun bloklara topla |
| `rotation.py` | maske-koruyan rotasyon |
| `quantize.py` | E8P codebook + LDLQ |
| `calibrate.py` | sıralı kalibrasyon, `LayerProblem` (**dikiş yeri**) |
| `hf_llama.py` | HF adaptörü — blok 0 girdilerini **yakalar**, yeniden üretmez |
| `eval/perplexity.py` | ppl + protokol koruması + yayımlanmış sayı tablosu |
| `eval/streamed.py` | katman-akışlı ppl (GPU'ya sığmayan model) |
| `experiments/m1_gates.py` | M1'in iki kapısı |
| `experiments/m0_dense_ppl.py` | dense ölçüm + protokol kimliği |
| `experiments/m0_vq_bits.py` | VQ checkpoint maliyeti — manifest'ten, indirmeden |
| `experiments/m0_gate_b_power.py` | Kapı B'nin gücü + boru hattının gürültüsü |
| `experiments/m0_transfer_pilot.py` | `Δ = Q + τ` transfer sapması → tolerans |
| `experiments/m0_cost_model.py` | ölçülen sabitlerle gerçek koşu maliyeti |
| `experiments/m0_rotation_value.py` | rotasyon gerçek katmanda kazandırıyor mu (**yarım**, §5.5) |

**Belgeler:** `spec_v7.md` (şartname) · `preregistration.md` (M1 ön-kaydı,
**dondurulmadı**) · `audit.md` (v6 denetimi, tarihsel kayıt) ·
`gate_a_dry_run.md` (literatür provası) · bu belge.

**Çalıştırma:**
```bash
python -m pytest tests/ -q                    # 353 test
HF_HUB_DISABLE_XET=1 python experiments/m0_dense_ppl.py --seqlens 2048 4096 --device cuda
python experiments/m1_gates.py --synthetic --n-out 64 --n-in 128 --budgets 1.5 --draws 5
python experiments/m0_gate_b_power.py --no-noise   # simülasyon (~15 dk), σ önbellekten
python experiments/m0_transfer_pilot.py --draws 3  # ~8 dk; --reuse ile saniyeler
python experiments/m0_cost_model.py               # ~1 dk, sabitler önbelleklenir
python experiments/m0_vq_bits.py --all       # ~100 KB ağ trafiği, saniyeler
```

---

## 10. Commit geçmişi — ne anlama geliyorlar

| Commit | Ne getirdi |
|---|---|
| `5d7726d` | Hattın tamamı: muhasebeden ppl'e, 322 test |
| `f94a8af` | Ön-kayıt taslağı; dondurma listesi görünür bir olay olsun diye |
| `6af48d2` | v6 denetimi repoya taşındı — kararların gerekçesi versiyonlansın |
| `94dbdce` | Spec v7: kafes VQ etrafında yeniden kuruldu, 4 aritmetik hata düzeldi |
| `2e3e7fc` | README — doğrulanan/varsayılan ayrımı etrafında |
| `c3c5632` | V:N:M formülü VENOM'dan; **özgünlük iddiasını daralttı** |
| `1e6218f` | HF adaptörü — küçük gerçek Llama ile test edilebilir |
| `33d66a4` | Katman-akışlı eval — 7B'yi 8 GB'da ölçmenin yolu |
| `3ee1628` | M0 dense ölçüm betiği (hipotez ölçümden önce kaydedildi) |
| `d80ab14` | **İlk gerçek ölçüm**: protokol sorusu çözüldü |
| `e5ec362` | Bu belge |
| `a1626c6` | VQ maliyeti checkpoint'ten ölçüldü; SU/SV ayrışması bulundu |
| `3d8658f` | Kapı B'nin gücü ölçüldü; `T*` küme oldu; çekiliş ekseni düzeldi |
| `7d1ee48` | Transfer pilotu: tolerans kuralı, ve modelin büyük `T` önyargısı |
| `797aa2e` | Maliyet modeli — ve hattın gerçek boyutta koşamadığının tespiti |
| `baa38a7` | Bellek duvarı kapandı, iki yükleyici hatası düzeldi, `fit_scale` modele girdi |

---

## 11. Çalışma tarzına dair not

Bu projede en pahalı hata sınıfı **sessizce yanlış bir sayı üretmek**. Bu yüzden:

- Golden sabitler elle yazılmaz, türetilir. `tests/golden.py` `accounting.py`'yi
  **import etmez** — golden değerleri çağıran bir test hiçbir şey kanıtlamaz
- Testlerin çoğu davranış değil **iddia** sınıyor
- Doğrulanmamış şeyler açıkça "varsayım" diye işaretlenir
- Bir hipotez ölçümden **önce** yazılır (protokol hipotezi böyle sınandı)
- Kesik/eksik ölçümlerden iddia üretmek koda gömülü olarak engellenir

Bu belge de aynı disiplinin parçası: ne bilindiğini ve ne bilinmediğini ayrı
tutuyor.
