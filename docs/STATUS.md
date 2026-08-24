# Durum ve Devir Belgesi

> **Bağlam kaybolduğunda projeye kaldığı yerden devam edebilmek için var.**
> Kod ne yaptığını söyler; bu belge **neden öyle olduğunu** söyler.
> Son güncelleme: 2026-08-24 · HEAD `cc3e0f4` · Testler: **525 geçiyor, 6 atlanıyor**

---

## 1. Nerede duruyoruz — beş cümle

Hat uçtan uca çalışıyor, gerçek Llama-2-7B'ye bağlı, ve gerçek ağırlıklar
üzerinde üç ölçüm var: dense perplexity (yayımlanmıştan 0.006 içinde),
rotasyonun katman değeri (**−70%**), ve blok genişliğinin etkisi. M0'ın
uçuş-öncesi kalemleri kapandı. **Maliyet artık bağlayıcı kısıt değil:** M1 bu
makinede 120 günden **17 güne**, `τ` süpürmesi 29 günden **5.1 güne** indi — yani
ön-kaydı bloke eden şey ortadan kalktı. Ama **sıkıştırılmış modelin
perplexity'si hâlâ hiç ölçülmedi**; Kapı A'nın ve Kapı B'nin tek bir gerçek
verisi yok. Ve bunun sebebi bilimsel bir karar değil: **tam modeli sıkıştıran
deney betiği hiç yazılmadı.**

---

## 2. Proje 60 saniyede

**Soru.** Yoğun PTQ'nun pratik tabanı ~2 bit; altında çöküyor (QuaRot-GPTQ
2-bit → 22.07 ppl). Seyrekliğin tabanı ise **indeks formatına** bağlı: bitmap
1 bit/pozisyonun altına inemez, ama `T` satırın paylaştığı bir indeks `1/T`'ye
iner. 2 bitin altındaki bütçelerde `(survivor quantizer, granularity, density)`
üçlüsü nasıl seçilmeli?

**Neden önemli.** Bit bütçesi doğrudan bağlam uzunluğudur. Llama-2-70B, 24 GiB
kart: 2.0 bit → ~15.6k bağlam, 1.5 bit → ~28.4k.

**Çekirdek özdeşlik.** Bitmap rejiminde `d(T) − d(1) = (1 − 1/T)/W` —
**bütçeden bağımsız sabit**. Oranın büyümesi paydanın küçülmesinden; "oran
büyüyor" yaygın ve yanlış bir okuma. Kaldıraç `W` küçüldükçe büyür: GPTQ-4bit
`0.2256`, E8P (`W=2.0`) **`0.4688`**.

**Tasarım değişmezi (H1).**
```
skorla → maske seç (döndürülmemiş bazda) → dondur
       → kompaktla → döndür → LDLQ → telafi
```
Maske **her zaman** döndürülmemiş bazda seçilir. Rotasyonlu bazda budama modeli
yok ediyor (QuaRot+Wanda %50 → **5868 ppl**, OBR Tablo 1). `prune()` yanlış
sırayla çağrılırsa hata fırlatır.

**İki kapı.** Kapı A: en iyi seyrek konfigürasyon PTQ tabanını (QTIP 2-bit)
geçiyor mu? Kapı B: optimum `T` içeride mi, uçta mı? **Bağımsızdırlar** — A
düşerken B ayakta kalabilir, ve o durumda çerçeve daralır, proje durmaz.

---

## 3. Ne doğrulandı, ne varsayıldı, ne hiç ölçülmedi

### 3.1 Doğrulanmış

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
| **`vq_bits = 2.0`'ın maliyet tarafı** | QuIP# E8P ve QTIP release'lerinin manifest'i. Yük **tam 2.000000**, yan bilgiyle 2.005204 / 2.006740. Manifest ve dosya boyutu iki bağımsız yol, tam aynı sayı |
| **Kapı B'nin istatistiksel gücü** | 600 denemelik simülasyon, **gerçek `gate_b` çağrılarak**. 5 çekiliş 2.29 σ saptıyor; ölçülen etki 6.7 σ |
| **Transfer sapması** | `Δ = Q + τ` tahmin edicisi gerçek hattın yanında koşuldu. `T=1` kimlik kontrolü **tam sıfır**; sapma çekiliş gürültüsünü **12.3×** aşıyor |
| **Rotasyonun değeri (gerçek katman)** | `o_proj`, gerçek ağırlıklar + 32,768 gerçek token: **−70.1% ortalama** (§5.1) |
| **Blok genişliğinin etkisi** | 51 kol, 5 genişlik × 3 aile × 3 tile (§5.7) |
| **Kafes çözücünün taramaya denkliği** | `nearest_e8p` ile kaba kuvvet, dört ölçekte birebir aynı indeks |
| **Analitik aramanın taramaya denkliği** | float64'te bir milyon vektörde **sıfır** uyuşmazlık; 65,536 kodsözcüğünün hepsi kendine çözülüyor (§6.4) |
| **Derlenmiş ile eager'ın denkliği** | Üç ölçek × üç satır sayısı (düzensiz dâhil), `torch.equal` |
| **Toplu süpürmenin tile-tile'a denkliği** | İki cihaz, iki dtype, iki ölçek politikası, chunk 2'den tile sayısının 4 katına |
| **Akıtılan alt-Hessian'ın yığılmışa denkliği** | Bit-birebir aynı çıktı |
| **Maliyet modelinin kendini doğrulaması** | Hiç uydurulmadığı bir genişlikte (k=7912) gerçek tile süresini %14 içinde tahmin ediyor |

### 3.2 Varsayım — doğrulanmadı

> **E8P'nin kompaktlanmış survivor alt-matrisinde 2 bit KALİTESİNİ koruduğu.**
> Survivor'lar tanım gereği dağılımın kalın kuyruğu; kafes quantizer Gauss'a
> yakın girdi ister.

Maliyet tarafı 08-21'de kapandı (tam 2 bit ödendiği kesin). Açık olan
**karşılığında 2 bitlik kalite alınıp alınmadığı.** Bu varsayımı sınayacak ucuz
deney bilinçli olarak atlandı (kullanıcı kararı, 08-20).

Rotasyonun gerçek katmanda −70% çıkması varsayımı **dolaylı olarak
güçlendiriyor** — gerekçesi "rotasyon kalın kuyruğu düzeltir"di ve rotasyonun
çalıştığı artık ölçüldü. Ama doğrudan kanıt değil.

**Erken uyarı kuralı:** ilk katman E8P'den geçtiğinde katman-çıkışı MSE'si dense
E8P referansının 2 katını aşarsa varsayım düşmüş sayılır; geri dönüş yolu
rotasyon + GPTQ-3bit (`W=3.148`), bant 1.83–2.83'e kayar.

### 3.3 Henüz hiç ölçülmemiş

**Sıkıştırılmış modelin perplexity'si.** Kapı A ve Kapı B'nin **hiçbir gerçek
verisi yok**. Sentetik smoke testte hata eğrisi U şeklinde çıkıyor ve Kapı A
geçiyor — **ama veriyi biz ürettik, bu tez lehine kanıt değil.**

Sebebi artık maliyet değil (bir U eğrisi 23 saat): **tam modeli sıkıştıran betik
yok.** `calibrate.sequential_calibrate` kütüphane olarak var ama yalnızca
testlerden çağrılıyor. Aynı şey `τ` süpürmesi için de geçerli — maliyeti
modellenmiş, kodu yazılmamış (§8.3).

Ayrıca hiç ölçülmemiş: **eval'in gerçek maliyeti** (238 s yalnız WikiText-2;
ön-kayıt §4 C4'ü de şart koşuyor ve 5 zero-shot görev istiyor), **TF32'nin
kalite etkisi**, **`fit_scale`'in doğru hedefe uydurulması**.

---

## 4. Alınan kararlar ve gerekçeleri

| Karar | Tarih | Gerekçe |
|---|---|---|
| Survivor quantizer **GPTQ-4bit → QuIP# E8P** | 08-20 | Kapı A provası: GPTQ-4bit survivor'larla literatürün konuşabildiği her yerde kaybediliyor. `W` 4.156 → 2.000, `B=1.5`'te `T=16` seyrekliği %65 → %28 |
| Ucuz E8P doğrulama deneyi **atlandı** | 08-20 | Kullanıcı kararı; risk §3.2'de açık varsayım olarak taşınıyor |
| Bant **1.75 / 1.60 / 1.50** | 08-20 | E8P'nin canlı bandı 1.40–1.80; çalışma kendiliğinden 2 bitin altına kaydı — motivasyonun tuttuğu yere |
| Çapa **QTIP/QuIP#**, GPTQ değil | 08-20 | GPTQ 3-bit sınıfın en zayıfı; ona çapalanırsa Kapı A kolay geçer ama savunulamaz |
| **LDLQ eklendi** | 08-20 | Rotasyon, Hessian-farkında yuvarlama olmadan maliyeti ödeyip faydasını toplamıyordu |
| Kapı B için **≥5 çekiliş** | 08-20 | 3 seed ile `gate_b` saf gürültüde "interior" verdi. Spec §6'nın "seed ≥ 3"ü bu kapı için yetersiz |
| Checkpoint: **NousResearch aynası** | 08-21 | Resmi repo kapılı; dense ppl ölçümü ağırlıkların doğruluğunu zaten teyit etti |
| **seqlen 4096 birincil** | 08-21 | `dense-5.12` ailesi hem budama baseline'larını hem QTIP/QuIP#'i taşıyor; Kapı A'nın rakibi orada |
| Izgara **`vq_bits = 2.0`'da donduruldu** | 08-21 | Düzeltme her hücrede aynı göreli miktarda (%0.26) — bütçe-eşleşmesini bozmuyor. Tam 2 ise `B=1.5` ızgarasını tam dyadic yapıyor ve `golden.py`'nin bağımsız türetmesi buna dayanıyor |
| Tolerans kuralı **`1.5 × max_T \|sapma(T)\|`** | 08-21 | Seed varyansından türetilseydi **12.3 kat** küçük olurdu ve ön-kayıt tanım gereği "tutmadı" dalına kilitlenirdi |
| `T*` **nokta değil küme** olarak raporlanır | 08-21 | Verdikt ile `T*` aynı güvenilirlikte değil: düz iç bölgede 20 çekilişle verdikt %77, argmin %41 |
| Çekiliş ekseni **kalibrasyon**, rotasyon seed'i değil | 08-21 | Ölçüldü: kalibrasyon gürültüsü rotasyon gürültüsünün **1.95 katı** |
| Hat **cuda/float32**'ye taşındı | 08-23 | Uçtan uca **16–45×**, ağırlık farkı 5e-08 — float32'nin kendi epsilon'u düzeyinde |
| **Katman-başı ölçek reddedildi** | 08-23 | Ölçüldü: %11 kalite kaybı, hız kazancı yok. Yeniden ölçümde T=4'te **+87.9%** |
| **E8 kafes çözücü** kaba kuvvetin yerine | 08-23 | Baskın terim buydu. CPU 3.51×, GPU 1.87×, çıktı birebir aynı |
| **`hessian_block=512`**, rotasyon tam genişlikte | 08-23 | Geri beslemeyi daraltmak kaliteyi **iyileştiriyor** (−11/−23/−16%), rotasyonu daraltmak bozuyor (+43/+38/+44%). Sonradan çıktı ki toplu süpürmenin de önkoşulu |
| **Süpürme tile'lar arasında toplu** (`chunk="auto"`) | 08-23 | Süpürme %99.6 boşta duruyordu. Süpürmede 5–12×, çıktı bit-birebir aynı |
| **Ölçek örneklemesi reddedildi** | 08-23 | Ortalama bedeli küçük ama tohumdan tohuma **15.8 puan** oynuyor — Kapı B'nin ayırmaya çalıştığı 0.31 σ'yı boğar (§5.8) |
| fp16 arama **eklendi, varsayılan kapalı** | 08-23 | 1.3–1.7×, bedel ≤%1 ve **belirlenimci**. Kaliteyi ölçülebilir biçimde değiştirdiği için varsayılan olması bir karar gerektirir |
| **Analitik en-yakın-kodsözcüğü** taramanın yerine | 08-23 | Kodsözcüğü uzayının yapısı aramayı çözüyor. Uçtan uca 1.3–4.0×, float64'te kesin (§6.4) |
| **Triton kuruldu, iki zincir füzyonlandı** | 08-24 | GPU %28.4 meşguldü; boşta geçenin %80'i fırlatma. Uçtan uca 1.64–1.87×, çıktı birebir aynı (§6.5) |
| Maliyet modelinin varsayılanı **hattın koştuğu konfigürasyon** | 08-24 | Kimsenin koşmadığı bir konfigürasyonu fiyatlamak, iyimser bir sabitle aynı hata — yalnız ters yöne bakıyor |

---

## 5. Bilimsel bulgular — planı değiştirenler

Önem sırasına göre, kronolojik değil.

### 5.1 ⭐ Rotasyon gerçek katmanda sandığımızdan çok daha değerli

`layers.0.self_attn.o_proj` (512 çıktı satırı), gerçek Llama-2-7B ağırlıkları,
32,768 gerçek kalibrasyon token'ı, `B=1.5`, cuda/float32:

| T | d | düz | rotasyonlu | **değişim** | sentetik hattın dediği |
|---|---|---|---|---|---|
| 4 | 0.6250 | 0.47422 | 0.09649 | **−79.7%** | −29.5% |
| 16 | 0.7188 | 0.54423 | 0.19530 | **−64.1%** | −31.0% |
| max | 0.7500 | 0.55738 | 0.18655 | **−66.5%** | — |

Ortalama gerçek **−70.1%**, sentetik −30.2%.

> **Bir çerçeveyi çürüttü.** "Sentetik bir kazanç için yapısal bir bedel
> ödüyoruz" diyordum. Kazanç sentetik değil — **sentetik olan, kazancı iki-üç
> kat eksik ölçmüş.**

**Sonuç: "rotasyonu bırak" seçeneği kapandı.** **Kapsam:** tek katman, tek
çekiliş, katman-çıkışı hatası — perplexity değil.

### 5.2 §0.5 tersine döndü

v6 incoherence processing'i en büyük risk sayıp QuIP#/QTIP'i toptan eliyordu.
Eleme fazla genişti: maske dondurulduktan sonra rotasyon onu bozamaz. Belgelenen
çöküş bir **sıra** problemi. Bu, E8P'ye geçişin kapısını açan adımdı.

### 5.3 Rotasyonun değeri LDLQ'dan değil, dağılımdan geliyor

İzole ölçüm (16×64 blok, korelasyonlu Hessian):

| blok | rotasyon, düz | rotasyon, LDLQ |
|---|---|---|
| Gaussian | +17.5% (zarar) | +4.8% (zarar) |
| kalın kuyruklu | **−61.7%** | **−39.0%** |

LDLQ yine de zorunlu: onsuz rotasyon maliyeti ödeyip faydasını toplamıyor
(hat ölçümü: T=4'te +2.6% → −29.5%).

### 5.4 SU ve SV aynı şey değil

QuIP#'in yan bilgisini ölçerken çıktı: `SU` (girdi ekseni) ince ayarın ±1'den
zar zor kıpırdattığı bir işaret vektörü; `SV` (çıktı ekseni) gerçek kanal-başı
ölçek. Önemi: tile başına **öğrenilmiş** bir sütun vektörü `16/T` bit demekti
(T=16'da 1.0 — bant kaldırmaz). Ölçüm bunu ödemek zorunda olmadığımızı
gösterdi: ayrıştırılmış tasarımda **0.0077 bit/survivor**, `T` ile neredeyse
sabit (`accounting.rotation_side_bits`).

### 5.5 Ayrılabilirlik varsayımı büyük `T`'ye karşı önyargılı

Transfer pilotu `τ`'nun quantization'sız ölçüldüğünde sistematik olarak
**büyük** çıktığını gösterdi — `T=2` dışında her yerde, fark `T` ile büyüyor.
Mekanizma: 2 bitte quantization hatası maske kalitesi farkının bir kısmını
zaten örtüyor. Sonuç: **model tile'ların maliyetini olduğundan pahalı
gösteriyor.** M1'de büyük `T` tahminden iyi çıkarsa bu beklenen bir şeydir,
tez lehine kanıt değil — ön-kayıt §5.1'e yazıldı.

İyi haber: sapma işaret değiştirdiği halde `T*` kaymadı (tahmin de ölçüm de
`T*=4`). Garanti değildi; her koşuda `argmin_agreement` olarak raporlanıyor.

### 5.6 Kapı B'nin verdikti güvenli, `T*` değil

Verdikt 5 çekilişle rahat kararlanıyor — bağlayıcı uç `T=max`, eşiğin üç katı
uzakta. Ama **komşu tile'lar ayrılmıyor**: `T=4` ↔ `T=8` arası 0.31 σ, %90
güvenilir bir argmin için ~53 çekiliş gerekir. Sonuç `m1_gates.t_star_set`:
argmin değil, argmin'den ayrılamayanların kümesi. Dürüst manşet "T=16 optimal"
değil, "optimum içeride, yeri 2–16 arasında".

### 5.7 Daraltılabilen şey geri besleme, rotasyon değil

Eski §6.3 "rotasyonu 8'lik gruplara blok-köşegen kısıtla" diyordu. İki yönden
yanlış çıktı.

**Öncül.** `rotate` zaten `share_across_tiles=True` ile **tek bir rotasyonu
bütün katmana** uyguluyor. Yani rotasyon, LDLQ'nun tile başına faktorize
etmesinin sebebi değil — sebep tile başına farklı sütun kümesi.

**Genişlik.** Maliyet eğrisi 512'de düzleşiyor; 512'den 8'e inmek toplam
tasarrufun %1.9'unu ekliyor ama atılan Hessian bağlantısını 64 katına çıkarıyor.

Ölçüm (`o_proj`, 512 satır, B=1.5), `full`'e karşı, T=4 / T=16 / T=max:

| genişlik | R (rotasyon daraltıldı) | H (geri besleme daraltıldı) |
|---|---|---|
| 2048 | +12.8 / +12.8 / +22.3% | −8.6 / −9.2 / −2.0% |
| 1024 | +25.3 / +13.4 / +23.5% | −7.1 / −16.7 / −6.8% |
| **512** | +43.0 / +38.1 / +44.3% | **−11.1 / −23.4 / −15.8%** |
| 128 | +117 / +69 / +95% | +13.4 / −20.6 / −11.7% |
| 8 | +375 / +169 / +192% | +147 / +49 / +57% |

**`H512` her tile boyutunda bütün ızgaranın en iyi kolu.** `R8` neredeyse
rotasyonsuza eşit (−3.3%): rotasyonu daraltmak mekanizmayı yok ediyor.

Mekanizma iki tarafta da tutarlı. Rotasyonun işi §5.3'te kurulduğu gibi kalın
kuyruğu **olabildiğince geniş** yaymak; 8 koordinat içinde döndürmek o
koordinatların normunu değiştiremez, yalnızca yönünü. Geri besleme tarafında ise
2560×2560'lık alt-Hessian 32,768 token'dan kestiriliyor; uzun menzilli
bağlantılar gürültülü, atmak düzenlileştirme gibi davranıyor.

### 5.8 Ölçek uydurmayı örneklemek ucuz değil — gürültülü

Tavanı M1'i 17 → 8.6 güne indiriyordu. Ölçüldü (`m0_scale_fit.py`, 54 kol);
**alınamaz.** Sebep ortalama bedel değil, **varyans**. Aynı ayar, yalnız hangi
vektörlerin örneklendiği farklı:

| T | tohum 0 | tohum 1 | tohum 2 | aralık |
|---|---|---|---|---|
| 4 | **+17.08%** | +1.49% | +1.25% | **15.8 pp** |
| 16 | −3.70% | −3.14% | −3.38% | 0.6 pp |
| max | −9.97% | +4.58% | −7.48% | **14.6 pp** |

Kapı B'nin ayırmaya çalıştığı komşu tile farkı **0.31 σ**, saptanabilir fark
hata seviyesinin **%3.2'si**. 15 puanlık, tile'dan tile'a bağımsız bir gürültü
kaynağı tam olarak ölçmeye çalıştığımız şeyi boğar.

**Adım sayısını düşürmek de çalışmıyor:** `n6` +45.6 / +17.8 / +13.6%,
`n12` +8.6 / −7.5 / −7.4% — işareti bile tutarsız.

> **Ayrıca kaydetmeye değer bir ders.** Daha önce "6 adım α'yı %1.4 içinde
> buluyor" demiştim; doğruydu ve yanıltıcıydı. **α'daki %1.4, çıktı hatasında
> %45.** Vekil bir ölçü, ölçtüğünü sandığın şey değildir.

**Yan bulgu — `fit_scale` yanlış hedefi optimize ediyor.** T=16 ve T=max'te
örnekleme sistematik olarak **iyileştiriyor** (s256/n12 T=max'te −20.8%). Sebep:
`fit_scale` `‖x − αQ(x/α)‖²`'yi **ağırlık uzayında** minimize ediyor, oysa
hattın hedefi `tr(E H Eᵀ)`. Daha kesin bir α, yanlış ölçüye göre daha kesin.
**Ölçülmedi, ve artık en büyük açık fikir bu.**

### 5.9 VENOM'un `V`'si bizim `T`'miz

V:N:M formülü VENOM'dan dolduruldu ve yapısal bir şey çıktı: VENOM `V` satırın
paylaştığı bir sütun seçimi kullanıyor, yani indeksi `1/V` ile amortize ediyor.
**İndeks amortizasyonu yeni değil.** Katkı "amortize etmek" değil, `(T, d)`
düzlemini bir bütçe altında taramak. Özgünlük iddiası buna göre daraltıldı.

### 5.10 Protokol ayrımı dizi uzunluğuymuş

Ölçümden önce hipotez olarak kaydedildi, ölçümde tuttu. Kural "birini seç"
değil **"pencereyi sabitle"**.

---

## 6. Maliyet: 120 gün → 17 gün

Bu bölüm mühendislik, §5 bilim. Ayrı tutuluyor çünkü buradaki hiçbir şey tezi
değiştirmiyor — yalnızca sınanabilir hâle getiriyor.

### 6.1 Bugünkü tablo (B=1.5, cuda/float32, Triton açık)

| T | d | nokta | codebook | rotasyon | cholesky |
|---|---|---|---|---|---|
| 1 | 0.2500 | 2.35 h | 1.48 | 0.46 | 0.34 |
| 2 | 0.5000 | 5.10 h | 2.97 | 1.72 | 0.34 |
| 4 | 0.6250 | **5.91 h** | 3.71 | 1.92 | 0.21 |
| 8 | 0.6875 | 5.53 h | 4.08 | 1.27 | 0.11 |
| 16 | 0.7188 | 2.11 h | 1.27 | 0.72 | 0.06 |
| 32 | 0.7344 | 1.77 h | 1.29 | 0.38 | 0.03 |
| max | 0.7500 | 0.64 h | 0.57 | 0.00 | 0.00 |

Eval'in kendisi **238 s** — ihmal edilebilir. Maliyet ızgaranın **ortasında**
tepe yapıyor: `n_tiles = n_out/T` düşerken `d` (dolayısıyla `k`) yükseliyor.

**M1 (3 bütçe × 7 tile × 5 çekiliş): 17.4 gün.**
**`τ` süpürmesi: 5.1 gün** (spec 25 *saat* diyordu; 123 saat çıktı).

**Artık hiçbir terim baskın değil.** Codebook hâlâ en büyüğü ama diğer ikisinin
toplamının yalnızca **1.7 katı** — bu oran 5× → 3.6× → 1.7× diye indi. İki kalan
kaldıraç da eşitlendi: ölçek uydurmayı düşürmek 8.8 gün, blok genişliği 8.3 gün.
Sıradaki optimizasyon varsayımla değil **ölçümle** seçilmeli; bu varsayım iki
kez yanlış çıktı.

### 6.2 Kapatılan duvarlar, sırayla

| # | Duvar | Neydi | Nasıl kapandı |
|---|---|---|---|
| 1 | **Bellek** | `T=2`'de 462 GiB tek tensör | `tile_hessian_stream` → 239 MiB. Yığılmış yolla bit-birebir aynı |
| 2 | **Yanlış cihaz** | Hat CPU/float64'te koşuyordu | cuda/float32, uçtan uca **16–45×**, ağırlık farkı 5e-08 |
| 3 | **Kaba kuvvet arama** | Baskın terim | `nearest_e8p` kafesi çözüyor: CPU 3.51×, GPU 1.87× |
| 4 | **Modelin kendi hatası** | Cholesky'yi 9.4× fazla yazıyordu | §6.3. 120 → **94 gün** |
| 5 | **Süpürme %99.6 boşta** | Grup başına 0.248 ms, hesap 0.0034 ms | Tile'lar arası toplu. 94 → **48 gün** |
| 6 | **Tarama hiç gerekmiyordu** | Kesinlik küçük α'da %0.7'ye çöküyor | Analitik arama. 48 → **29 gün** |
| 7 | **GPU %28 meşgul** | Boştanın %80'i çekirdek fırlatma | Triton füzyonu. 29 → **17 gün** |

### 6.3 Maliyet modelinin dört hatası

Üçü iyimser, dördüncüsü **kötümser** — ve o en çok zarar veren oldu, çünkü bu
sayı M1'in koşulup koşulmayacağını söyleyen sayı.

1. **`fit_scale` hiç yoktu** — 6× az. `ldlq_quantize` quantize etmeden önce 24
   aday ölçeği tile'ın tamamı üzerinde tarıyor.
2. **Arama, 16 satırlık sıkı bir döngüde ölçülmüş hızla fiyatlandı** — orada
   codebook önbellekte kalıyor, gerçek çağrılar arasında Hessian güncellemeleri
   onu atıyor. Uçtan uca **tile** süreleri ölçülerek düzeldi.
3. **Her tile boyutuna tek bir ağırlık-başı sabit** — sabit satır sayısıyla üç
   kat düşüyor, yani ızgaranın kaba ucu fazla yazılıyordu. Ve kaba uç, tam da
   granülerlik sorusunun ilgilendiği uç.
4. **Cholesky hızı k=2048'de, ısınmamış benchmark'la ölçülmüş** —
   `cholesky_inverse` ısıtılmıyordu (1.6×), ve tek bir flop/s bu çekirdeği
   tanımlamıyor (k=1024→8192 arası **6.8×** değişiyor, 2.6× daha). Gerçek
   genişliklerde **9.4× fazla**.

Ders: **kernel mikro-benchmark'larından maliyet kurmak burada işlemiyor.** Her
eğri, kodun onu çağıracağı boyutlarda ölçülüyor, ve artık ileri telafi de
modelde (ölçülen %8.6).

### 6.4 Aramayı taramaktan çıkarmak

**65536 kodsözcüğünü taramaya hiç gerek yokmuş.**

Bir kodsözcüğü `σ⊙p + s`: `p` 256 **negatif olmayan** kaynak örüntüsünden biri,
`σ` ilk yedi koordinatta serbest, sekizinci koordinat toplamı çift yapacak
şekilde belirli. `p` negatif olmadığı için sabit `p` altında en iyi işaretler
koordinat koordinat okunuyor (`σ_i = sign(z_i)`); bu atama tek parite ise
geçersiz, ve her koordinat yarım-tamsayı olduğundan **herhangi bir tek işaret
çevirisi pariteyi değiştiriyor** — yani onarım tek ve en ucuz çeviri, bedeli
`2|z_i|p_i`. 128 işaret seçimi bir arama uzayı değil, aritmetik.

Bu, taramanın yerine değil **geri düşme yolunun** yerine kondu. Kafes çözücü bir
satırı çözebildiğinde hâlâ daha ucuz (8K ve 80K satırda aynı 0.2 ms — fırlatma
bağımlı). Düzelttiği şey `fit_scale`'in küçük-α adımları:

| f | kesinlik | geri düşen | adım |
|---|---|---|---|
| 0.40 | **%0.7** | 5,845 / 5,888 | 30.0 ms |
| 1.03 | %63.7 | 2,136 | 12.5 ms |
| 2.00 | %99.9 | 8 | 2.0 ms |

**Fit'in %88'i oradaydı.** Kazanç: `fit_scale` 3.25× (5,888 vektör) → **10.8×**
(196,608 vektör); tile başına 1.35 / 2.65 / 5.62×; gerçek katmanda 1.29 / 2.17 /
3.96×.

**Kesinlik.** float64'te bir milyon vektörde **sıfır** uyuşmazlık; her
kodsözcüğü kendine çözülüyor. float32'de milyonda bir satır farklı seçiliyor ve
o satırlar gerçek berabere — mesafe farkı 3e-6. İddia "kesin", "float32'de
bit-birebir" değil, ve test hangisi olduğunu söylüyor.

### 6.5 Triton: Windows'ta var

GPU **%28.4** meşguldü. Bir chunk'ta 414,841 çekirdek çağrısı, ölçülen fırlatma
maliyeti **10.1 µs**, yani boşta geçen 5,258 ms'nin **4,190 ms'si** doğrudan
fırlatma.

Upstream `triton` Windows tekerleği yayımlamıyor — `has_triton()` bu yüzden
False'tu ve ilk deneme 15 dakika asıldı. Ama **`triton-windows` PyPI'da** ve
sürümler tutuyor: torch 2.12 → triton 3.7.0 → `triton-windows==3.7.0.post26`.

İki elementwise zincir ayrı saf fonksiyonlara çıkarıldı ve `dynamic=True` ile
derlendi (`_analytic_shift`, `_lattice_shift`):

| ölçek | kazanç |
|---|---|
| `_nearest_halfinteger_even` | 2.30× |
| analitik aramanın gövdesi | 5.96–6.62× |
| **tile başına uçtan uca** | **1.64× / 1.72× / 1.87×** |

**Tahminim 3.5–5× idi, gerçekleşen 1.7×.** Fazla iyimserdim: hesabım bütün
fırlatma yükünün gideceğini varsayıyordu, oysa yalnız iki blok derlendi ve LDLQ
süpürmesinin küçük işlemleri hâlâ eager.

İki ayrıntı taşıyıcı:
- **`dynamic=True` şart.** Satır sayısı, kafes çözücünün çözemediği satır
  sayısı — her çağrıda değişiyor. Statik derlense her yeni şekilde birkaç saniye
  harcardı; dinamik, bir kez derleyip 64 kat aralıkta **sıfır yeniden derleme**.
- **Derleme bir sondayla zorlanıyor.** Inductor tembel; bırakılsa hata katmanın
  ortasında patlardı. Burada CUDA Triton'la derleniyor ama **CPU `cl` (MSVC)
  istiyor ve bulamıyor**, o yüzden cihaz/dtype başına sondalanıyor.

**Çıktı birebir aynı, ve bu teste bağlandı.** Tören değil: çekirdek,
toolchain'i olmayan yerde eager'a düşüyor; iki yol farklı sonuç verseydi işi
hangi makinenin koştuğu modeli değiştirirdi ve başka hiçbir test bunu
yakalamazdı.

> **Uyarı:** `TILE_TIMINGS` artık **Triton'lu bir makineyi** tanımlıyor.
> Triton'suz makinede cevaplar aynı, saat ~1.7× yavaş — model o kadar iyimser
> olur. `TILESPARSE_NO_COMPILE=1` derlemeyi kapatır.

### 6.6 Izgara seçenekleri

Bağlayıcı kısıtlar: **`min_seeds=5`** (Kapı B'nin verdikti için 08-21'de
ölçülerek donduruldu) ve **`T ∈ {1,2,4,8,16,32,max}`** (ön-kayıt `{1,16,max}`
üçlüsünü *açıkça* reddediyor — "yanlış-durdurma taşırdı"; tile eksenini budamak
tezin kendi eksenini budamak). Bağlı **olmayan**: 5 çekilişin kaç bütçede
koşacağı.

| tasarım | süre | +fp16 |
|---|---|---|
| A. Tam M1 (3 bütçe × 7 tile × 5 çekiliş) | 17.4 g | 12.0 g |
| **C. B=1.5'te 5 çekiliş, iç tile'lar 2, diğer bütçeler 1** | **5.8 g** | 4.0 g |
| D. Tek bütçe, 5 çekiliş, 7 tile | 4.9 g | 3.4 g |
| F. Tek bütçe, 1 çekiliş, 7 tile — ilk gerçek U eğrisi | **23 saat** | 16 saat |
| G. Yalnız iki uç (T=1, T=max), 5 çekiliş | 0.6 g | — |

---

## 7. Denenip **reddedilenler** — tekrar denenmesin

Bu bölüm kasıtlı olarak uzun. Bir fikrin denenip elendiğini kaydetmemek, onu
ikinci kez denemek demek.

### 7.1 Bilimsel gerekçeyle reddedilenler

| fikir | ölçülen | neden reddedildi |
|---|---|---|
| **Ölçek örneklemesi** (`fit_scale(sample=N)`) | ortalama küçük, **tohum aralığı 15.8 pp** | Kapı B 0.31 σ'lık farkları ayırıyor; bu gürültü onu boğar (§5.8) |
| **Adım azaltma** (`n_steps` 24→6) | +45.6 / +17.8 / +13.6% | Kaliteyi doğrudan bozuyor, ve `n12`'de işaret bile tutarsız |
| **Katman-başı ölçek** (`per_layer`) | %11, yeniden ölçümde T=4'te **+87.9%** | Küçük `T`'de tile'ların ölçekleri gerçekten farklı |
| **Rotasyonu daraltmak** (blok-köşegen RHT) | `R8` ≈ rotasyonsuz (−3.3%) | Rotasyonun işi kalın kuyruğu **geniş** yaymak; 8 koordinat içinde norm değişmez |
| **TF32** | Hessian 1.74×, rotasyon 1.66×, **arama 1.04×** | Mantisi 10 bite düşürüyor ve Hessian LDLQ'nun girdisi — kalite ölçülmeden alınmaz |
| **fp16 arama** | 1.3–1.7×, bedel ≤%1 | Reddedilmedi, **varsayılan kapalı**: kaliteyi ölçülebilir biçimde değiştirdiği için bir karar gerektirir |

### 7.2 Mühendislik gerekçesiyle elenenler

| fikir | ölçülen | neden |
|---|---|---|
| **`torch.compile` (ilk deneme)** | 15 dk asıldı | Triton yoktu. **Sonra çözüldü** — `triton-windows` (§6.5) |
| **Inductor CPU yolu** | `Compiler: cl is not found` | MSVC yok; cihaz başına eager'a düşülüyor |
| **Elementwise'ı elle azaltmak** (18→11 işlem) | 0.94–1.10× | Çözücü **fırlatma** bağımlı, işlem sayısı bağımlı değil: 8K ve 80K satır aynı süre |
| **Mesafe matrisini maddileştirmemek** | 1.00× | Hesap/bant dengeli; kazanç yok |
| **Yalnız kafes alt sınırıyla dal-sınır** | 1.03–1.10× | Kafes sonsuz, küçük α'da sınır zayıf — tam da pahalı uçta |
| **Birleşik alt sınırla dal-sınır** | 1.21–1.77×, kesin | Reddedilmedi, **alınmadı**: analitik arama (§6.4) getirisini büyük ölçüde sildi |
| **Fit'i tile'lar arasında toplamak** | 2.16× | Alınmadı: bit-birebir **değil** (indirgeme sırası), ve §6.4'ten sonra getirisi küçüldü |
| **Blok başına CPU↔GPU aktarımı** | nokta başına ~220 s | Saatlere karşı ihmal edilebilir |
| **İki kaydırmayı tek çözümde yığmak** | 1.9–2.2× | Alınmadı: Triton füzyonu aynı kazancı zaten topluyor |

### 7.3 Bilinçli olarak yapılmayanlar

| ne | tarih | neden |
|---|---|---|
| **E8P'nin ucuz doğrulama deneyi** | 08-20 | Kullanıcı kararı. Risk §3.2'de açık varsayım olarak taşınıyor — bu projenin en büyük tek riski |
| **Kernel yazmak** | — | Spec §8 kapsam dışı bırakıyor. Roofline alt sınır olarak sunulur, hız iddiası yapılmaz |
| **AQLM-survivor** | 08-20 | VQ ailesinin en zayıf ve en pahalı üyesi; codebook yeniden kalibrasyonu katman başına saatler |
| **Izgarayı daraltmak** | 08-24 | Artık gerekmiyor (17 gün). Gerekseydi bile tile eksenine dokunulmazdı |

---

## 8. Sırada ne var

### 8.1 Bir sonraki oturumun ilk işi — kritik yol

**`experiments/m1_run.py` yaz: tam model sürücüsü + checkpoint.**

Şu an yok, ve "sıkıştırılmış ppl hiç ölçülmedi" durumunun sebebi bu.
`calibrate.sequential_calibrate` (mevcut, `compress_fn` alıyor) ile
`eval.streamed.streamed_perplexity` (mevcut) arasını bağlayacak; `compress_fn`
içinde `m1_gates.run_config`'in hattı çağrılacak. `hf_llama.load_llama` ve
`capture_block_inputs` hazır.

**Kesintiye dayanıklılık bunun parçası, sonradan eklenen bir şey değil.**
23 saatlik bir koşu dizüstünde kesilir. Kayıt birimi blok (nokta başına 32);
anahtar `(model, budget, tile, draw, block)`.

> **Dikkat:** `sequential_calibrate` Hessian'ı **sıkıştırılmış** modelden okuyor
> (Spec v6 tuzak 20). Devam ederken bir sonraki bloğun **girdileri de** kayıttan
> gelmeli — yoksa devam eden koşu kesilmeyenden farklı sonuç verir. Doğrulaması
> net: kesip devam ettirilen koşu, kesilmeden koşulanla aynı ppl vermeli
> (`tests/test_hf_llama.py`'nin tiny Llama'sı ile test edilebilir).

### 8.2 Sonra: ilk gerçek koşu

**Tasarım F** — tek bütçe, tek çekiliş, 7 tile, **23 saat**. Hattın gerçek
modelde uçtan uca çalıştığını kanıtlar ve **ilk gerçek U eğrisini** verir.

Kapı B'yi karara bağlamaz (§5.6: verdikt için ≥5 çekiliş, `gate_b` altında
"undetermined" döner). Ama Kapı A için ve eğrinin şeklini görmek için yeterli.

### 8.3 Ön-kaydı dondurmak — artık maliyet engeli yok

İki kutu açık: **`Δ(T)` tahmin eğrisi** ve **`T*_tahmin`**. İkisi de `τ`
süpürmesine bağlı, ve süpürme 29 gündü. **Artık 5.1 gün** (123 saat,
`m0_cost_model.sweep_cost`) — yani maliyet artık engel değil.

> **Ama süpürme betiği de yok.** Bu belgenin önceki sürümleri `tau_sweep.py`'ye
> mevcut bir şeymiş gibi atıfta bulunuyordu; değil. Modellenen **maliyeti**,
> yazılmış olan **kodu** değil. §8.1 ile aynı durum: engel bilimsel değil,
> yazılmamış betik.

İki betik de aynı dikiş yerini kullanacak (`sequential_calibrate` +
`streamed_perplexity`), o yüzden §8.1'i yazmak §8.3'ün yarısını da yazmış olur.

Sıra önemli: ön-kayıt donmadan M1 başlamaz. Ama F koşusu ön-kayda girmiyor
(tek çekiliş, kapıları karara bağlamıyor), o yüzden paralel gidebilir.

### 8.4 Ölçülmemiş kalan en büyük fikir

**`fit_scale`'i doğru hedefe uydurmak** (§5.8). Şu an ağırlık uzayında
`‖x − αQ(x/α)‖²` minimize ediliyor, oysa hattın hedefi `tr(E H Eᵀ)`.
Örneklemenin T=16 ve T=max'te kaliteyi **iyileştirmesi** bunun belirtisi.
Doğru hedefe uydurmak hem daha iyi hem daha ucuz olabilir.

### 8.5 Açık kalan kod işleri

- **Axis A için LDLQ** — şu an `NotImplementedError`; Axis B'de indeks ekseni
  girdi kanalları olduğu için Hessian doğrudan uygulanıyor, Axis A'da sweep
  tile'ın sütunları boyunca olmalı
- **Blockwise (tam SparseGPT) maske seçimi** — M3 teslimatı, şu an `upfront`
- **§3.6'nın üç ablasyonu** — grup konvansiyonu (iki koşullu olmalı),
  quantization/maske hatası ayrımı, hizalama
- **Attention koordinasyonu formülü** — `v_proj`↔`o_proj`, GQA, RoPE çiftleri;
  `T=max` için sert kısıt, hâlâ yalnızca ima edilmiş
- **Eval maliyeti** — 238 s yalnız WikiText-2; C4 ve 5 zero-shot görev hiç
  ölçülmedi ve ön-kayıt §4 ikisini de şart koşuyor

---

## 9. Açık riskler

**Kapı A'nın düşme olasılığı yüksek.** Prova (`gate_a_dry_run.md`) GPTQ-4bit
survivor'larla her satırın düştüğünü gösterdi. E8P aritmetiği değiştiriyor ama
**gösterilmedi**. Karar tablosunun `✗/✓` dalı hazır: proje durmaz, çerçeve
daralır.

**E8P kalite varsayımı** (§3.2). Projenin en büyük tek riski. Düşerse bant
1.83–2.83'e kayar ve tezin "2 bitin altı" motivasyonu zayıflar.

**`T*`'ın belirsizliği.** Verdikt tarafı çözüldü; eğri iç bölgede düzse küme
büyük çıkar ve *hangi* granülerlik sorusu cevapsız kalır. Başarısızlık değil
ama manşeti zayıflatır.

**Sentetik σ.** Kapı B'nin gücü ve transfer toleransı sentetik katmandan
ölçüldü. Gerçeği ilk M1 bütçesinden gelecek; ön-kayıt §7.4'ün uyarlanabilir
kontrolü bunun için var.

**Maliyet artık birincil risk değil ama sıfır da değil.** 17 gün hâlâ uzun, ve
bütün süreler **bu makineye ve Triton'lu bir kuruluma** ait. Başka donanımda
eğriler yeniden ölçülmeli. Model dört kez yanıldı (§6.3).

**Kesinti.** 23 saatlik bir koşu bile dizüstünde kesilir ve şu an devam etme
yok. §8.1'in checkpoint'i bu yüzden kritik yolda.

---

## 10. Ortam tuzakları — saatlere mal oldu, tekrar etmesin

| Sorun | Çözüm |
|---|---|
| **HF indirmeleri takılıyor** (0 B/s) | `HF_HUB_DISABLE_XET=1` |
| **Kimliksiz HF istekleri sert kısıtlanıyor** | `hf auth login` (diske yazar, her süreç görür). `$env:HF_TOKEN` yalnız o pencerede geçerli |
| **`snapshot_download` oturumlar arası devam ETMİYOR** | Bir kez başlat, kesme |
| **Arka plan görev bildirimleri güvenilmez** | Wrapper çıkışı işin bitişi değil. Log dosyasına veya süreç listesine bak |
| **`torchvision` ABI uyumsuzluğu transformers'ı komple kırıyor** | torch'u yükseltirken eşleştir, ya da kaldır |
| **`load_dataset("wikitext", ...)` reddediliyor** | `Salesforce/wikitext` — `namespace/name` gerekiyor |
| **Süreç sayarken kendi ölçüm sürecini sayma** | PowerShell filtresini `python -c` içinden çağırınca kendini yakalıyor |
| **`_on_device` önbelleği cihaz DİZESİYLE anahtarlı** | `"cuda"` ile `"cuda:0"` **farklı nesneler** döndürüyor, ve hızlı yol bir `is` kontrolüyle seçiliyor. Hat `str(tensor.device)` kullandığı için doğru yolda, ama **kıyaslama yazarken aynısını kullan** — yoksa sessizce yavaş yol ölçülür. Bu oturumda birkaç ölçümü gerçekten yanılttı |
| **Python stdout tamponu arka plan koşularında** | `python -u` |
| **`torch.compile` Windows'ta çalışmıyor sanılıyordu** | `pip install triton-windows==3.7.0.post26` (torch 2.12 → triton 3.7.0). Sonra `has_triton()` True |
| **Inductor CPU'da `cl` (MSVC) istiyor** | CUDA derleniyor, CPU derlenemiyor. `quantize._shift_kernel` cihaz/dtype başına sondalıyor ve eager'a düşüyor — sessiz, çünkü iki yol birebir aynı |
| **`TILESPARSE_NO_COMPILE=1`** | Derlemeyi kapatır; derli/derlisiz karşılaştırma ve toolchain sorunları için |

**Donanım:** RTX 5060 Laptop, 8 GB VRAM, sm_120 (Blackwell → cu128+;
`torch 2.12.0+cu130` kurulu). 23.7 GiB RAM, 16 torch thread'i.
7B fp16 (13.5 GB) GPU'ya sığmıyor → **katman-akışlı zorunlu**, ~2.8 GB tepe.

---

## 11. Repo haritası ve çalıştırma

| Modül | İş |
|---|---|
| `accounting.py` | bit bütçeleri, `1−1/T`, `B*`, canlı bant, V:N:M, `rotation_side_bits` |
| `scoring.py` | saliency — iki ağırlık-başı metrik, iki toplama yönü |
| `tiling.py` | tile bölümlemesi, dondurulmuş maske, `align` |
| `prune.py` | maske seçimi + ileriye telafi; **H1 assert'i burada** |
| `compact.py` | survivor'ları tile başına yoğun bloklara topla |
| `rotation.py` | maske-koruyan rotasyon, blok-köşegen varyant |
| `quantize.py` | E8P codebook, kafes çözücü, **analitik arama**, LDLQ, ölçek politikası, füzyon çekirdekleri |
| `calibrate.py` | sıralı kalibrasyon, `LayerProblem` (**dikiş yeri**), `sequential_calibrate` (**sürücü — henüz yalnız testlerden çağrılıyor**) |
| `hf_llama.py` | HF adaptörü — blok 0 girdilerini yakalar; `to_device` |
| `eval/perplexity.py` | ppl + protokol koruması + yayımlanmış sayı tablosu |
| `eval/streamed.py` | katman-akışlı ppl |
| `experiments/m1_gates.py` | M1'in iki kapısı, `t_star_set`, çekiliş ekseni, `HESSIAN_BLOCK` |
| `experiments/m0_dense_ppl.py` | dense ölçüm + protokol kimliği |
| `experiments/m0_vq_bits.py` | VQ checkpoint maliyeti — manifest'ten, indirmeden |
| `experiments/m0_gate_b_power.py` | Kapı B'nin gücü + hattın gürültüsü |
| `experiments/m0_transfer_pilot.py` | `Δ = Q + τ` transfer sapması → tolerans |
| `experiments/m0_cost_model.py` | ölçülen eğrilerden gerçek koşu maliyeti |
| `experiments/m0_rotation_value.py` | rotasyon gerçek katmanda kazandırıyor mu; blok genişliği süpürmesi |
| `experiments/m0_scale_fit.py` | ölçek uydurmayı ucuzlatmanın kalite bedeli |

**Belgeler:** `docs/spec_v7.md` (şartname) · `preregistration.md` (M1 ön-kaydı,
**dondurulmadı** — iki kutu kaldı, artık maliyet engeli yok) ·
`docs/audit.md` (v6 denetimi) · `docs/gate_a_dry_run.md` (literatür provası) ·
bu belge.

**Henüz yazılmamış betikler** (§8.1, §8.3): `experiments/m1_run.py` —
tam model sürücüsü + checkpoint · `experiments/tau_sweep.py` — `τ` yüzeyi.
İkisi de `sequential_calibrate` + `streamed_perplexity` dikişini kullanacak.

```bash
python -m pytest tests/ -q                         # 525 test, ~105 s
HF_HUB_DISABLE_XET=1 python experiments/m0_dense_ppl.py --seqlens 2048 4096 --device cuda
HF_HUB_DISABLE_XET=1 python -u experiments/m0_rotation_value.py \
    --tiles 4 16 max --seqs 16 --rows 512 --solve-device cuda --solve-dtype float32
    # ~30 dk, 51 kol.  --families H ile yalnız kazanan aile
HF_HUB_DISABLE_XET=1 python -u experiments/m0_scale_fit.py \
    --tiles 4 16 max --rows 512                    # ~20 dk, 54 kol
python experiments/m0_cost_model.py                # ~2 dk, sabitler önbelleklenir
python experiments/m1_gates.py --synthetic --n-out 64 --n-in 128 --budgets 1.5 --draws 5
python experiments/m0_gate_b_power.py --no-noise   # ~15 dk, σ önbellekten
python experiments/m0_transfer_pilot.py --draws 3  # ~8 dk; --reuse ile saniyeler
python experiments/m0_vq_bits.py --all             # ~100 KB ağ, saniyeler
```

---

## 12. Commit geçmişi — ne anlama geliyorlar

| Commit | Ne getirdi |
|---|---|
| `5d7726d` | Hattın tamamı: muhasebeden ppl'e |
| `f94a8af` | Ön-kayıt taslağı; dondurma listesi görünür bir olay olsun diye |
| `6af48d2` | v6 denetimi repoya taşındı — kararların gerekçesi versiyonlansın |
| `94dbdce` | Spec v7: kafes VQ etrafında yeniden kuruldu, 4 aritmetik hata düzeldi |
| `c3c5632` | V:N:M formülü VENOM'dan; **özgünlük iddiasını daralttı** |
| `1e6218f` | HF adaptörü |
| `33d66a4` | Katman-akışlı eval — 7B'yi 8 GB'da ölçmenin yolu |
| `d80ab14` | **İlk gerçek ölçüm**: protokol sorusu çözüldü |
| `a1626c6` | VQ maliyeti checkpoint'ten ölçüldü; SU/SV ayrışması bulundu |
| `3d8658f` | Kapı B'nin gücü ölçüldü; `T*` küme oldu; çekiliş ekseni düzeldi |
| `7d1ee48` | Transfer pilotu: tolerans kuralı, ve modelin büyük `T` önyargısı |
| `797aa2e` | Maliyet modeli — hattın gerçek boyutta koşamadığının tespiti |
| `baa38a7` | Bellek duvarı kapandı, iki yükleyici hatası düzeldi, `fit_scale` modele girdi |
| `31f9761` | **Rotasyon gerçek katmanda −70%**; hat GPU'ya taşındı |
| `0201f93` | E8 kafes çözücü: CPU 3.5×, GPU 1.9×, çıktı birebir aynı |
| `f425880` | Bu belge, bilinenin etrafında yeniden yazıldı |
| `f00fe9c` | Blok genişliği: **geri besleme daraltılır, rotasyon daraltılmaz**; maliyet modelinin Cholesky eğrisi düzeldi (120 → 94 gün) |
| `40c8d9c` | Süpürme tile'lar arasında toplu — bit-birebir aynı, 94 → 48 gün |
| `7da170c` | Ölçek örneklemesi ölçüldü ve **reddedildi**; fp16 eklendi (kapalı) |
| `a33839b` | **Analitik en-yakın-kodsözcüğü**: arama çözülüyor, taranmıyor. 48 → 29 gün |
| `1a27ead` | Analitik aramanın parça boyutu genişletildi (fırlatma bağımlı) |
| `cc3e0f4` | **Triton kuruldu**, iki elementwise zincir füzyonlandı. 29 → 17 gün |

---

## 13. Çalışma tarzına dair not

Bu projede en pahalı hata sınıfı **sessizce yanlış bir sayı üretmek**. Bu yüzden:

- Golden sabitler elle yazılmaz, türetilir. `tests/golden.py` `accounting.py`'yi
  **import etmez** — golden değerleri çağıran bir test hiçbir şey kanıtlamaz
- Testlerin çoğu davranış değil **iddia** sınıyor
- Doğrulanmamış şeyler açıkça "varsayım" diye işaretlenir
- Bir hipotez ölçümden **önce** yazılır
- Hız iddiaları **uçtan uca** ölçülür, çekirdek mikro-benchmark'ıyla değil —
  maliyet modeli tam olarak bu yüzden dört kez yanıldı
- Bir optimizasyonun kabul kriteri **çıktının değişmemesi**; değişiyorsa
  değişimin ne olduğu ölçülür ve karar tablosuna yazılır
- Aleyhe bulgular da kaydedilir (eşleştirme kazancı 1.16×, ayrılabilirlik
  önyargısı, maliyet modelinin dört hatası, Triton tahminimin 2–3 kat iyimser
  çıkması)
- **Denenip elenen fikirler kaydedilir** (§7) — kaydetmemek ikinci kez denemek

Bu belge de aynı disiplinin parçası: ne bilindiğini, ne bilinmediğini ve neyin
denenip bırakıldığını ayrı tutuyor.
