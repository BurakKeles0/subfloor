# Durum ve Devir Belgesi

> **Bağlam kaybolduğunda projeye kaldığı yerden devam edebilmek için var.**
> Kod ne yaptığını söyler; bu belge **neden öyle olduğunu** söyler.
> Son güncelleme: 2026-08-25 · HEAD `96f973b` · Testler: **619 geçiyor, 6 atlanıyor**
> Bu oturumun ölçüm dersleri **§14**'te — hız kazançlarından daha taşınabilir.

---

## 1. Nerede duruyoruz — beş cümle

Hat uçtan uca çalışıyor, gerçek Llama-2-7B'ye bağlı, ve gerçek ağırlıklar
üzerinde üç ölçüm var: dense perplexity (yayımlanmıştan 0.006 içinde),
rotasyonun katman değeri (**−70%**), ve blok genişliğinin etkisi. M0'ın
uçuş-öncesi kalemleri kapandı. **Maliyet artık bağlayıcı kısıt değil:** M1 bu
makinede 120 günden **14.9 güne**, `τ` süpürmesi 29 günden **5.5 güne** indi — yani
ön-kaydı bloke eden şey ortadan kalktı. Sayının bir kez **yukarı** gittiğine
dikkat: 12'ye inmişti, sonra modelin hiç yazmadığı iki terim bulununca gerçek
maliyetin ~40 olduğu anlaşıldı, ve 15'e o terimler düzeltilerek inildi (§6.2). Ama **sıkıştırılmış modelin
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
| **Kronecker kongrüansının kalite bedeli (gerçek katman)** | Aynı kurulum, `full` kolu birinci turu 7.0e-07 ile üretiyor. Hattın kolunda −0.03…−0.31%, tam genişlik geri beslemede +0.5…+1.9% (§6.8) |
| **Blok genişliğinin etkisi** | 51 kol, 5 genişlik × 3 aile × 3 tile (§5.7) |
| **Kafes çözücünün taramaya denkliği** | `nearest_e8p` ile kaba kuvvet, dört ölçekte birebir aynı indeks |
| **Analitik aramanın taramaya denkliği** | float64'te bir milyon vektörde **sıfır** uyuşmazlık; 65,536 kodsözcüğünün hepsi kendine çözülüyor (§6.4) |
| **Derlenmiş ile eager'ın denkliği** | Üç ölçek × üç satır sayısı (düzensiz dâhil), `torch.equal` |
| **Toplu süpürmenin tile-tile'a denkliği** | İki cihaz, iki dtype, iki ölçek politikası, chunk 2'den tile sayısının 4 katına |
| **Akıtılan alt-Hessian'ın yığılmışa denkliği** | Bit-birebir aynı çıktı |
| **Maliyet modelinin kendini doğrulaması** | Hiç uydurulmadığı bir genişlikte (k=7912) gerçek tile süresini 16 satırda %11.9 içinde tahmin ediyor. 4 satırda %39.8 sapıyor — **fazla** yazarak, yani 12 gün bir üst sınır (§6.3) |
| **§8.1 dikişinin GPU'da uçtan uca koştuğu** | Gerçek Llama blokları → `sequential_calibrate` → `run_config` → sıkıştırılmış ağırlıklar → `streamed_perplexity`, hepsi cuda'da. 08-25'e kadar **koşmuyordu** ve bunu hiçbir test görmüyordu (§6.12) |

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

Sebebi artık maliyet değil (bir U eğrisi **19.9 saat**): **tam modeli sıkıştıran
betik yok.** `calibrate.sequential_calibrate` kütüphane olarak var ama yalnızca
testlerden çağrılıyor. Aynı şey `τ` süpürmesi için de geçerli — maliyeti
modellenmiş, kodu yazılmamış (§8.3).

Ayrıca hiç ölçülmemiş: **eval'in gerçek maliyeti** (238 s yalnız WikiText-2;
ön-kayıt §4 C4'ü de şart koşuyor ve 5 zero-shot görev istiyor) ve
**`fit_scale`'in doğru hedefe uydurulması**. *(TF32 bu listedeydi; 08-24'te
ölçüldü ve reddedildi — §6.9.)*

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
| **Kronecker kongrüansı eklendi, varsayılan kapalı** | 08-24 | Gerçek katmanda `H512` kolunda −0.03…−0.31% (lehte), rotasyon terimi **5.52×**. Bit-birebir olmadığı için açmak ayrı bir karar (§6.8, §8.5) |
| **fp16 arama ve telafi bloklaması da kapalı kaldı** | 08-24 | Kullanıcı kararı: şimdiye kadarki her kalite sayısı üçü de kapalıyken alındı. Üçü birlikte M1 14.9 → 7.5 g (§8.5) |
| **TF32 kapandı — kalite yüzdesiyle değil** | 08-24 | Hattı kırıyor: döndürülmüş alt-Hessian Cholesky'den geçmiyor, sönümleme payının %85'i gidiyor. Çalıştığı yerde de %3.2'yi aşan tek kol (§6.9) |
| **Analitik en-yakın-kodsözcüğü** taramanın yerine | 08-23 | Kodsözcüğü uzayının yapısı aramayı çözüyor. Uçtan uca 1.3–4.0×, float64'te kesin (§6.4) |
| **Triton kuruldu, iki zincir füzyonlandı** | 08-24 | GPU %28.4 meşguldü; boşta geçenin %80'i fırlatma. Uçtan uca 1.64–1.87×, çıktı birebir aynı (§6.5) |
| Maliyet modelinin varsayılanı **hattın koştuğu konfigürasyon** | 08-24 | Kimsenin koşmadığı bir konfigürasyonu fiyatlamak, iyimser bir sabitle aynı hata — yalnız ters yöne bakıyor |
| **Ölçek adayları tek aramada toplandı** | 08-24 | Arama fırlatma bağımlı: 1,280 vektör 41.3 ms, 5,888 vektör 43.4 ms. 24 ayrı geçiş sabit bedeli 24 kez ödüyordu. Tile başına 3.78×/2.01×/1.09×, çıktı **bit-birebir** (§6.7) |

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

## 6. Maliyet: 120 gün → 14.9 gün — ve bir kez yukarı gitti

Bu bölüm mühendislik, §5 bilim. Ayrı tutuluyor çünkü buradaki hiçbir şey tezi
değiştirmiyor — yalnızca sınanabilir hâle getiriyor.

### 6.1 Bugünkü tablo (B=1.5, cuda/float32, Triton açık)

| T | d | **nokta** | codebook | rotasyon | telafi | kalib | chol | eval |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.2500 | **4.42 h** | **2.86** | 0.46 | 0.36 | 0.33 | 0.34 | 0.07 |
| 2 | 0.5000 | **4.41 h** | 1.59 | **1.72** | 0.36 | 0.33 | 0.34 | 0.07 |
| 4 | 0.6250 | 3.77 h | 0.88 | **1.92** | 0.36 | 0.33 | 0.21 | 0.07 |
| 8 | 0.6875 | 2.81 h | 0.66 | 1.27 | 0.36 | 0.33 | 0.11 | 0.07 |
| 16 | 0.7188 | 1.99 h | 0.46 | 0.72 | 0.36 | 0.33 | 0.06 | 0.07 |
| 32 | 0.7344 | 1.55 h | 0.38 | 0.38 | 0.36 | 0.33 | 0.03 | 0.07 |
| max | 0.7500 | 0.97 h | 0.21 | 0.00 | 0.36 | 0.33 | 0.00 | 0.07 |

Sağdaki üç sütun **tile boyutundan bağımsız** — aynı 32 bloğu aynı şekilde
dolaşıyorlar. Bu, tasarım ekonomisini yeniden sıralıyor: sıkıştırma baskınken
maliyeti *hangi* tile'ları koştuğun belirliyordu, düz nokta-başı terimler
varken **kaç nokta** koştuğun belirliyor (§6.10).

Toplam paylar (7 tile): codebook %35.3, rotasyon %32.5, **telafi %12.7**,
kalibrasyon %11.6, cholesky %5.5, eval %2.3.

> **Tepe 08-25'te yer değiştirdi ve bu tablodaki en önemli değişiklik.** Bu
> bölüm "maliyet ızgaranın **ortasında** tepe yapıyor" diyordu; yapmıyor.
> `TILE_TIMINGS` 4 satırın altında hiç örnek taşımıyordu, yani T=1 ve T=2
> 4-satır oranını **ödünç alıyordu** — ve ağırlık başına maliyet 1 satırdan
> 4096'ya **41 kat**, ilk adımda tek başına 8 kat düşüyor. Gerçek eğri T=1'den
> itibaren **monoton azalıyor**, ve pahalı uç ince uç (§6.14).

**M1 (3 bütçe × 7 tile × 5 çekiliş): 14.9 gün.**
**`τ` süpürmesi: 5.5 gün** (spec 25 *saat* diyordu).
**Tasarım F (ilk gerçek U eğrisi): 19.9 saat.**

> **Baskın terim tile'a göre değişiyor, ve tek bir cevabı yok.** T=2 ve T=4'te
> en büyük kalem **rotasyon** (1.72h ve 1.92h). **T=1'de codebook** (2.86h,
> diğer ikisinin 3.6 katı) — orada duvar `fit_scale`'in tile başına sabit
> bedeli: tek satırlık bir tile ona amortize edecek 128 vektör veriyor, T=4
> 1280. Ve T=1 tezin kıyas grubu olan yapısız taban.

**Ölçek kaldıracı da çöktü.** Per-tile fit'i tamamen atmak 8.8 gün değil
**1.4 gün** kazandırıyor — çünkü fit artık tile'ın %28'i, %83'ü değil. Bu,
`per_layer`'ın ve örneklemenin *maliyet* gerekçesini de siliyor; ikisi de zaten
kalite gerekçesiyle reddedilmişti (§5.8, §7.1). Geriye büyük ve kaliteye mal
olan bir kaldıraç kalmadı.

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
| 8 | **Fit sabit bedeli 24 kez ödüyordu** | 1,280 vektör 41.3 ms, 5,888 vektör 43.4 ms | Adaylar tek aramada. 17 → **12 gün** (§6.7), iki iyimser ölçüm geri çekilerek |

**Ve burada sayı yukarı gitti.** Dokuzuncu duvar bir hızlanma değil, modelin
ölçmediği bir şeydi — ve bulunduğunda M1'in gerçekte 12 değil ~40 gün olduğu
ortaya çıktı:

| # | Duvar | Neydi | Nasıl kapandı |
|---|---|---|---|
| 9 | **Kalibrasyon hiç yazılmamış** | Nokta başına 5.6 saat, sıkıştırmanın tamamından pahalı | Modele yazıldı: 12 → **~40 gün** (§6.10). Sonra Hessian GPU'da biriktirildi (25×): → **13.4 gün** |
| 10 | **İleri telafi de yazılmamış** | Nokta başına 0.36 saat, tile boyutundan bağımsız | Modele yazıldı: → **15.0 gün** (§6.11b). Bloklanabilir ama bit-birebir değil, varsayılan kapalı |
| 11 | **`_nearest`'in ikinci kapısı** | 21 hücrenin 10'u 65,536 kodsözcüğü tarıyordu | 384–1024 aralığı analitik yola açıldı. Süpürmede **2.0–3.5×** (§6.11a) |
| 12 | **Üç sabitin yazılmamış eşitsizliği** | Süpürmenin her grubu taranıyordu; 21 hücrenin 8'i | İki sabit ölçülerek oynadı. Tile'da **1.25×**, kalite bit-birebir (§6.13) |
| 13 | **`TILE_TIMINGS`'te 4 satırın altı yok** | T=1 ve T=2 4-satır oranını ödünç alıyordu | `n_tiles` kaydedilerek yeniden ölçüldü: 15.0 → **14.9 gün**, ama **tepe ortadan ince uca kaydı** (§6.14) |

> **9, 10 ve 13 hızlanma değil, düzeltme.** İlk ikisi M1'i *pahalılaştırdı* çünkü
> zaten öyleydi; 13 toplamı neredeyse hiç oynatmadı ama **eğrinin şeklini**
> değiştirdi — tepe ortadan T=1'e kaydı, ve T=1 tezin kıyas grubu. Bu tablodaki
> en değerli satırlar onlar: 1–8 ve 11–12 kodu hızlandırdı, 9, 10 ve 13
> **sayıyı doğru yaptı**.
>
> Ve üçü de aranarak değil, **başka bir şey düzeltilirken** çıktı: ikisi eksik
> terim arayışında, biri provenance düzeltilirken.

### 6.3 Maliyet modelinin sekiz hatası

İlk üçü iyimser, dördüncüsü **kötümser** — ve o en çok zarar veren oldu, çünkü
bu sayı M1'in koşulup koşulmayacağını söyleyen sayı. Beşincisi yine iyimser.
Altıncı, yedinci ve sekizinci (§6.10, §6.11b, §6.14) **modelin bilmediği
şeyler** — ve sekizin altısı bu sınıftan. Yani bu modelde asıl soru "oran doğru mu" değil, **"listede
ne yok"**.

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

5. **Tile süresinden yanlış genişlikte Cholesky çıkarılıyordu** — `TILE_TIMINGS`
   `hessian_block=512` ile ölçülmüş, ama `codebook_seconds_per_vector` **tam
   genişlikte** bir Cholesky çıkarıyordu, yani tile'ın hiç harcamadığı zamanı.
   Codebook %34 / %24 / %9 eksik yazılıyordu — ve en çok ince granülerlikte,
   ızgaranın maliyetinin yaşadığı yerde. `TILE_TIMING_BLOCK` her satırın hangi
   genişlikte ölçüldüğünü kaydediyor; cpu ve cuda satırları farklı düzenlerde
   alındığı için tek bir varsayım ikisi için birden doğru olamıyor.

**Ayrıca: iki ölçüm geri çekildi.** 08-24'te kaydedilen üç `cuda_f32` tile
süresinden ikisi tekrar üretilmiyor, ikisi de iyimser yönde: (2944,16) için
0.0631 yazılmış, aynı konfigürasyon bugün 0.0810 ölçüyor (**1.28×**);
(3072,128) için 0.1851 yazılmış, bugün 0.3058 (**1.65×**). Bu bir kurulum farkı
**değil**: (2560,4) satırı 1.00× üretiliyor, ve superseded *eager* satırı
%0.2 içinde üretiliyor (0.3883'e karşı `TILESPARSE_NO_COMPILE=1` ile ölçülen
0.3874). Tutmayan şey, o iki geniş satır için iddia edilen **Triton kazancı**:
1.72× ve 1.87× yazılmış, bugün 1.18× ve 1.09× ölçülüyor.

> **Mekanizma taşınmaya değer: bu iki kaldıraç çarpılmıyor.** Triton'un
> kazandırdığı şey fırlatma yüküydü; adayları toplamak *aynı* yükü bir kat
> yukarıdan siliyor. Tek bir israfa iki çare onu paylaşır, katlamaz. Modele
> ikisinin çarpımı asla verilmemeli.

Ders: **kernel mikro-benchmark'larından maliyet kurmak burada işlemiyor.** Her
eğri, kodun onu çağıracağı boyutlarda ölçülüyor, ve artık ileri telafi de
modelde (ölçülen %8.6). Modelin kendi dışında sınanması da sürüyor: k=7912'de
16 satırda %11.9, 4 satırda %39.8 sapıyor — ikisi de **fazla** yazarak, yani
12 gün bir üst sınır.

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
| **tile başına uçtan uca** | ~~1.64× / 1.72× / 1.87×~~ |

> **Bu satırın son sütunu geri çekildi (§6.3).** İnce satır tutuyor, iki geniş
> satır tutmuyor: bugün 1.18× ve 1.09×. Ve §6.7 geldikten sonra Triton'un
> marjinal katkısı zaten küçüldü — ikisi aynı israfı paylaşıyor.

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

| tasarım | nokta | süre |
|---|---|---|
| A. Tam M1 (3 bütçe × 7 tile × 5 çekiliş) | 105 | **14.9 g** |
| C. B=1.5'te 5 çekiliş, diğer bütçeler 1 | 49 | 6.3 g |
| D. Tek bütçe, 5 çekiliş, 7 tile | 35 | 4.2 g |
| **F. Tek bütçe, 1 çekiliş, 7 tile — ilk gerçek U eğrisi** | **7** | **19.9 saat** |
| G. Yalnız iki uç (T=1, T=max), 5 çekiliş | 10 | **26.9 saat** |

**Ve G/F kıyası 08-25'te bir kez daha döndü — üçüncü kez.** Önce G, F'in otuzda
biriydi; kalibrasyon modele yazılınca nokta sayısı baskın oldu ve ikisi eşitlendi
(19.1'e karşı 20.4); şimdi `TILE_TIMINGS` ince ucu doğru fiyatlayınca G **F'ten
pahalı** (26.9'a karşı 19.9). Sebep tek: G'nin iki ucundan biri **T=1**, ve T=1
ızgaranın en pahalı hücresi çıktı (§6.14).

Ders G hakkında değil: **ucuz kaçış kapısı diye bir tasarıma bakmak, maliyetin
nerede olduğunu bildiğini varsayıyor.** Üç kez yanlış bilindi — ve F zaten daha
ucuz, üstelik **yedi tile'ın tamamını** veriyor.

fp16 sütunu kaldırıldı: üç kaldıraç da §8.5'te tek yerde toplandı.

---

### 6.7 Aday ölçekleri tek aramada toplamak

`fit_scale` 24 adayı **tek tek** geçiyordu. Arama fırlatma bağımlı olduğu için
bu, sabit bedeli 24 kez ödemek demekti. İmza net:

| vektör | süre |
|---|---|
| 1,280 | 41.3 ms |
| 5,888 | 43.4 ms |

**4.6 kat veri, 1.05 kat süre.** Profil: GPU **%21.6** meşgul, tek `fit_scale`
çağrısında **3,380** çekirdek — aday başına 141.

Adaylar bağımsız (her biri aynı vektörlerin farklı bir ölçeklemesinin neye
yuvarlandığını soruyor), yani yığmak salt bir yeniden düzenleme. Ölçülen
(`FIT_ROW_BUDGET=1`, yani eski düzeni birebir üreten kola karşı):

| tile | `fit_scale` | tile başına uçtan uca | çıktı |
|---|---|---|---|
| 4 × 2560 | 9.0× | **3.78×** | bit-birebir |
| 16 × 2944 | 10.2× | **2.01×** | bit-birebir |
| 128 × 3072 | 2.0× | 1.09× | bit-birebir |

Kazanç ince granülerlikte en büyük — ızgaranın pahalı ucunda.

**§7.2'deki reddedilen kalemle karıştırılmamalı.** O, fit'i *tile'lar arasında*
toplamaktı ve her tile'ın hatasını birlikte indirgediği için aritmetiği
değiştiriyordu. Aday ekseninde her adayın hatası kendi `[n,8]`'i üzerinde,
eskisiyle aynı sırada toplanıyor.

**Testlerin gerçekten ısırdığı mutasyonla doğrulandı** — geçen bir test hiçbir
şey kanıtlamaz. Öldürdükleri: adayı komşunun kodsözcükleriyle eşleştirmek (8),
satırları harmanlamak (18), her adayı seed ölçeğiyle puanlamak (15), α'yı başka
adayın hatasıyla eşleştirmek (18), her geçişten bir aday düşürmek (1).
**Öldüremedikleri**, çünkü hiçbiri cevabı oynatmıyor: hatayı adaylar arasında
ortak indirgemek (40 float32 çekilişinde argmin aynı), berabereyi `<=` ile
bozmak, Python float yerine tensör ile bölmek, her adayı %0.1 oynatmak. İlk üçü
kimse "taşıyıcı" diye savunmasın diye kaydedildi.

### 6.8 Rotasyonun Kronecker yapısı — gerçek katmanda ölçüldü

`tile_hessian_stream` `q @ H @ q.T`'yi yoğun GEMM olarak yapıyordu, oysa `q`
`kron(RHT(p), O(m))` (`rotation.structured_orthogonal`). Çarpanlara kasılınca
maliyet `2k³`'ten `2k²(p+m)`'e iniyor.

**Kalite — gerçek Llama-2-7B blok 0 `o_proj`, 512 satır, 32,768 gerçek token,
B=1.5** (`m0_rotation_value.py --families K`). `full` kolu birinci turu
**7.0e-07** sapmayla yeniden üretiyor, yani koşu geçerli:

| çift | T=4 | T=16 | T=max |
|---|---|---|---|
| `full` → `fullK` (tam genişlik geri besleme) | **+1.85%** | +0.94% | +0.53% |
| **`H512` → `H512K` (hattın koştuğu kol)** | **−0.31%** | **−0.03%** | **−0.15%** |

> **Sentetik ölçüm iki mertebe yanıldı — §5.1'in aynı deseni.** Sentetik
> Hessian'larda etki %0.003 ve işareti rastgeleydi. Gerçek katmanda tam
> genişlik geri beslemeyle **%0.5–1.9**, ve **sistematik** (üç tile'da da aynı
> işaret). Yani "sentetikte önemsiz" bu projede bir kanıt değil.

**Ama hattın koştuğu kolda tersine dönüyor.** `hessian_block=512` ile fark
−0.03% … −0.31%, yani ölçülemeyecek kadar küçük ve **lehte**. Mekanizma §5.7 ile
tutarlı: geri besleme 512'lik bloklara kapatılınca faktorizasyon k×k değil
512×512, ve rotasyonun yuvarlama farkını büyüten şey o koşullanma. Uzun menzilli
bağlantıları atmak düzenlileştirdiği gibi hata yayılımını da bastırıyor.

Kapı B'nin ayırabildiği fark hata seviyesinin %3.2'si (§5.6). `H512K` bunun
**10 katı altında**; ama `fullK`'nın %1.85'i yalnız 1.7 kat altında — tam
genişlik geri besleme koşulacaksa yeniden ölçülmeli.

**Maliyet — ızgaranın gerçek genişliklerinde ölçüldü**, tile sayısıyla ağırlıklı:

| k | çarpanlar | yoğun | kron | |
|---|---|---|---|---|
| 2048 | 2048×1 | 3.51 ms | 3.54 ms | **0.99×** |
| 2560 | 512×5 | 7.64 ms | 2.87 ms | 2.66× |
| 2944 | 128×23 | 11.12 ms | 1.82 ms | 6.11× |
| 5504 | 128×43 | 78.3 ms | 6.06 ms | 12.91× |
| 7912 | 8×989 | 238.8 ms | 36.3 ms | 6.58× |
| 8256 | 64×129 | 271.6 ms | 15.7 ms | **17.31×** |

Ağırlıklı: **5.52×**. Tam ikinin kuvvetinde hiç kazandırmıyor (`m=1`, çarpanlara
ayrılacak tek sayı yok) ve k=2048 ızgaranın en kalabalık genişliği — ortalamayı
aşağı çeken şey o. Orada kazanmak için gerçek bir hızlı Hadamard gerekir.

> **Aşağıdaki mutlak günler o günün tabanına ait (11.98).** Taban sonra üç kez
> değişti — kalibrasyon (§6.10), telafi (§6.11b) ve `TILE_TIMINGS`'in yeniden
> ölçümü (§6.14) ile 14.9 oldu.
> **Ölçülen şey oran**, ve oran duruyor: rotasyon terimi 5.52×. Güncel toplam
> için §8.5.

| | o günkü taban | +kron |
|---|---|---|
| **M1** | 11.98 g | **8.17 g** (1.47×) |
| Tasarım F | 15.56 saat | **10.61 saat** |
| `τ` süpürmesi | 3.34 g | **2.28 g** |

Tahminim 7.8 gündü, ölçülen 8.17 — bu kez %5 iyimserdim.

**Varsayılan kapalı** (`run_config(rotate_kron=False)`): bit-birebir değil, ve
şimdiye kadarki her kalite sayısı yoğun kongrüansla ölçüldü. Açmak bir karar
gerektiriyor, fp16 gibi.

### 6.9 Üç hassasiyet kaldıracı, tek tek ve kombine

`m0_precision_levers.py` — 8 kol (2³), gerçek katman, kalite ve hız ayrı
fazlarda (hız faz'ı boş GPU istiyor, kalite istemiyor).

**Kalite** (gerçek `o_proj`, `-` kolun'a göre; eksi = **daha iyi**):

| kol | T=4 | T=16 | T=max |
|---|---|---|---|
| `tf32` | **ÇALIŞMIYOR** | +2.36% | −0.78% |
| `kron` | −0.31% | −0.03% | −0.15% |
| `fp16` | −0.34% | +0.90% | −0.07% |
| `fp16+kron` | −1.54% | +1.26% | +0.11% |
| `kron+tf32` | **ÇALIŞMIYOR** | **ÇALIŞMIYOR** | +4.80% |
| `fp16+kron+tf32` | **ÇALIŞMIYOR** | **ÇALIŞMIYOR** | +4.65% |

**Hız** (boş GPU, 4 dönüşümlü geçiş, medyan):

| kol | T=4 | T=16 | T=max | medyan |
|---|---|---|---|---|
| `tf32` | ÇALIŞMIYOR | 1.06× | 1.03× | 1.04× |
| `kron` | 1.18× | 1.11× | 1.02× | 1.11× |
| `fp16` | 1.09× | 1.16× | 1.22× | 1.16× |
| **`fp16+kron`** | **1.29×** | **1.30×** | 1.20× | **1.29×** |

> **TF32 ölçülecek bir kalite bedeli değil — hattı kırıyor.** §3.3 onu
> "hiç ölçülmemiş" diye taşıyordu; ölçüldü ve T=4'te döndürülmüş alt-Hessian
> Cholesky'den geçmiyor. Kongrüansı tek başına TF32'ye almak hiçbir bloku
> çökertmiyor ama **sönümleme payının %85'ini yiyor** (kalan 0.154×); maske ve
> telafi de TF32 altında değişince bir tile payı aşıyor. Çalıştığı yerlerde de
> en kötü kalite onda: kombine hâlde **+%4.8**, Kapı B'nin ayırabildiği %3.2'yi
> **aşan tek kol**. Kapandı.

**Bileşim — ve sıfır hipotezi çarpım değil.** Zamanın `a` kesrini kaldıran bir
kaldıraç `1/(1−a)` verir, ayrık işe binen iki tanesi `1/(1−a−b)` — ki bu
`1/((1−a)(1−b))` çarpımından **büyük**. Çarpımı beklenti saymak her bağımsız
çifti sinerjik gösterir ve asıl örtüşmeyi gizler. Doğru null'a göre:

| çift | ayrık olsaydı | ölçülen | |
|---|---|---|---|
| `fp16+kron` | 1.30× | 1.29× | **%99 — bağımsız** |
| `fp16+tf32` | 1.21× | 1.25× | %103 — bağımsız |
| `kron+tf32` | 1.16× | 1.07× | **%92 — rotasyonu paylaşıyorlar** |

Yani `kron` ve `tf32` aynı terime biniyor, `fp16` ayrı terime. Öngörülmüştü ve
ölçüm tuttu.

**M1'e etkisi** (model, ölçülen terim oranlarıyla; `o_proj`'un %24'ü rotasyon,
M1 ortalaması `down_proj` ağırlıklı olduğu için daha büyük):

> Yine o günün tabanı (11.98). Oranlar geçerli, mutlak günler §8.5'te güncel.

| kol | M1 | hızlanma | Tasarım F | `τ` |
|---|---|---|---|---|
| — | 11.98 g | 1.00× | 15.6 saat | 3.34 g |
| `fp16` | 10.44 g | 1.15× | 13.6 saat | 2.91 g |
| `kron` | **8.17 g** | 1.47× | 10.6 saat | 2.28 g |
| **`fp16+kron`** | **6.63 g** | **1.81×** | **8.6 saat** | 1.85 g |

M1 düzeyinde de ayrık: 1.15 ve 1.47'den ayrık-null 1.82×, ölçülen 1.81×.

---

### 6.10 Maliyet modelinin altıncı hatası: kalibrasyon hiç yazılmamış

**Bu, altı sürümde bulunanların en büyüğü, ve `m1_run.py`'nin süresini soran
soru ortaya çıkardı.**

`sequential_calibrate` nokta başına her bloğu **iki kez** dolaşıyor: bir kez
hook'larla Hessian'ları toplamak, bir kez de sonraki blok sıkıştırılmış çıktıyı
görsün diye (Spec v6 tuzak 20). Model ikisini de yazmıyordu.

Ölçüldü (Llama-2-7B blok 0, 7 linear, 16,384 token):

| biriktirici | süre | nokta başına | float64'e bağıl fark |
|---|---|---|---|
| **CPU float64** (kodun geldiği hâl) | 19.65 s | **5.59 h** | 0 |
| CUDA float64 | 29.86 s | 8.49 h | 3.1e-17 |
| **CUDA float32** | **0.91 s** | **0.26 h** | 5.06e-06 |
| CUDA float64 + float32 çarpım | 0.99 s | 0.28 h | 5.08e-06 |

Nokta başına 5.59 saat — **her tile boyutunda sıkıştırmanın tamamından
pahalı**. M1'in 105 noktasında 28 gün.

**Sebebi mekanik.** `collect_block_statistics` biriktiriciyi `device="cpu"` ile
kuruyordu ve hook her aktivasyonu `.to("cpu")` ile kopyalıyordu — yani `Xᵀ X`,
aktivasyonlar ve blok zaten GPU'dayken CPU'da yapılıyordu. Bloğun kendi
cihazında biriktirmek **25×**.

> **Ve kendi önerimi çürüttüm.** "float32 çarpımı float64 bir toplayıcıya
> eklemek hassasiyeti neredeyse bedava geri alır" demiştim. Ölçüldü: %9 daha
> pahalı ve **hiçbir şey kazandırmıyor** (5.08e-6'ya karşı 5.06e-6). Hata
> toplamada değil **çarpımda**; daha geniş bir toplayıcı çarpımın attığını geri
> getiremiyor. Parçaları küçültmek de işe yaramıyor — toplam hata her hâlükârda
> `√(toplam token)` gidiyor. API'de kaldı ve *neden işe yaramadığı* yazıldı.

**Tasarım ekonomisi tersine döndü.** Sıkıştırma baskınken maliyeti *hangi*
tile'ları koştuğun belirliyordu; kalibrasyon baskınken **kaç nokta** koştuğun
belirliyor. Tasarım G (2 tile × 5 çekiliş = 10 nokta) Tasarım F'ten (7 nokta)
**pahalı** hâle geliyordu — ucuz kaçış kapısı olmaktan çıkıyordu. Hessian
GPU'ya alınınca G yeniden ucuza döndü ve ikisi eşitlendi (19.1'e karşı 20.4).
**08-25'te üçüncü kez döndü:** `TILE_TIMINGS` ince ucu doğru fiyatlayınca G,
F'ten belirgin biçimde pahalı çıktı (26.9'a karşı 19.9), çünkü uçlarından biri
T=1 (§6.14). Gözlem duruyor; yönü üç kez değişti.

**`m1_run.py` bugün başlatılsaydı:**

| senaryo | M1 | Tasarım F |
|---|---|---|
| modelin söylediği (iki terim de yok) | 11.98 g | 15.6 saat |
| **gerçek kod, iki düzeltmeden önce** | **~38–40 g** | **~57–60 saat** |
| + Hessian GPU'da (§6.10) | 15.0 g | 20.4 saat |
| + `TILE_TIMINGS` yeniden ölçüldü (§6.14) | **14.9 g** | **19.9 saat** |
| + telafi bloklanmış (§6.11c, varsayılan kapalı) | 13.6 g | 17.8 saat |
| + fp16 + kron (varsayılan kapalı) | 8.8 g | 12.7 saat |
| + üçü birden (varsayılan kapalı) | **7.5 g** | **10.5 saat** |

"Önce" satırı aralık, çünkü aynı ölçümün iki koşusu 19.65 s ve 22.37 s verdi —
bu makinede %14 koşudan koşuya. Kod artık öyle yapmadığı için kesinleştirmeye
değmez; kaydedilmesi gereken şey aralığın kendisi.

**Neyin altı sürüm boyunca saklanmasına izin verdiği kayda değer:** tam
sürücüyü kimse koşmadı, çünkü `m1_run.py` yok. §8.1'in kritik yol olmasının
sebebi yalnızca "veri yok" değil — **ölçülmeyen maliyet de orada birikiyor.**

---

### 6.11 İki kapı, bir eksik terim, ve kendi kaydımın düzeltilmesi

**a) `_nearest` ızgaranın 21 hücresinin 10'unda 65,536 kodsözcüğünü tarıyordu.**

Hızlı yol tek bir kapıyla açılıyordu — `_LATTICE_MIN_ROWS` (cuda'da 1024) — ve
analitik aramanın **kendi** eşiği (`_ANALYTIC_MIN_ROWS = 384`) yalnız o kapının
*içinde* okunuyordu. Yani **384 ≤ satır < 1024 aralığı analitik yola hiç
ulaşamıyordu.**

Köşe durum değil: LDLQ süpürmesi `_nearest`'e `chunk × lines_per_tile` satır
veriyor — T=1 ve T=2'de 512, T=4'te 816. Tam ızgaranın ince ucu, tile sayısının
en büyük olduğu yer.

Ölçülen doğrudan-yol krossoveri **256**, 384 değil — çünkü 384 *geri düşme*
yolunun eşiği ve orada kafes çözücünün bedeli zaten ödenmiş:

| n | 0.05 | 0.6 | 6.0 |
|---|---|---|---|
| 128 | 0.41× | 0.63× | 0.41× |
| 256 | 0.99× | 1.21× | 1.68× |
| 512 | 2.04× | 1.61× | 3.33× |
| 816 | 2.68× | 5.19× | 6.04× |

Kapı açılınca **taramaya sıfır satır** düşüyor, ve ızgaranın gerçek
şekillerinde süpürme:

| hücre | satır | önce | sonra | |
|---|---|---|---|---|
| T=1 4096×4096 | 512 | 0.332 s | 0.163 s | **2.04×** |
| T=2 4096×4096 | 512 | 0.682 s | 0.294 s | **2.32×** |
| T=4 4096×4096 | 816 | 1.337 s | 0.382 s | **3.50×** |
| T=8 4096×11008 | 560 | 2.701 s | 1.101 s | **2.45×** |

> **Sınıfı tanıdık: `_on_device` ile aynı.** Bir algoritma için kalibre edilmiş
> bir kapı, sonradan gelen daha iyisini sessizce dışarıda bırakıyor, ve belirti
> yanlış cevap değil yavaş cevap. Bu yüzden testler **yolu** izliyor: tarama
> ile analitik zaten yapı gereği aynı cevabı veriyor — boşluğun fark
> edilmemesinin sebebi de o.

**Kesinlik.** Analitik arama exact; float32'de milyonda bir gerçek berabere
başka türlü bozulabiliyor (§6.4'te zaten kabul edilmiş bir takas). Uçtan uca
dört tile'ın üçü bit-birebir aynı kaldı, T=16'da katman hatası **5.8e-5**
oynadı — Kapı B'nin görebildiğinin **550 katı altında**.

**b) Maliyet modelinin yedinci hatası: ileri telafi hiç yazılmamış.**

`run_config` her şeyden önce `prune`'u çağırıyor, `TILE_TIMINGS` ise
`ldlq_quantize_blocks`'tan başlıyor. Arada `forward_compensate` var — `n_in`
uzunluğunda bir Python döngüsü, ve her yinelemesi kalan bütün genişliğe
dokunuyor. Ölçüldü: blok başına **40.7 s**, nokta başına **0.362 saat**,
M1'de **1.58 gün**. Kalibrasyon gibi tile boyutundan bağımsız.

**c) Ve "bloklamak kazandırmıyor" kaydım yanlıştı.**

§7.2'ye "0.90× / 0.87× / 1.06× — kazanç yok" diye yazmıştım. O ölçümü yalnız
(512, 2048) ve (512, 4096)'da yapmışım — **fırlatma bağımlı** rejimde, ve orada
bloklama gerçekten hiçbir şey kaldırmıyor. Gerçek katman genişlikleri **bant
genişliği bağımlı**:

| n_out × n_in | kesin | blok=512 | |
|---|---|---|---|
| 4096 × 4096 | 2431 ms | 665 ms | **3.65×** |
| 11008 × 4096 | 6345 ms | 820 ms | **7.74×** |
| 4096 × 11008 | 18260 ms | 1837 ms | **9.94×** |

Terimin tamamında **6.63×** (0.362 → 0.055 h/nokta). Bit-birebir **değil**
(ertelenen kuyruk tek matmul, 2.7e-6…4.8e-6), o yüzden `compensate_block`
eklendi ve **varsayılan `None`** — kesin düzen.

> **Ders, hızdan daha değerli:** *yanlış rejimde ölçülmüş bir ret, hiç ölçmemekten
> kötüdür* — çünkü bir sonrakinin bakmasını durdurur. Bu kayıt beni sekiz gün
> boyunca yanlış yerde tuttu.

**Bugünkü dağılım** (B=1.5, 7 tile toplamı):

| terim | süre | pay |
|---|---|---|
| codebook | 7.51 h | 36.8% |
| rotasyon | 6.48 h | 31.8% |
| **telafi** | **2.53 h** | **12.4%** |
| kalibrasyon | 2.31 h | 11.3% |
| cholesky | 1.10 h | 5.4% |
| eval | 0.46 h | 2.3% |

**M1 = 14.9 gün** (§6.14'ten sonra). Telafi yazılınca 13.4'ten 15.0'a çıkmıştı;
bloklanırsa 13.6'ya döner (§8.5).

---

### 6.12 Dikiş GPU'da hiç koşmamış — bir yolda beş kusur

08-25. §8.1'i yazmadan önce onun **kullanacağı dikişe** bakıldı:
`sequential_calibrate` → `run_config`. `sequential_calibrate(device="cuda")` —
`m1_run.py`'nin yapmak zorunda olduğu tam çağrı — **çalışmıyordu**, ve arkasında
üst üste beş kusur vardı:

| # | kusur | belirtisi |
|---|---|---|
| 1 | `block_kwargs` cihaza hiç taşınmıyordu | rotary CPU'da kalıyor, blok `apply_rotary_pos_emb` içinde ölüyor |
| 2 | `LayerProblem`'in W'si CPU'ya sabitlenmiş, H bloğu izliyor | cihaz uyuşmazlığı — **ve hiçbir argüman uzlaştırmıyordu** |
| 3 | `dtype` varsayılanı GPU'da float64 | 1/64 hız: blok başına 29.9 s'ye karşı 0.9 s |
| 4 | `inputs` liste sözleşmesi `device` verilince bozuluyor | çağıranın listesi blok 0'da donuyor |
| 5 | `run_config` `W_hat`'ı hesaplayıp atıyor | `compress_fn` ağırlık döndürmek zorunda — iki yarı **bağlanamıyordu** |

**Beşi de 599 testin kör noktasında**, ve hepsi aynı sebeple: CPU'da bunların
hiçbiri bir şey yapmıyor. `.cpu()` zaten CPU'daysa no-op, CPU `block_kwargs`
CPU bloğun yanında zaten doğru, hiçbir şey taşınmıyorsa yeniden bağlama zararsız.

> **Bu kör nokta §14.1'de zaten kayıtlıydı** — iki önceki düzeltmeden. Yine
> vurdu, ve bu kez tam da hızlandırma commit'inin (`8c56f1e`) kendisinde:
> o commit dokunduğu **parçaya** CUDA testi ekledi (`collect_block_statistics`),
> onu **çağıran sürücüye** eklemedi.

**3'ün fiyatı ölçülmedi, hesaplandı** — modelin kendi kayıtlı oranıyla:

| | M1 |
|---|---|
| modelin fiyatladığı (`cuda_f32`) | **14.9 gün** |
| kodun varsayılanının ürettiği (`cuda_f64`) | **50.9 gün** |

25× kazanç gerçek ama **çağıranın `dtype=torch.float32` yazmasına bağlı**.
Varsayılan yine de float64 bırakıldı: float32 kolunun 5.06e-06'sının ölçüldüğü
referans o, ve bu sürücüden geçen **hiçbir kalite sayısı henüz yok** — ucuz kolu
seçmek koşacak olanın kararı, bir yan etki değil. Docstring'e ve §10'a yazıldı.

**Şimdi kapalı, ve testler cevabı değil yolu izliyor** (§14.2): blok 0'a bir
pre-hook takıp rotary'nin *hangi cihazda geldiğini* sayan bir test, `compress_fn`
içinde W/H/act_norm yerleşimini kümeye toplayan bir test, ve dikişin tamamını
koşan bir uçtan uca test. Beşi de HEAD'e karşı **kırmızı** olduğu gösterilerek
kabul edildi.

Kanıt, uçtan uca: gerçek Llama blokları → `sequential_calibrate(device="cuda")`
→ `run_config(return_weight=True)` → ağırlıklar modele geri → `streamed_perplexity`.

---

### 6.13 Üç sabit, tek eşitsizlik, ızgaranın ortası

08-25, ikinci bulgu. Üç sabit birbirinden habersiz ayarlanmış ve aralarında
**hiçbir yerde yazılmayan** bir eşitsizlik varmış:

```
CHUNK_TARGET_ROWS * DECODER_MISS_FRACTION  >  _ANALYTIC_MIN_ROWS
```

Soldan sağa: `auto_chunk` süpürmeyi kaç satıra nişanlıyor, çözücü bunların ne
kadarını devrediyor, ve o artık analitik yola geçecek kadar büyük mü. Ölçülen
üçüncü sayı çözücünün **%34.9**'u çözemediği (üç şekilde de aynı — yapısal).

```
1024 × 0.349 = 357  >  384   →  YANLIŞ
```

Yani süpürmenin **her grubu** 65,536 kodsözcüğünü tarıyordu.

**Ve bu ızgaranın köşesi değil, ortası.** `auto_chunk`'ın doyum tavanı
`ceil(hedef / lines)`, yani `lines` 1024'ü böldüğü her yerde chunk **tam 1024
satıra** oturuyor — B=1.5'te 21 hücrenin **sekizi**:

| hücreler | satır | yol |
|---|---|---|
| T=1, 2, 4 ve T=8'de `down_proj` | 192–816 | doğrudan analitik, temiz |
| **T=8, 16, 32 — sekiz hücre** | **1024** | çözücü + **TAM TARAMA** |
| T=max | 4096–11008 | çözücü + analitik, temiz |

Gerçek katmanda sayıldı (blok 0 `o_proj`, 2048 satır, B=1.5) — grup başına bir
çağrı, her seferinde:

| T | tarama çağrısı | taranan satır | düzeltmeden sonra |
|---|---|---|---|
| 8 | 581 | 184,915 | **0** |
| 16 | 623 | 193,184 | **0** |
| 32 | 645 | 196,712 | **0** |

**İki sabit birden oynadı, çünkü hiçbiri tek başına yetmiyor.** Satır hedefini
büyütmek eriştiği hücreleri kurtarıyor; `down_proj`'a erişemiyor — orada k=7912
chunk'ı **bellekten** 67 tile'a kapatıyor ve satır 1072'de kalıyor. Eşiği
düşürmek de yalnız o hücrede kayda değer.

**Eşik ölçülürken bir tuzak daha çıktı.** Tek ölçekte (a=0.6) krossover 192
görünüyor ve orada 1.13×. Üç ölçekte bakınca 192 diğer ikisinde **kaybediyor**;
her ölçekte kazanan ilk değer **320**. Mevcut `_ANALYTIC_DIRECT_MIN_ROWS`'un
kendi notu aynı şeyi söylüyordu ("256, 192 değil — 192 üçünden birinde
kaybediyor") ve yine de aynı tuzağa düştüm.

| satır | a=0.05 | a=0.6 | a=6.0 |
|---|---|---|---|
| 192 | 0.74× | 0.90× | 0.66× |
| 256 | 0.93× | 1.42× | 1.42× |
| **320** | **1.65×** | **1.78×** | **1.61×** |

**Satır hedefi de bayattı.** `CHUNK_TARGET_ROWS`'un docstring'i "256'dan 1024'e
%3, üstünde alacak bir şey yok" diyordu — analitik aramadan, toplu fit'ten ve
Triton'dan **önce** ölçülmüş, ve yalnız **doyumu** fiyatlıyordu, satır sayısının
hangi arama **yolunu** seçtiğini değil. Gerçek şekillerde ve gerçek tile
sayılarıyla yeniden ölçüldü; plato **2048**'de (toplam 1.20× / 1.24×, aradaki
fark bu ölçümlerin %2–5 yayılımının içinde), ve ötesinde üç şekilde zaten
`CHUNK_BUDGET_BYTES` bağlıyor.

**Hız — gerçek şekiller, gerçek tile sayıları, tek süreçte dönüşümlü, boş kart:**

| şekil | eski 384/1024 | yeni 320/2048 | | taranan satır |
|---|---|---|---|---|
| T=8 k=2816 | 8208 ms | 6992 ms | 1.17× | 479,415 → 0 |
| T=16 k=2944 | 6559 ms | 4750 ms | **1.38×** | 505,066 → 0 |
| T=32 k=3008 | 5662 ms | 3658 ms | **1.55×** | 510,812 → 0 |
| T=16 k=7912 (`down`) | 14439 ms | 12577 ms | 1.15× | 1,329,201 → 0 |
| **toplam** | 34.9 s | 28.0 s | **1.25×** | |

Yayılımlar %1–10. İki uçtaki 1.15–1.17×'in sebebi aynı: orada **bellek**
bağlıyor, satır hedefi erişemiyor — `down_proj` 67 tile'da, T=8 195'te kapanıyor.

**Kalite bedeli — gerçek katmanda, ve neredeyse yok.** T=8 ve T=16 **bit-birebir**
aynı; T=32'de bağıl hata **−0.00002%** (ve lehte). Kapı B'nin ayırabildiği
%3.2'nin 160,000 katı altında. §6.11a aynı takası 5.8e-5 ile almıştı; bu ondan
da küçük.

> **Maliyet modeli o gün güncellenmedi; ertesi gün yeniden ölçüldü** (§6.14).
> Çarpanla yamamak yerine `TILE_TIMINGS`'in tamamı `n_tiles` kaydedilerek
> yeniden alındı — ve doğru olan buymuş: ölçüm, tahmin edilen ~%4'ü değil,
> ızgaranın **şeklini** değiştirdi.

> **Mikro-benchmark yine fazla vaat etti.** 353 satırlık artık kümede analitik,
> taramaya karşı izole ölçümde **2.03×**. Aynı taramaları süpürmeden kaldırmak
> **1.04×** ediyor — tarama, peşinden gelen üçgen çözüm ve geri besleme
> matmul'üyle örtüşüyor, yani izole maliyetinin çoğu yerinde zaten gizli. §6.3'ün
> kuralı bir kez daha, ve bu sefer **sildiğin** çekirdek için.

---

### 6.14 Maliyet modelinin sekizinci hatası: 4 satırın altında hiç örnek yokmuş

`TILE_TIMINGS`'in üç satırı elle ölçülmüştü ve **hangi `n_tiles` ile** ölçüldüğü
hiçbir yerde yazmıyordu. Bu bir ayrıntı değil: `auto_chunk` `n_tiles`'ı chunk'a,
chunk'ı da `_nearest`'in gördüğü satır sayısına çeviriyor — yani **hangi arama
yolunun koştuğunu** o sayı belirliyor. Bir tile süresi, alındığı tile sayısı
olmadan yorumlanamaz.

Bedeli §6.13'te göründü: iki sabit oynadı ve eski satırlar yeni davranışa
**ölçeklenemedi**, çünkü hangi rejimde alındıklarını kimse söyleyemiyordu.

`experiments/m0_tile_timings.py` yazıldı; şekilleri seçmiyor, `accounting`'den
**türetiyor**. B=1.5'te 4096×4096 katmanının bütün tile ekseni:

| T | k | satır | n_tiles | chunk | s/tile | yayılım |
|---|---|---|---|---|---|---|
| 1 | 1024 | 1 | 4096 | 512 | 0.00729 | — |
| 2 | 2048 | 2 | 2048 | 256 | 0.00881 | %1 |
| 4 | 2560 | 4 | 1024 | 204 | 0.00997 | %7 |
| 8 | 2816 | 8 | 512 | 195 | 0.01410 | %5 |
| 16 | 2944 | 16 | 256 | 128 | 0.01882 | %4 |
| 32 | 3008 | 32 | 128 | 64 | 0.03000 | %2 |
| max | 3072 | 4096 | 1 | 1 | 1.95030 | %4 |

**İki örnek şekli bilerek değişti.** `(3072, 128)` emekli — ızgarada 128 satırlı
hücre yok, o nokta kaba ucu temsil etsin diye konmuş bir sondaydı ve T=max'in
gerçek şekli **tek tile'da 4096 satır**, bambaşka bir rejim. T=1, 2, 8 ve 32
eklendi.

**Ve asıl bulgu o eklemede.** Eski küme 4 satırın altında hiçbir şey
taşımıyordu, model de en yakın satır sayısını log uzayında seçtiği için T=1 ve
T=2 **4-satır oranını ödünç alıyordu**. Ağırlık başına maliyet:

| satır | 1 | 2 | 4 | 8 | 16 | 32 | 4096 |
|---|---|---|---|---|---|---|---|
| s/ağırlık | **6.36e-6** | 1.77e-6 | 7.82e-7 | 5.36e-7 | 3.54e-7 | 2.89e-7 | 1.55e-7 |

Uçtan uca **41 kat**, ilk adımda tek başına **8 kat**. Sebebi `fit_scale`: tile
başına bir kez uydurulyor, ve tek satırlık bir tile ona amortize edecek **128
vektör** veriyor, T=4 **1280**.

**Toplam neredeyse kıpırdamadı, şekil kaydı.** M1 15.0 → **14.9 gün** — ince uç
pahalılaştı, kaba uç ucuzladı, ikisi birbirini götürdü. Ama:

- **Tepe ortadan ince uca kaydı.** §6.1 "maliyet ızgaranın ortasında tepe
  yapıyor" diyordu; eğri artık T=1'den itibaren monoton azalıyor.
- **T=1'de bir duvar geri geldi.** Codebook 2.86h, rotasyon + cholesky toplamı
  0.80h — **3.6 kat**. `test_no_single_term_dominates_the_pass_any_more`'un
  "hiçbir terim kaçmıyor" iddiası bu yüzden kırmızıya döndü ve **haklı olarak**:
  o iddia da süresi dolmuş bir olguymuş, ince uç yanlış fiyatlandığı sürece
  doğru duruyordu.
- **Tasarım G, F'ten pahalı oldu** (26.9'a karşı 19.9 saat) — üçüncü kez yer
  değiştirdiler, ve sebebi G'nin uçlarından birinin T=1 olması (§6.6).
- **`τ` süpürmesi 4.5 → 5.5 gün.**

> **Ve T=1 ızgaranın herhangi bir hücresi değil.** Yapısız taban orası — tezin
> `d(T) − d(1)` özdeşliğinde kıyas grubu. Yani modelin en yanlış bildiği hücre,
> tezin en çok konuştuğu hücreymiş.

**Bu hata üçüncü kez provenance düzeltilirken çıktı**, aranarak değil. Diğer
ikisi kalibrasyon ve ileri telafiydi (§6.10, §6.11b). Üçünün ortak yanı: model
bir şeyi *yanlış* hesaplamıyordu — **bilmiyordu**.

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
| **TF32** | **hattı kırıyor** | Ölçüldü 08-24 ve iş kalite yüzdesine kalmadı: T=4'te döndürülmüş alt-Hessian Cholesky'den geçmiyor, sönümleme payının %85'i gidiyor. Çalıştığı yerde de +%4.8 ile Kapı B'nin %3.2'sini aşan tek kol (§6.9). **Kapandı** |
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
| **Fit'i TILE'LAR ARASINDA toplamak** | 2.16× | Alınmadı: bit-birebir **değil** (indirgeme sırası). Aday ekseninde toplamakla karıştırılmasın — o **alındı** ve bit-birebir (§6.7) |
| ~~**`forward_compensate`'i GPTQ gibi bloklamak**~~ → **bu kayıt YANLIŞTI** | (512,2048) ve (512,4096)'da 0.87–1.06× | Ölçüm doğru, **rejim yanlıştı**. O genişlikler fırlatma bağımlı. Gerçek katmanlarda (4096×4096, 11008×4096, 4096×11008) bant genişliği bağımlı ve bloklama **3.65× / 7.74× / 9.94×**. Terimde 6.63×. §6.11c |
| **Blok başına CPU↔GPU aktarımı** | nokta başına ~220 s | Saatlere karşı ihmal edilebilir |
| **İki kaydırmayı tek çözümde yığmak** | 1.9–2.2× | Alınmadı: Triton füzyonu aynı kazancı zaten topluyor |

### 7.3 Bilinçli olarak yapılmayanlar

| ne | tarih | neden |
|---|---|---|
| **E8P'nin ucuz doğrulama deneyi** | 08-20 | Kullanıcı kararı. Risk §3.2'de açık varsayım olarak taşınıyor — bu projenin en büyük tek riski |
| **Kernel yazmak** | — | Spec §8 kapsam dışı bırakıyor. Roofline alt sınır olarak sunulur, hız iddiası yapılmaz |
| **AQLM-survivor** | 08-20 | VQ ailesinin en zayıf ve en pahalı üyesi; codebook yeniden kalibrasyonu katman başına saatler |
| **Izgarayı daraltmak** | 08-24 | Artık gerekmiyor (12 gün). Gerekseydi bile tile eksenine dokunulmazdı |

---

## 8. Sırada ne var

### 8.1 Bir sonraki oturumun ilk işi — kritik yol

**`experiments/m1_run.py` yaz: tam model sürücüsü + checkpoint.**

Şu an yok, ve "sıkıştırılmış ppl hiç ölçülmedi" durumunun sebebi bu.
`calibrate.sequential_calibrate` (mevcut, `compress_fn` alıyor) ile
`eval.streamed.streamed_perplexity` (mevcut) arasını bağlayacak; `compress_fn`
içinde `m1_gates.run_config`'in hattı çağrılacak. `hf_llama.load_llama` ve
`capture_block_inputs` hazır.

> **08-25: dikiş artık gerçekten bağlanıyor.** Yukarıdaki paragraf "mevcut"
> diyerek **fazlasını vaat ediyordu** — `sequential_calibrate(device="cuda")` çalışmıyordu,
> `run_config` ağırlık döndürmüyordu, ve arada beş kusur vardı (§6.12).
> Şimdi ikisi de test altında ve zincir uçtan uca koşuyor. Yani `m1_run.py`
> gerçekten ince bir adaptör olabilir; kalanı **checkpoint** ve **sürücünün
> kendi argümanları**.

Sürücünün geçirmesi gereken iki argüman, ikisi de varsayılan değil:
`dtype=torch.float32` (yoksa M1 15 değil 51 gün — §6.12) ve
`return_weight=True`.

**Kesintiye dayanıklılık bunun parçası, sonradan eklenen bir şey değil.**
15 saatlik bir koşu dizüstünde kesilir. Kayıt birimi blok (nokta başına 32);
anahtar `(model, budget, tile, draw, block)`.

> **Dikkat:** `sequential_calibrate` Hessian'ı **sıkıştırılmış** modelden okuyor
> (Spec v6 tuzak 20). Devam ederken bir sonraki bloğun **girdileri de** kayıttan
> gelmeli — yoksa devam eden koşu kesilmeyenden farklı sonuç verir. Doğrulaması
> net: kesip devam ettirilen koşu, kesilmeden koşulanla aynı ppl vermeli
> (`tests/test_hf_llama.py`'nin tiny Llama'sı ile test edilebilir).

### 8.2 Sonra: ilk gerçek koşu

**Tasarım F** — tek bütçe, tek çekiliş, 7 tile, **19.9 saat** (§8.5'in üç
kaldıracı açılırsa 10.5). Hattın gerçek modelde uçtan uca çalıştığını kanıtlar
ve **ilk gerçek U eğrisini** verir.

Kapı B'yi karara bağlamaz (§5.6: verdikt için ≥5 çekiliş, `gate_b` altında
"undetermined" döner). Ama Kapı A için ve eğrinin şeklini görmek için yeterli.

### 8.3 Ön-kaydı dondurmak — artık maliyet engeli yok

İki kutu açık: **`Δ(T)` tahmin eğrisi** ve **`T*_tahmin`**. İkisi de `τ`
süpürmesine bağlı, ve süpürme 29 gündü. **Artık 5.5 gün**
(`m0_cost_model.sweep_cost`) — yani maliyet artık engel değil.

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

> **Gerekçesi 08-24'te daraldı ama yön değiştirdi.** "Hem daha iyi hem daha
> ucuz olabilir" diyordu; ucuzluk tarafı gitti — fit artık tile'ın %28'i, yani
> tamamen atsan bile 1.4 gün. Geriye **yalnız kalite** gerekçesi kaldı, ve o
> gerekçe zayıflamadı: yanlış ölçüye göre daha kesin bir α hâlâ yanlış.

### 8.5 Maliyet tarafında elde duran, ölçülmüş ama açılmamış kaldıraçlar

Üçü de **kodda var, test edilmiş, ölçülmüş ve varsayılan kapalı.** Hiçbiri
bit-birebir değil, ve şimdiye kadarki her kalite sayısı üçü de kapalıyken
alındı — açmak bir karar, bir sonuç değil (kullanıcı kararı, 08-24).

| kaldıraç | nasıl | hız | kalite bedeli |
|---|---|---|---|
| `rotate_kron=True` | Kronecker kongrüansı (§6.8) | rotasyon terimi 5.52× | gerçek katmanda −0.03…−0.31% (**lehte**) |
| `search_dtype=float16` | arama fp16'da (§6.9) | codebook terimi 1.24–1.52× | ≤%0.90 |
| `compensate_block=512` | telafi bloklanmış (§6.11c) | telafi terimi 6.63× | 2.7e-6…4.8e-6 |

Birlikte **M1 14.9 → 7.5 gün** (2.00×), Tasarım F 19.9 → 10.5 saat. `fp16` ile
`kron` ayrık terimlere biniyor ve ayrık-null'a göre %99 birleşiyorlar (§6.9).
Tek tek: kron 10.2 g (1.46×), telafi 13.6 g (1.10×), fp16 13.5 g (1.10×).

**Kapsam dışı bırakılanlar** (ölçüldü ya da tanımlandı, yapılmadı):
- **TF32** — reddedildi, kalite yüzdesi yüzünden değil: hattı **kırıyor** (§6.9)
- **Hızlı Hadamard dönüşümü** — kron tam ikinin kuvvetinde hiçbir şey
  kazandırmıyor (`m=1`, ayrılacak tek çarpan yok), ve **k=2048 ızgaranın en
  kalabalık genişliği** (T=1'de blok başına 11,008 tile). Gerçek yeni kod
- **LDLQ süpürmesine CUDA graph / `torch.compile`** — §6.5'in açıkça bıraktığı
  yer. §6.11a şekilleri veriden bağımsız yaptığı için artık **mümkün**: analitik
  arama sabit şekilli, kafes çözücünün geri düşmesi değil

### 8.6 Açık kalan kod işleri

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
- **`SCALE_FIT_MULTIPLIER` tek bir sabit** (1.39) ve satır sayısıyla değişmiyor —
  oysa `fit_scale`'in payı T=1'de çok daha büyük olmalı (§6.14'ün mekanizması).
  `TILE_TIMINGS` ile aynı sınıf: satır eksenine yayılmamış bir ölçüm

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

**Maliyet artık birincil risk değil ama sıfır da değil.** 15 gün hâlâ uzun, ve
bütün süreler **bu makineye ve Triton'lu bir kuruluma** ait. Başka donanımda
eğriler yeniden ölçülmeli. Model **sekiz** kez yanıldı, ve 08-24'te iki ölçümü de
geri çekildi (§6.3) — yani bu sayının belirsizliği modelin kendi hata payından
değil, ölçümlerin tekrarlanabilirliğinden geliyor. Ve 08-25 bir sınır daha
gösterdi: model, kodun **varsayılanlarıyla** koşulduğunda ne olacağını yazmıyor
(§6.12'de 14.9 güne karşı 51).

**Kesinti.** 15 saatlik bir koşu bile dizüstünde kesilir ve şu an devam etme
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
| ~~**`_on_device` önbelleği cihaz DİZESİYLE anahtarlı**~~ → **düzeltildi 08-24** | `"cuda"` ile `"cuda:0"` **farklı nesneler** döndürüyordu ve hızlı yol bir `is` kontrolüyle seçiliyor, yani kısa yazımı kullanan her çağıran sessizce kaba kuvvete düşüyordu. Üç oturum boyunca belgede durdu, kodda düzeltilmedi, ve **08-24'te dört ölçümü daha yanılttı** — ikisi önce GPU çekişmesine, sonra saat düşüşüne yoruldu, ikisi de yanlış. Belirtisi kötü: optimizasyon **1.00× görünüyor**, yani hata değil sonuç gibi okunuyor. Artık `_device_key` anahtarı normalleştiriyor; ölçülen ek maliyet dönüşümlü A/B'de −%0.8/+%0.4 |
| **Kıyaslamada hızlı yolun açık olduğunu doğrula** | `quantize.is_canonical_codebook(cb)` — bir `assert` ile. Kendi codebook kopyasını kuran ya da cihazı kısa yazan bir kıyaslama hâlâ taramayı ölçer, ve bu **doğru** davranış; tek sorun sessiz olmasıydı. Zamanlama yazarken bunu iddia et |
| **Python stdout tamponu arka plan koşularında** | `python -u` |
| **`torch.compile` Windows'ta çalışmıyor sanılıyordu** | `pip install triton-windows==3.7.0.post26` (torch 2.12 → triton 3.7.0). Sonra `has_triton()` True |
| **Inductor CPU'da `cl` (MSVC) istiyor** | CUDA derleniyor, CPU derlenemiyor. `quantize._shift_kernel` cihaz/dtype başına sondalıyor ve eager'a düşüyor — sessiz, çünkü iki yol birebir aynı |
| **`TILESPARSE_NO_COMPILE=1`** | Derlemeyi kapatır; derli/derlisiz karşılaştırma ve toolchain sorunları için |
| **Mutlak süreler koşular arasında karşılaştırılamaz** | Aynı ölçüm iki koşuda %14–37 oynadı, bazen tanımlanabilir bir sebep olmadan. **Yalnız tek süreçte dönüşümlü A/B geçerli.** Bu oturumda bir kez +%37 okuyup değişikliğe yordum; makineymiş |
| **GPU çekişmesi fırlatma kaldıraçlarını GİZLER** | Başka bir iş kartı doldurunca darboğaz GPU'ya geçiyor ve silmeye çalıştığın gecikme zaten gizleniyor — optimizasyon **1.00× okuyor**. Dönüşümlü A/B bunu düzeltmez; hız fazından önce `bench_guard.require_quiet_gpu()` çağır |
| ~~**`nvidia-smi`'nin `utilization.gpu`'suna bak**~~ → **düzeltildi 08-25** | Bu makinede o sayı **yükle ters korele**: boş kartta %42, kendi yükümüzde %25. Sebebi WDDM — kart ekranı da sürüyor, ve listelenen "compute" süreçleri Windows kabuğu, Edge WebView ve **Claude uygulamasının kendisi**. `mem_get_info` daha kötü: **kör**, 3 GiB tutan yabancı süreci hiç görmüyor. Çalışan tek gösterge `clocks.sm` (boşta %42, yabancı yükte %89). Ölçülüp `experiments/bench_guard.py`'ye yazıldı |
| **TF32 hattı kırıyor** | `allow_tf32=True` ile döndürülmüş alt-Hessian Cholesky'den geçmiyor: sönümleme payının %85'i gidiyor. Açma (§6.9) |
| **`sequential_calibrate` varsayılanı GPU'da float64** | Yani 1/64 hız: blok başına 29.9 s'ye karşı 0.9 s, ve yerine geçtiği CPU float64'ten (19.7 s) bile yavaş. Maliyet modeli `cuda_f32`'yi fiyatlıyor; varsayılanla koşmak **M1'i 15 günden 51'e** çıkarır. Sürücü `dtype=torch.float32` geçmeli — varsayılan bilerek değiştirilmedi (§6.12) |
| **CPU'da koşan bir test cihaz varsayılanını sınayamaz** | `.cpu()` zaten CPU'daysa no-op, CPU `block_kwargs` CPU bloğun yanında zaten doğru. Bu oturumlarda **üç kez** vurdu; sonuncusunda tek bir yolda beş kusur biriktirdi (§6.12). Cihaza dokunan her değişikliğin CUDA işaretli bir testi olmalı, ve test **parçanın değil çağıranın** üstünde |

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
| `calibrate.py` | sıralı kalibrasyon, `LayerProblem` (**dikiş yeri**), `sequential_calibrate` (**sürücü — henüz yalnız testlerden çağrılıyor**; GPU'da `dtype=torch.float32` geçilmeli, §6.12), Hessian biriktirici (cihazda) |
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
| `experiments/m0_precision_levers.py` | fp16 / kron / TF32, tek tek ve sekiz kombinasyonda |
| `experiments/m0_pass_breakdown.py` | bir geçişin fazları — modelin yazmadıklarını bulmak için |
| `experiments/bench_guard.py` | kartın ölçülecek kadar boş olduğunu **fırlatarak** doğrular; dönüşümlü A/B ve yayılım raporu |
| `experiments/m0_chunk_rows.py` | `auto_chunk`'ın satır hedefi ile arama eşiğinin etkileşimi — yol sayımı + zaman |
| `experiments/m0_tile_timings.py` | `TILE_TIMINGS`'i ızgaranın gerçek hücrelerinden **türeterek** ölçer; `n_tiles` kaydeder |

**Belgeler:** `docs/spec_v7.md` (şartname) · `preregistration.md` (M1 ön-kaydı,
**dondurulmadı** — iki kutu kaldı, artık maliyet engeli yok) ·
`docs/audit.md` (v6 denetimi) · `docs/gate_a_dry_run.md` (literatür provası) ·
bu belge.

**Henüz yazılmamış betikler** (§8.1, §8.3): `experiments/m1_run.py` —
tam model sürücüsü + checkpoint · `experiments/tau_sweep.py` — `τ` yüzeyi.
İkisi de `sequential_calibrate` + `streamed_perplexity` dikişini kullanacak.

```bash
python -m pytest tests/ -q                         # 619 test, ~2 dk
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
| `8f5f59f` | Bu belge yeniden yazıldı: yapılan / yapılmayan / reddedilen ayrıldı |
| `1efa971` | **Ölçek adayları tek aramada** (§6.7); maliyet modelinin **beşinci hatası** ve iki geri çekilen ölçüm (§6.3). 17 → 12 gün, ve baskın terim codebook'tan **rotasyona** geçti |
| `0a19f90` | **`_on_device` tuzağı kapatıldı** (§10). Üç oturumdur belgede duran, kodda durmayan hata; dört ölçümü bozmuştu |
| `de8a5ec` | **Kronecker kongrüansı gerçek katmanda ölçüldü** (§6.8). Sentetik ölçüm iki mertebe yanılmıştı; hattın kolunda etki lehte, M1 11.98 → 8.17 g. Varsayılan kapalı |
| `383a64a` | **Üç hassasiyet kaldıracı, tek tek ve kombine** (§6.9). TF32 **hattı kırıyor** — §3.3'ün açık kalemi kapandı. `fp16+kron` bağımsız, birlikte M1 11.98 → **6.63 g** |
| `8c56f1e` | **Kalibrasyon Hessian'ı GPU'da** (25×) ve maliyet modelinin **altıncı hatası** (§6.10). `m1_run.py` bugün 40 gün sürerdi, 13.4 değil |
| `98c0413` | **`_nearest`'in ikinci kapısı** (§6.11a, süpürmede 2.0–3.5×) ve **yedinci eksik terim** (§6.11b). Ayrıca §7.2'deki bir ret **yanlış rejimde ölçülmüş** çıktı (§6.11c) |
| `a3d5a05` | **§8.1'in dikişi GPU'da hiç koşmamış** (§6.12). Bir yolda **beş** kusur, beşi de 599 CPU testinin kör noktasında. Zincir artık uçtan uca koşuyor: gerçek Llama blokları → `sequential_calibrate` → `run_config` → `streamed_perplexity` |
| `7872949` | **Süpürme ızgaranın ortasında 65,536 kodsözcüğü tarıyordu** (§6.13). Üç sabit arasındaki yazılmamış eşitsizlik; 21 hücrenin 8'i. Ağırlıklı **1.25×**, kalite bit-birebir. Ayrıca `bench_guard`: boşluk testi artık alışkanlık değil **assert**, çünkü alışkanlığın baktığı sayı boş kartta %42 okuyor |
| `96f973b` | **`TILE_TIMINGS` `n_tiles` kaydedilerek yeniden ölçüldü** ve modelin **sekizinci hatası** çıktı (§6.14): 4 satırın altında hiç örnek yokmuş. Toplam 15.0 → 14.9 g ama **tepe ortadan ince uca kaydı**, T=1'de duvar geri geldi, Tasarım G/F üçüncü kez yer değiştirdi |

**08-24 oturumunun yayı, tek satırda:** hat 17 günden 12'ye indi (`1efa971`),
sonra modelin iki eksik terimi bulununca gerçeğin ~40 olduğu anlaşıldı
(`8c56f1e`, `98c0413`), ve o terimler düzeltilerek **15 güne** inildi. Aradaki
fark hızlanma değil, **modelin doğrulanması** — ve M1'in koşulup koşulmayacağını
söyleyen sayı o.

---

## 13. Çalışma tarzına dair not

Bu projede en pahalı hata sınıfı **sessizce yanlış bir sayı üretmek**. Bu yüzden:

- Golden sabitler elle yazılmaz, türetilir. `tests/golden.py` `accounting.py`'yi
  **import etmez** — golden değerleri çağıran bir test hiçbir şey kanıtlamaz
- Testlerin çoğu davranış değil **iddia** sınıyor
- Doğrulanmamış şeyler açıkça "varsayım" diye işaretlenir
- Bir hipotez ölçümden **önce** yazılır
- Hız iddiaları **uçtan uca** ölçülür, çekirdek mikro-benchmark'ıyla değil —
  maliyet modeli tam olarak bu yüzden sekiz kez yanıldı
- Bir optimizasyonun kabul kriteri **çıktının değişmemesi**; değişiyorsa
  değişimin ne olduğu ölçülür ve karar tablosuna yazılır
- Aleyhe bulgular da kaydedilir (eşleştirme kazancı 1.16×, ayrılabilirlik
  önyargısı, maliyet modelinin **sekiz** hatası, Triton tahminimin 2–3 kat
  iyimser çıkması, Kronecker'ın sentetik ölçümümün iki mertebe yanılması)
- **Denenip elenen fikirler kaydedilir** (§7) — kaydetmemek ikinci kez denemek

Bu belge de aynı disiplinin parçası: ne bilindiğini, ne bilinmediğini ve neyin
denenip bırakıldığını ayrı tutuyor.

---

## 14. Ölçüm hijyeni — bu oturumun asıl çıktısı

Hız kazançları geçici; bunlar değil. 08-24'te **kendi ölçümlerimin dördü**
sessizce yanlıştı ve ikisini önce yanlış teşhis ettim. Dördünün de ortak yanı
şu: **yanlış cevap vermiyorlar, inandırıcı bir cevap veriyorlar.**

### 14.1 Dört tuzak, dördü de gerçekten vurdu

| tuzak | belirtisi | nasıl yakalandı |
|---|---|---|
| **`_on_device` cihaz-dizesi anahtarı** | Optimizasyon **1.00×** okuyor | Kaba kuvvete düştüğünü fark edince. Önce GPU çekişmesine, sonra saat düşüşüne yordum — ikisi de yanlıştı |
| **Mutlak süreleri koşular arası kıyaslamak** | Değişiklik **+%37** okuyor | Tek süreçte dönüşümlü A/B: −%0.8. Makineymiş |
| **Yanlış rejimde ölçülmüş ret** | "Kazanç yok, tekrar deneme" | Ölçümü gerçek genişliklerde tekrarlayınca: **9.94×**. Kayıt beni sekiz gün yanlış yerde tuttu |
| **Cevabı sınayan test** | Test yeşil, hata duruyor | Yolu saymaya geçince. Aynı oturumda **üç kez** oldu |
| **Hiç koşulmamış bir kompozisyon** | Her parça yeşil, birleşimleri çalışmıyor | 08-25: `sequential_calibrate` + `run_config` **beş** kusur taşıyordu ve parça testlerinin hepsi geçiyordu (§6.12) |
| **Değiştirdiğin yolu geçmeyen ölçüm** | Kalite farkı **tam sıfır** — inandırıcı ve boş | 08-25: yönlendirme değişiminin bedelini 512 satırlık bir katmanda ölçtüm; o katman ölü bandın altında kalıyor, yani iki kolda da **sıfır tarama** oldu ve %0.0000 hiçbir şey kanıtlamadı (§6.13) |

### 14.2 Kurallar

- **Kıyaslama yazarken `assert quantize.is_canonical_codebook(cb)`.** Hızlı
  yolun açık olduğunu iddia et, varsayma. Bu yüzden dışa açıldı.
- **Hız fazından önce GPU'nun boş olduğunu doğrula — ama `utilization.gpu` ile
  DEĞİL.** Çekişme yalnız gürültü eklemiyor; darboğazı karta taşıyıp *tam da
  silmeye çalıştığın gecikmeyi* gizliyor, ve sonuç 1.00× diye okunuyor. Ama bu
  kural 08-25'e kadar **yanlış göstergeyi** işaret ediyordu ve bir kez yanlış
  karar verdirdi: bu makinede `utilization.gpu` boş kartta **%42**, yabancı
  yükte %99, kendi yükümüzde %25 okuyor. `torch.cuda.mem_get_info` ise **kör** —
  3 GiB tutan yabancı bir süreç onu **bir bayt** oynatmıyor (WDDM çağıran
  bağlamın bütçesini veriyor). Çalışan tek gösterge **`clocks.sm`**: boşta
  %42, yabancı yükte %89. Artık elle bakılmıyor —
  `experiments/bench_guard.require_quiet_gpu()` **fırlatıyor**.
- **Yalnız tek süreçte dönüşümlü A/B.** Bu makinede aynı ölçüm koşudan koşuya
  %14–37 oynuyor.
- **Bir reddi kaydederken hangi rejimde ölçüldüğünü yaz.** Yanlış rejimde
  ölçülmüş bir ret hiç ölçmemekten kötüdür: bir sonrakinin bakmasını durdurur.
- **Test cevabı değil YOLU izlesin.** İki doğru algoritma zaten aynı cevabı
  verir — bir hatanın uzun yaşamasının sebebi tam olarak budur.
- **Yeni testi mutasyonla doğrula.** Eski koda karşı kırmızı olduğu
  gösterilmeden kabul etme. Bu oturumda ilk yazdığım testler üç kez gerçek
  hataları kaçırdı: tek çekiliş kullandıkları için, CPU'da `.to("cpu")` no-op
  olduğu için, ve cevabı sınadıkları için.
- **Parçayı değil ÇAĞIRANI test et.** Bir düzeltmenin dokunduğu fonksiyona test
  yazmak yetmiyor: `8c56f1e` `collect_block_statistics`'e CUDA testi ekledi ve
  onu çağıran `sequential_calibrate` üç kusurla kaldı (§6.12). Kusurun yaşadığı
  yer parça değil, **kompozisyon**.
- **Koşulmamış bir kompozisyon çalışmıyor sayılır.** Bu projede iki kez böyle
  oldu: maliyet modelinin iki eksik terimi de, dikişin beş kusuru da, "her parça
  yeşil ama zinciri kimse koşmadı" durumundan çıktı.
- **Yalnız test değil, ÖLÇÜM de yolu izlesin.** "Test cevabı değil yolu
  izlesin" kuralının ölçüm hâli, ve 08-25'te bunu unuttum: bir değişikliğin
  kalite bedelini, değişikliğin hiç tetiklenmediği bir şekilde ölçtüm ve tertemiz
  bir %0.0000 aldım. Bir A/B'de **önce iki kolun gerçekten farklı yollardan
  geçtiğini say**, sonra sayılara bak. Burada sayılacak şey `_brute_force`'a
  düşen satırdı; sıfır/sıfırdı.

### 14.3 Ve modele dair olan

Maliyet modeli sekiz kez yanıldı; **altısı modelin bilmediği şeydi**, oran hatası
değil. Yani
bu modelde sorulacak soru "oran doğru mu" değil, **"listede ne yok"**. Son ikisi
(kalibrasyon, ileri telafi) hiçbir şey patlamadığı için değil, *ne eksik* diye
arandığı için bulundu — ve ikisi birden M1'i 12 günden ~40'a çıkardı.

Neyin saklanmalarına izin verdiği de kayda değer: **tam sürücüyü kimse
koşmadı.** §8.1 yalnız "gerçek veri yok" diye kritik yolda değil; **ölçülmeyen
maliyet de orada birikiyor.**

08-25'te aynı boşluktan ikinci bir şey çıktı, ve bu sefer maliyet değil
**çalışmayan kod**: sürücünün kullanacağı dikiş GPU'da hiç koşmamıştı ve tek bir
yolda beş kusur biriktirmişti (§6.12). Yani "kimse koşmadı" bu projede iki ayrı
şey üretti — **eksik terim** ve **kırık zincir** — ve ikisi de aynı yerde
duruyordu. Kural buradan çıkıyor: bir kompozisyonun test edilmemiş olması, onun
çalıştığına dair kanıtın *yokluğu* değil, **çalışmadığına dair beklenti**.
