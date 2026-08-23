# Durum ve Devir Belgesi

> **Bağlam kaybolduğunda projeye kaldığı yerden devam edebilmek için var.**
> Kod ne yaptığını söyler; bu belge **neden öyle olduğunu** söyler.
> Son güncelleme: 2026-08-23. Testler: **502 geçiyor, 5 atlanıyor.**

---

## 1. Nerede duruyoruz — dört cümle

Hat uçtan uca çalışıyor, gerçek Llama-2-7B'ye bağlı, ve artık **gerçek
ağırlıklar üzerinde iki ölçüm** var: dense perplexity (yayımlanmıştan 0.006
içinde) ve rotasyonun katman düzeyindeki değeri (**−70%**). M0'ın uçuş-öncesi
kalemlerinin dördü kapandı (`vq_bits`, Kapı B'nin gücü, transfer pilotu,
maliyet modeli). **Ama sıkıştırılmış modelin perplexity'si hâlâ hiç ölçülmedi**
— Kapı A'nın ve Kapı B'nin tek bir gerçek verisi yok. Ve önümüzdeki asıl engel
bilimsel değil, **hesaplama maliyeti**: M1 bu makinede **48 gün** (120'den
düştü), ve kalanın neredeyse tamamı tek bir fonksiyonda: `fit_scale`.

---

## 2. Proje 60 saniyede

**Soru.** Yoğun PTQ'nun pratik tabanı ~2 bit; altında çöküyor (QuaRot-GPTQ
2-bit → 22.07 ppl). Seyrekliğin tabanı ise **indeks formatına** bağlı: bitmap
1 bit/pozisyonun altına inemez, ama `T` satırın paylaştığı bir indeks `1/T`'ye
iner. 2 bitin altındaki bütçelerde `(survivor quantizer, granularity, density)`
üçlüsü nasıl seçilmeli?

**Neden önemli.** Bit bütçesi doğrudan bağlam uzunluğudur. Llama-2-70B, 24 GiB
kart: 2.0 bit → ~15.6k bağlam, 1.5 bit → ~28.4k. 2 bitin altı, bağlamı ikiye
katlamak demek.

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
| **`vq_bits = 2.0`'ın maliyet tarafı** | QuIP# E8P ve QTIP release'lerinin manifest'i. Yük **tam 2.000000**, yan bilgiyle 2.005204 / 2.006740. Manifest ve dosya boyutu iki bağımsız yol, tam aynı sayı |
| **Kapı B'nin istatistiksel gücü** | 600 denemelik simülasyon, **gerçek `gate_b` çağrılarak**. 5 çekiliş 2.29 σ saptıyor; ölçülen etki 6.7 σ. Tip-I her yerde %5'in altında |
| **Transfer sapması** | `Δ = Q + τ` tahmin edicisi kurulup gerçek hattın yanında koşuldu. `T=1` kimlik kontrolü **tam sıfır**; sapma çekiliş gürültüsünü **12.3×** aşıyor |
| **Rotasyonun değeri (gerçek katman)** | `o_proj`, gerçek ağırlıklar + 32,768 gerçek token: **−70.1% ortalama** (§5) |
| **Kafes çözücünün taramaya denkliği** | `nearest_e8p` ile kaba kuvvet, dört ölçekte **birebir aynı indeks** |
| **Akıtılan alt-Hessian'ın yığılmışa denkliği** | Bit-birebir aynı çıktı |
| **Blok genişliğinin kaliteye etkisi** | `o_proj`, 51 kol, 5 genişlik × 3 aile × 3 tile. Geri besleme 512'de daraltılınca kalite **iyileşiyor**, rotasyon daraltılınca bozuluyor (§5.9) |
| **`hessian_block=b`'nin ne yaptığı** | Hessian'ın blok-köşegen parçasına karşı LDLQ'ya **tam denk**; hiçbir hata blok sınırını geçmiyor (testle sabitlendi) |
| **Maliyet modelinin Cholesky eğrisi** | k'ye bağlı, ölçülmüş, iki kernel de ısıtılarak. Model bir tile'ın gerçek süresini artık hiç uydurulmadığı bir genişlikte (k=7912) %14 içinde tahmin ediyor |

### Varsayım — doğrulanmadı

> **E8P'nin kompaktlanmış survivor alt-matrisinde 2 bit KALİTESİNİ koruduğu.**
> Survivor'lar tanım gereği dağılımın kalın kuyruğu; kafes quantizer Gauss'a
> yakın girdi ister.

Maliyet tarafı 2026-08-21'de kapandı (tam 2 bit ödendiği kesin). Açık olan
**karşılığında 2 bitlik kalite alınıp alınmadığı.** Bu varsayımı sınayacak ucuz
deney bilinçli olarak atlandı (kullanıcı kararı, 2026-08-20).

Rotasyonun gerçek katmanda −70% çıkması bu varsayımı **dolaylı olarak
güçlendiriyor** — varsayımın gerekçesi "rotasyon kalın kuyruğu düzeltir"di ve
rotasyonun gerçekten çalıştığı artık ölçüldü. Ama doğrudan kanıt değil.

**Erken uyarı kuralı:** ilk katman E8P'den geçtiğinde katman-çıkışı MSE'si dense
E8P referansının 2 katını aşarsa varsayım düşmüş sayılır; geri dönüş yolu
rotasyon + GPTQ-3bit (`W=3.148`), bant 1.83–2.83'e kayar.

### Henüz hiç ölçülmemiş

**Sıkıştırılmış modelin perplexity'si.** Kapı A ve Kapı B'nin **hiçbir gerçek
verisi yok**. Sentetik smoke testte hata eğrisi U şeklinde çıkıyor ve Kapı A
geçiyor — **ama veriyi biz ürettik, bu tez lehine kanıt değil.**

---

## 4. Alınan kararlar ve gerekçeleri

| Karar | Tarih | Gerekçe |
|---|---|---|
| Survivor quantizer **GPTQ-4bit → QuIP# E8P** | 08-20 | Kapı A provası: GPTQ-4bit survivor'larla literatürün konuşabildiği her yerde kaybediliyor. `W` 4.156 → 2.000, `B=1.5`'te `T=16` seyrekliği %65 → %28 |
| Ucuz E8P doğrulama deneyi **atlandı** | 08-20 | Kullanıcı kararı; risk §3'te açık varsayım olarak taşınıyor |
| Bant **1.75 / 1.60 / 1.50** | 08-20 | E8P'nin canlı bandı 1.40–1.80; çalışma kendiliğinden 2 bitin altına kaydı — motivasyonun tuttuğu yere |
| Çapa **QTIP/QuIP#**, GPTQ değil | 08-20 | GPTQ 3-bit sınıfın en zayıfı; ona çapalanırsa Kapı A kolay geçer ama savunulamaz |
| **LDLQ eklendi** | 08-20 | Rotasyon, Hessian-farkında yuvarlama olmadan maliyeti ödeyip faydasını toplamıyordu |
| Kapı B için **≥5 çekiliş** | 08-20 | 3 seed ile `gate_b` saf gürültüde "interior" verdi. Spec §6'nın "seed ≥ 3"ü bu kapı için yetersiz |
| Checkpoint: **NousResearch aynası** | 08-21 | Resmi repo kapılı; ayna kapısız ve dense ppl ölçümü ağırlıkların doğruluğunu zaten teyit etti |
| **seqlen 4096 birincil** | 08-21 | `dense-5.12` ailesi hem budama baseline'larını hem QTIP/QuIP#'i taşıyor; Kapı A'nın rakibi orada |
| Izgara **`vq_bits = 2.0`'da donduruldu**, düzeltme rapora eklenir | 08-21 | Düzeltme her hücrede aynı göreli miktarda (%0.26) — bütçe-eşleşmesini bozmuyor. Buna karşılık tam 2, `B=1.5` ızgarasını tam dyadic yapıyor ve `golden.py`'nin bağımsız türetmesi buna dayanıyor |
| Tolerans kuralı **`1.5 × max_T \|sapma(T)\|`** | 08-21 | Seed varyansından türetilseydi **12.3 kat** küçük olurdu ve ön-kayıt tanım gereği "tutmadı" dalına kilitlenirdi (denetim §B3) |
| `T*` **nokta değil küme** olarak raporlanır | 08-21 | Verdikt ile `T*` aynı güvenilirlikte değil: düz iç bölgede 20 çekilişle verdikt %77, argmin %41 |
| Çekiliş ekseni **kalibrasyon**, rotasyon seed'i değil | 08-21 | Ölçüldü: kalibrasyon gürültüsü rotasyon gürültüsünün **1.95 katı**. Öyle koşulsa Kapı B iki kat fazla kendinden emin çıkardı |
| Hat **cuda/float32**'ye taşındı | 08-23 | Uçtan uca **16–45×**, ağırlık farkı 5e-08 — float32'nin kendi epsilon'u düzeyinde |
| **Katman-başı ölçek reddedildi** | 08-23 | Ölçüldü: **%11 kalite kaybı** ve bu ölçekte hız kazancı yok. %70'lik bir etkiyi ölçerken hatta %11 bozulma sokmak ölçümü kirletirdi. Politika kodda duruyor (`scale=`), M1 için bir kaldıraç, bu deney için yanlış araç |
| **E8 kafes çözücü** kaba kuvvetin yerine | 08-23 | Baskın terim buydu. CPU 3.51×, GPU 1.87×, çıktı birebir aynı |
| **`hessian_block=512`**, rotasyon tam genişlikte | 08-23 | Ölçüldü: geri beslemeyi 512'ye daraltmak her tile boyutunda kaliteyi **iyileştiriyor** (−11 / −23 / −16%), rotasyonu daraltmak bozuyor (+43 / +38 / +44%). Bedava alınan kalite; hız katkısı ikincil (§5.9) |
| Maliyet modelinin Cholesky eğrisi **ölçülerek düzeltildi** | 08-23 | Isınmayan benchmark + k'ye bağımlılık, birlikte 9.4× fazla yazıyordu. M1 120 → 94 gün. Bu sayı M1'in koşulup koşulmayacağını söyleyen sayı (§6.4) |

---

## 5. Planı değiştiren bulgular

Kronolojik değil, önem sırasına göre.

### 5.1 ⭐ Rotasyon gerçek katmanda sandığımızdan çok daha değerli

`layers.0.self_attn.o_proj` (512 çıktı satırı), gerçek Llama-2-7B ağırlıkları,
32,768 gerçek kalibrasyon token'ı, `B=1.5`, cuda/float32:

| T | d | düz | rotasyonlu | **değişim** | sentetik hattın dediği |
|---|---|---|---|---|---|
| 4 | 0.6250 | 0.47422 | 0.09649 | **−79.7%** | −29.5% |
| 16 | 0.7188 | 0.54423 | 0.19530 | **−64.1%** | −31.0% |
| max | 0.7500 | 0.55738 | 0.18655 | **−66.5%** | — |

Ortalama gerçek **−70.1%**, sentetik −30.2%. SNR farkı her hücrede ~3.3 dB.

> **Bir çerçeveyi çürüttü.** "Sentetik bir kazanç için yapısal bir bedel
> ödüyoruz" diyordum. Kazanç sentetik değil — **sentetik olan, kazancı iki-üç
> kat eksik ölçmüş.** Fixture'ın kuyruğu gerçek ağırlıklarınki kadar ağır değil,
> ve rotasyon tam olarak o kuyruğu yaymak için var.

**Sonuç: "rotasyonu bırak" seçeneği kapandı.** Bırakmak katman hatasını üçe
katlıyor.

**Kapsam:** tek katman, tek çekiliş, katman-çıkışı hatası — perplexity değil.
Mekanizmaya dair yeter, manşete dair yetmez.

### 5.2 §0.5 tersine döndü

v6 incoherence processing'i en büyük risk sayıp QuIP#/QTIP'i toptan eliyordu.
Eleme fazla genişti: maske dondurulduktan sonra rotasyon onu bozamaz. Belgelenen
çöküş bir **sıra** problemi. Bu, E8P'ye geçişin kapısını açan adımdı.

### 5.3 Rotasyonun değeri LDLQ'dan değil, dağılımdan geliyor

Önce "rotasyonun faydası tamamen Hessian-farkındalığa bağlı" dedim, fazla
genelmiş. İzole ölçüm (16×64 blok, korelasyonlu Hessian):

| blok | rotasyon, düz | rotasyon, LDLQ |
|---|---|---|
| Gaussian | +17.5% (zarar) | +4.8% (zarar) |
| kalın kuyruklu | **−61.7%** | **−39.0%** |

LDLQ yine de zorunlu: onsuz rotasyon maliyeti ödeyip faydasını toplamıyor
(hat ölçümü: T=4'te +2.6% → −29.5%).

### 5.4 SU ve SV aynı şey değil

QuIP#'in yan bilgisini ölçerken çıktı: `SU` (girdi ekseni) ince ayarın ±1'den
zar zor kıpırdattığı bir işaret vektörü; `SV` (çıktı ekseni) gerçek kanal-başı
ölçek. QTIP'te asimetri daha da keskin.

Önemi: tile başına **öğrenilmiş** bir sütun vektörü `16/T` bit demekti (T=16'da
1.0 — bant kaldırmaz), paketlenmiş işaret olarak `1/T` (0.0625). Ölçüm bunu
ödemek zorunda olmadığımızı gösterdi: köşegen gather ile yer değiştirdiği için
girdi ekseni köşegeni **global** tutulabilir; tile başına kalan tek şey
rotasyonun kendisi, o da seed'den üretilirse yük taşımıyor. Ayrıştırılmış
tasarımda **0.0077 bit/survivor**, `T` ile neredeyse sabit
(`accounting.rotation_side_bits`).

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
argmin değil, argmin'den ayrılamayanların kümesi. Duman testinde Kapı B
"interior" derken küme `{2, 4, 8, 16}` — dürüst manşet "T=16 optimal" değil,
"optimum içeride, yeri 2–16 arasında".

### 5.9 Daraltılabilen şey geri besleme, rotasyon değil

§6.3'ün "rotasyonu 8'lik gruplara blok-köşegen kısıtla" önerisi iki yönden
yanlış çıktı ve ölçüm ikisini de düzeltti.

**Önce öncül.** `rotate` zaten `share_across_tiles=True` ile **tek bir
rotasyonu bütün katmana** uyguluyor. Yani rotasyon, LDLQ'nun tile başına
faktorize etmesinin sebebi değil — sebep tile başına farklı sütun kümesi, ve
onu hiçbir rotasyon genişliği değiştirmiyor. Rotasyonu tamamen atsak `k³`
terimi yerinde durur.

**Sonra genişlik.** Maliyet eğrisi 512'de düzleşiyor; 512'den 8'e inmek toplam
tasarrufun %1.9'unu ekliyor ama atılan Hessian bağlantısını 64 katına çıkarıyor.
8, en az kazanç için en çok kaliteyi harcayan noktaydı.

Ölçüm (`o_proj`, 512 satır, 32,768 token, B=1.5), `full`'e karşı:

| genişlik | R (rotasyon daraltıldı) | H (geri besleme daraltıldı) |
|---|---|---|
| 2048 | +12.8 / +12.8 / +22.3% | −8.6 / −9.2 / −2.0% |
| 1024 | +25.3 / +13.4 / +23.5% | −7.1 / −16.7 / −6.8% |
| **512** | +43.0 / +38.1 / +44.3% | **−11.1 / −23.4 / −15.8%** |
| 128 | +117 / +69 / +95% | +13.4 / −20.6 / −11.7% |
| 8 | +375 / +169 / +192% | +147 / +49 / +57% |

*(üç sayı = T=4 / T=16 / T=max)*

**`H512` her tile boyutunda bütün ızgaranın en iyi kolu** — hem tam rotasyondan
hem düz yuvarlamadan iyi, ve Cholesky'nin 1/25–1/36'sı. `R8` neredeyse
rotasyonsuza eşit (−3.3%): rotasyonu daraltmak mekanizmayı yok ediyor.

Mekanizma iki tarafta da tutarlı. Rotasyonun işi §5.3'te kurulduğu gibi kalın
kuyruğu **olabildiğince geniş** yaymak; 8 koordinat içinde döndürmek o
koordinatların normunu değiştiremez, yalnızca yönünü — ve tek E8P ölçeğinin
kapsaması gereken şey tam da gruplar arası norm yayılımı. Geri besleme tarafında
ise 2560×2560'lık alt-Hessian 32,768 token'dan kestiriliyor; uzun menzilli
bağlantılar gürültülü, onları atmak düzenlileştirme gibi davranıyor.

**Sonuç: rotasyon tam genişlikte kalır, geri besleme 512'ye daraltılır.** Ama
kazanç hız değil kalite — §6.4'ün düzeltmesinden sonra Cholesky zaten pasın
%10'u, yani bu M1'in 94 gününden 8 gün götürüyor.

### 5.7 VENOM'un `V`'si bizim `T`'miz

V:N:M formülü VENOM'dan dolduruldu ve yapısal bir şey çıktı: VENOM `V` satırın
paylaştığı bir sütun seçimi kullanıyor, yani indeksi `1/V` ile amortize ediyor.
**İndeks amortizasyonu yeni değil.** Katkı "amortize etmek" değil, `(T, d)`
düzlemini bir bütçe altında taramak. Özgünlük iddiası buna göre daraltıldı.

### 5.8 Protokol ayrımı dizi uzunluğuymuş

Ölçümden önce hipotez olarak kaydedildi, ölçümde tuttu. Kural "birini seç"
değil **"pencereyi sabitle"**.

---

## 6. Maliyet gerçeği — projenin şu anki asıl engeli

Bilimsel sorular büyük ölçüde yerinde; tıkanan şey hesaplama.
`experiments/m0_cost_model.py`, **bu makinede uçtan uca ölçülen** tile
sürelerine oturuyor (kernel mikro-benchmark'larına değil — onlar üç kez
iyimser yönde yanılttı, §6.2).

### 6.1 Bugünkü tablo (cuda/float32, bir sıkıştırma geçişi)

**08-23'te iki kez değişti:** modelin Cholesky eğrisi düzeltildi (§6.4) ve hat
hızlandırıldı (§6.5). Sıra: 120 → 94 → **48 gün**.

Aşağıdaki tablo hattın **şu an koştuğu** düzen: geri besleme 512'ye daraltılmış,
süpürme tile'lar arasında toplu.

| T | toplam | codebook | rotasyon | cholesky |
|---|---|---|---|---|
| 1 | 5.3 h | 4.4 h | 0.46 h | 0.34 h |
| 4 | **13.2 h** | 11.0 h | 1.92 h | 0.21 h |
| 16 | 9.1 h | 8.3 h | 0.72 h | 0.06 h |
| max | 7.5 h | 7.4 h | 0 | 0 |

**M1 (3 bütçe × 7 tile × 5 çekiliş): 48 gün.**

**Baskın terim Cholesky değil, codebook taraması** — ve onun da içinde
`fit_scale`. Cholesky artık pasın %2'si; kalan neredeyse tamamen ölçek uydurma,
ve onu düşürmenin tavanı **14 gün**. `affordable()` bir zaman bütçesine neyin
sığdığını hesaplıyor ve `T=1` ile `T=max`'ı asla düşürmüyor (onlar kapının
tanımı).

### 6.5 Süpürmeyi tile'lar arasında toplamak (08-23)

Süpürme hesap-bağımlı değildi: grup başına **0.248 ms** duvar saati, **0.0034 ms**
aritmetik — %99.6 boşta. Sebep, `[lines, 8]`'lik bir tensörün 65,536 kodsözcüğüne
bakması: GPU'yu dolduramayacak kadar küçük, ve arka arkaya `k/8` tane var.

Tile'lar kendi Hessian'ları verildiğinde bağımsız olduğundan grup döngüsü tile
döngüsünün dışına alındı: her grupta `C` tile birlikte quantize ediliyor.
Süpürmedeki kazanç:

| şekil | chunk | hızlanma |
|---|---|---|
| k=2944, 16 satır | 64 | 6.95× |
| k=7912, 16 satır | 16 | 4.55× |
| **k=2560, 4 satır** (T=4) | 64 | **12.1×** |

En büyük kazanç `lines=4`'te, yani ızgaranın en pahalı sütununda.

**Uçtan uca kazanç çok daha küçük: 2.07× / 1.43× / 1.06×.** Çünkü süpürme artık
bir tile'ın zamanını harcadığı yer değil — `fit_scale` orada, ve o toplu
değil. Bu, kaldıracı Faz 2'ye taşıyor.

`hessian_block=512` bunun **önkoşulu**: geri besleme daraltılınca `U` tile başına
`k×512` tutuluyor, `k²` değil (k=7912'de 250 MB → 16 MB), ve `C=64` ancak böyle
VRAM'e sığıyor. Dün kalite için alınan karar bugün hızın önkoşulu çıktı.

**Çıktı bit-birebir aynı** — iki cihaz, iki dtype, iki ölçek politikası, chunk
2'den tile sayısının 4 katına kadar `torch.equal` ile doğrulandı ve teste
sabitlendi. Hızlanma bir sayıyı değiştirseydi o farklı bir hat olurdu, hızlı bir
hat değil.

### 6.2 Kapatılan duvarlar

| Duvar | Neydi | Nasıl kapandı |
|---|---|---|
| **Bellek** | `T=2`'de **462 GiB** tek tensör | `tile_hessian_stream` → 239 MiB. Yığılmış yolla **bit-birebir** aynı |
| **Yanlış cihaz** | Hat CPU/float64'te koşuyordu | cuda/float32, uçtan uca **16–45×**, ağırlık farkı 5e-08 |
| **Kaba kuvvet codebook araması** | Baskın terim | `nearest_e8p` kafesi çözüyor: CPU 3.51×, GPU 1.87×, çıktı birebir aynı |

**Kafes çözücünün inceliği** (tekrar etmemek için): E8P **tam bir kafes değil**
— `D8 + ½`, `±¼` kaydırılmış, norm topuyla kesilmiş, artı lexicographic seçilmiş
29 dolgu örüntüsü. "En yakın kafes noktasına yuvarla" tek başına yanlış cevap
verebilir. Kesinlik kanıtı: codebook iki kaydırılmış kafesin **birleşiminin
içinde**, o birleşimin en yakın noktası herhangi bir kodsözcüğüne mesafenin
alt sınırı; o nokta kodsözcüğüyse en yakındır. Değilse satır taramaya düşüyor.
Kesinlik oranı çalışma ölçeğinde **%72**. Üyelik testi arama değil gather:
`|h_i| ∈ {0.5, 1.5, 2.5}` olduğundan bütün örüntü uzayı 3⁸ = 6561 giriş.

Çözücünün **sabit maliyeti** var, küçük batch'lerde kaybediyor (16 satırda CPU
0.63×, GPU 0.11×). Ölçülen geçiş noktasına bağlı: `fit_scale`'in tüm-tile
taramaları hızlı yolda, LDLQ'nun grup-başı çağrıları taramada.

### 6.3 Kalan duvar ve seçenekler

**Duvar ölçek uydurma, Cholesky değil.** Bir tile'ın `ldlq_quantize` çağrısı
önce `fit_scale`'i koşuyor, o da 24 aday ölçeği tile'ın **bütün** vektörleri
üzerinde tarıyor — pasın %84'ü. Faktorizasyon %10.

| seçenek | ölçülen | durum |
|---|---|---|
| `hessian_block=512` — geri beslemeyi blok-köşegen yap | M1 94 → **86 gün**, ve kalite **iyileşiyor** (§5.9) | **ÖLÇÜLDÜ, alınmalı.** Bedava; tek gerekçesi artık hız değil kalite |
| Tile-başı `fit_scale`'i küçük örnekle | Tavan: M1 94 → **28 gün** | **KALİTE ETKİSİ ÖLÇÜLMEDİ** — bir sonraki iş |
| Donanım kirala | A100/H100 fp32'de ~10–20× | Birkaç yüz dolar |

**Ölçek örneklemesi hakkında bir düzeltme.** Bu belge daha önce "68 gün =
örneklenmiş ölçek" diyordu; öyle değil. `scale_fit=False` ölçeği **tamamen**
kaldırmayı fiyatlıyor, bir **tavan**. `fit_scale(sample=N)` bir tile'ın sahip
olduğundan fazla vektöre bakamaz ve B=1.5'te tile başına vektör sayısı T=1'de
128, T=4'te 1,280, T=16'da 5,888 — varsayılan 8,192 sınırının **altında**. Yani
varsayılan örnekleme, maliyetin yaşadığı her tile boyutunda **tam olarak
etkisiz**; yalnızca zaten en ucuz sütun olan T=max'te ısırıyor. Isıracak
sınırlar (≈256) yeterince küçük ki kalite bedeli varsayılamaz, ölçülmeli.
Kontrol koda girdi: `m0_cost_model.scale_sample_bites`.

### 6.4 Maliyet modelinin dördüncü hatası — ve ilk kez kötümser olanı

08-23'te ölçüldü: modelin Cholesky terimi gerçek genişliklerde **9.4×** fazla
yazıyordu. İki bağımsız sebep birleşiyordu:

- `cholesky_rate()` yalnız `torch.linalg.cholesky`'yi ısıtıyor, `cholesky_inverse`'ü
  ısıtmıyordu; ilk ölçülen tekrar cuSOLVER handle kurulumunu ödüyordu → **1.6×**
- Tek bir flop/s sayısı bu çekirdeği tanımlamıyor. k=1024'te kart dolmuyor,
  k=8192'de neredeyse doluyor; etkin hız 5.7e11 → 3.8e12, yani **6.8×** değişiyor.
  k=2048'de ölçülen hızı her genişliğe uygulamak en geniş katmanları **2.6×**
  fazla yazıyordu — ve `down_proj` T=16'da k=7912

İkisi de düzeltildi (`CHOL_TIMINGS`, k'ye bağlı ölçülen eğri). Ayrıca Cholesky
küçülünce **rotasyonun kendisi ondan büyük hâle geldi** (k=7912'de 0.25s'e karşı
0.22s), o yüzden `q @ H_t @ q.T` de artık ayrı bir kalem (`ROT_TIMINGS`).

Bu, modelin dördüncü hatası ve **ilk kez iyimser değil kötümser** olanı. Önemi:
bu sayı M1'in koşulup koşulmayacağını söyleyen sayı. 120 gün diyordu, doğrusu
94.

---

## 7. Sırada ne var

### Bir sonraki oturumun ilk işi

**Tile-başı `fit_scale` örneklemesinin kalite bedelini ölç.** Maliyetin %84'ü
orada ve tavanı M1'i 94 günden 28'e indiriyor — kalan tek büyük kaldıraç bu.
Kurulum hazır: `m0_rotation_value.py`'nin kol yapısı aynen kullanılabilir,
`sample ∈ {2048, 512, 256, 128}` süpürülür. Dikkat: örnekleme ancak tile'ın
vektör sayısından küçük sınırlarda ısırıyor (§6.3), o yüzden süpürme küçük
sayılarda olmalı. Bir de `fit_scale`'in `n_steps=24`'ü var — 24 aday ölçek,
hiç sorgulanmadı; 8 aday muhtemelen aynı α'yı buluyor ve 3× getiriyor.

`ldlq_quantize(..., scale_sample=N)` için tile-başı yolda **tesisat henüz yok**
— `scale_sample` şu an yalnız `per_layer` dalında kullanılıyor. Küçük iş.

### Ön-kaydı dondurmak için kalan iki kutu

İkisi de `tau_sweep.py`'ye bağlı: **`Δ(T)` tahmin eğrisi** ve **`T*_tahmin`**.
Süpürme 29 gün olduğu için ikisi de bloke. Maliyet düşmeden ön-kayıt donmaz,
ön-kayıt donmadan M1 başlamaz.

### Sonra

- **M1** — iki kapı, `B ∈ {1.75, 1.60, 1.50}`, `T ∈ {1,2,4,8,16,32,max}`
- **M2** — bütçe süpürmesi, sıra ablasyonu (prune-then-quant vs joint)

### Açık kalan kod işleri

- **Axis A için LDLQ** — şu an `NotImplementedError`; Axis B'de indeks ekseni
  girdi kanalları olduğu için Hessian doğrudan uygulanıyor, Axis A'da sweep
  tile'ın sütunları boyunca olmalı
- **Blockwise (tam SparseGPT) maske seçimi** — M3 teslimatı, şu an `upfront`
- **§3.6'nın üç ablasyonu** — grup konvansiyonu (iki koşullu olmalı),
  quantization/maske hatası ayrımı, hizalama
- **Attention koordinasyonu formülü** — `v_proj`↔`o_proj`, GQA, RoPE çiftleri;
  `T=max` için sert kısıt, hâlâ yalnızca ima edilmiş
- **`fit_scale` örneklemesinin kalite etkisi** — §7'nin ilk işi; tile-başı yol
  için tesisat da yazılacak (`scale_sample` şu an yalnız `per_layer`'da)
- **`fit_scale(n_steps=24)`** — aday ölçek sayısı hiç sorgulanmadı

---

## 8. Açık riskler

**Kapı A'nın düşme olasılığı yüksek.** Prova (`gate_a_dry_run.md`) GPTQ-4bit
survivor'larla her satırın düştüğünü gösterdi. E8P aritmetiği değiştiriyor ama
**gösterilmedi**. Karar tablosunun `✗/✓` dalı hazır: proje durmaz, çerçeve
daralır.

**E8P kalite varsayımı** (§3). Düşerse bant 1.83–2.83'e kayar ve tezin "2 bitin
altı" motivasyonu zayıflar.

**`T*`'ın belirsizliği.** Verdikt tarafı çözüldü; eğri iç bölgede düzse küme
büyük çıkar ve *hangi* granülerlik sorusu cevapsız kalır. Başarısızlık değil
ama manşeti zayıflatır.

**Maliyet.** 94 gün bir dizüstünde koşulamaz. Ölçek kaldıracı tutarsa 28 güne
iner ve kiralık bir A100'de günlere; tutmazsa proje ya donanım kiralamaya ya da
ızgarayı daraltmaya mecbur — ve daraltmak tam da kanıtın olduğu yeri budar.

**Maliyet modeli dört kez yanıldı.** Üçü iyimser, sonuncusu kötümser yöndeydi
(§6.4). Bu belgedeki her gün rakamı ölçülmüş bir eğriye dayanıyor ama eğriler
bu makineye ait; başka donanımda yeniden ölçülmeli.

**Sentetik σ.** Kapı B'nin gücü ve transfer toleransı sentetik katmandan
ölçüldü. Gerçeği ilk M1 bütçesinden gelecek; ön-kayıt §7.4'ün uyarlanabilir
kontrolü bunun için var.

---

## 9. Ortam tuzakları — saatlere mal oldu, tekrar etmesin

| Sorun | Çözüm |
|---|---|
| **HF indirmeleri takılıyor** (0 B/s) | `HF_HUB_DISABLE_XET=1`. Xet arka ucu bu ağda çalışmıyor |
| **Kimliksiz HF istekleri sert kısıtlanıyor** | `hf auth login` (diske yazar, her süreç görür). `$env:HF_TOKEN` yalnızca o pencerede geçerli |
| **`snapshot_download` oturumlar arası devam ETMİYOR** | Bir kez başlat, kesme |
| **Arka plan görev bildirimleri güvenilmez** | Wrapper çıkışı işin bitişi değil. Log dosyasına veya süreç listesine bak |
| **`torchvision` ABI uyumsuzluğu transformers'ı komple kırıyor** | torch'u yükseltirken eşleştir, ya da kaldır |
| **`load_dataset("wikitext", ...)` reddediliyor** | `Salesforce/wikitext` — `namespace/name` gerekiyor |
| **Süreç sayarken kendi ölçüm sürecini sayma** | PowerShell filtresini `python -c` içinden çağırınca kendini yakalıyor |
| **`e8p_codebook(dtype).to(device)` her çağrıda 2 MB kopyalıyor** | `_on_device(dtype, device)` — cihaz-başı önbellek. Kopya, hızlı yolu seçen kimlik kontrolünü GPU'da hep yanlış yapıyordu ve bir ölçümü tamamen yanılttı |
| **Python stdout tamponu arka plan koşularında** | `python -u`, yoksa ilerleme satırları sonuna kadar görünmüyor |

**Donanım:** RTX 5060 Laptop, 8 GB VRAM, sm_120 (Blackwell → cu128+;
`torch 2.12.0+cu130` kurulu). 23.7 GiB RAM, 16 torch thread'i.
7B fp16 (13.5 GB) GPU'ya sığmıyor → **katman-akışlı zorunlu**, ~2.8 GB tepe.

---

## 10. Repo haritası ve çalıştırma

| Modül | İş |
|---|---|
| `accounting.py` | bit bütçeleri, `1−1/T`, `B*`, canlı bant, V:N:M, `rotation_side_bits` |
| `scoring.py` | saliency — iki ağırlık-başı metrik, iki toplama yönü |
| `tiling.py` | tile bölümlemesi, dondurulmuş maske, `align` |
| `prune.py` | maske seçimi + ileriye telafi; **H1 assert'i burada** |
| `compact.py` | survivor'ları tile başına yoğun bloklara topla |
| `rotation.py` | maske-koruyan rotasyon |
| `quantize.py` | E8P codebook, **kafes çözücü**, LDLQ, ölçek politikası |
| `calibrate.py` | sıralı kalibrasyon, `LayerProblem` (**dikiş yeri**) |
| `hf_llama.py` | HF adaptörü — blok 0 girdilerini **yakalar**; `to_device` |
| `eval/perplexity.py` | ppl + protokol koruması + yayımlanmış sayı tablosu |
| `eval/streamed.py` | katman-akışlı ppl |
| `experiments/m1_gates.py` | M1'in iki kapısı, `t_star_set`, çekiliş ekseni |
| `experiments/m0_dense_ppl.py` | dense ölçüm + protokol kimliği |
| `experiments/m0_vq_bits.py` | VQ checkpoint maliyeti — manifest'ten, indirmeden |
| `experiments/m0_gate_b_power.py` | Kapı B'nin gücü + hattın gürültüsü |
| `experiments/m0_transfer_pilot.py` | `Δ = Q + τ` transfer sapması → tolerans |
| `experiments/m0_cost_model.py` | ölçülen tile sürelerinden gerçek koşu maliyeti |
| `experiments/m0_rotation_value.py` | rotasyon gerçek katmanda kazandırıyor mu; blok genişliği süpürmesi |

**Belgeler:** `spec_v7.md` (şartname) · `preregistration.md` (M1 ön-kaydı,
**dondurulmadı** — iki kutu kaldı) · `audit.md` (v6 denetimi, tarihsel kayıt) ·
`gate_a_dry_run.md` (literatür provası) · bu belge.

```bash
python -m pytest tests/ -q                         # 502 test, ~85 s
HF_HUB_DISABLE_XET=1 python experiments/m0_dense_ppl.py --seqlens 2048 4096 --device cuda
HF_HUB_DISABLE_XET=1 python -u experiments/m0_rotation_value.py \
    --tiles 4 16 max --seqs 16 --rows 512 --solve-device cuda --solve-dtype float32
python experiments/m1_gates.py --synthetic --n-out 64 --n-in 128 --budgets 1.5 --draws 5
python experiments/m0_gate_b_power.py --no-noise   # simülasyon (~15 dk), σ önbellekten
python experiments/m0_transfer_pilot.py --draws 3  # ~8 dk; --reuse ile saniyeler
python experiments/m0_cost_model.py                # ~1 dk, sabitler önbelleklenir
python experiments/m0_vq_bits.py --all             # ~100 KB ağ, saniyeler
```

---

## 11. Commit geçmişi — ne anlama geliyorlar

| Commit | Ne getirdi |
|---|---|
| `5d7726d` | Hattın tamamı: muhasebeden ppl'e |
| `f94a8af` | Ön-kayıt taslağı; dondurma listesi görünür bir olay olsun diye |
| `6af48d2` | v6 denetimi repoya taşındı — kararların gerekçesi versiyonlansın |
| `94dbdce` | Spec v7: kafes VQ etrafında yeniden kuruldu, 4 aritmetik hata düzeldi |
| `2e3e7fc` | README — doğrulanan/varsayılan ayrımı etrafında |
| `c3c5632` | V:N:M formülü VENOM'dan; **özgünlük iddiasını daralttı** |
| `1e6218f` | HF adaptörü |
| `33d66a4` | Katman-akışlı eval — 7B'yi 8 GB'da ölçmenin yolu |
| `3ee1628` | M0 dense ölçüm betiği (hipotez ölçümden önce kaydedildi) |
| `d80ab14` | **İlk gerçek ölçüm**: protokol sorusu çözüldü |
| `e5ec362` | Bu belgenin ilk hâli |
| `a1626c6` | VQ maliyeti checkpoint'ten ölçüldü; SU/SV ayrışması bulundu |
| `3d8658f` | Kapı B'nin gücü ölçüldü; `T*` küme oldu; çekiliş ekseni düzeldi |
| `7d1ee48` | Transfer pilotu: tolerans kuralı, ve modelin büyük `T` önyargısı |
| `797aa2e` | Maliyet modeli — hattın gerçek boyutta koşamadığının tespiti |
| `baa38a7` | Bellek duvarı kapandı, iki yükleyici hatası düzeldi, `fit_scale` modele girdi |
| `31f9761` | **Rotasyon gerçek katmanda −70%**; hat GPU'ya taşındı |
| `0201f93` | E8 kafes çözücü: CPU 3.5×, GPU 1.9×, çıktı birebir aynı |
| `f425880` | Bu belge, bilinenin etrafında yeniden yazıldı |
| *(bu oturum)* | Blok genişliği süpürmesi: **geri besleme daraltılır, rotasyon daraltılmaz**; maliyet modelinin Cholesky eğrisi düzeltildi (120 → 94 gün) |

---

## 12. Çalışma tarzına dair not

Bu projede en pahalı hata sınıfı **sessizce yanlış bir sayı üretmek**. Bu yüzden:

- Golden sabitler elle yazılmaz, türetilir. `tests/golden.py` `accounting.py`'yi
  **import etmez** — golden değerleri çağıran bir test hiçbir şey kanıtlamaz
- Testlerin çoğu davranış değil **iddia** sınıyor
- Doğrulanmamış şeyler açıkça "varsayım" diye işaretlenir
- Bir hipotez ölçümden **önce** yazılır (protokol hipotezi böyle sınandı;
  `m0_rotation_value.py` sentetik referansı aynı sebeple gömülü tutuyor)
- Kesik/eksik ölçümlerden iddia üretmek koda gömülü olarak engellenir
- Aleyhe bulgular da kaydedilir (eşleştirme kazancı 1.16×, ayrılabilirlik
  önyargısı, maliyet modelinin üç kez yanılması)

Bu belge de aynı disiplinin parçası: ne bilindiğini ve ne bilinmediğini ayrı
tutuyor.
