# Durum ve Devir Belgesi

> **Bağlam kaybolduğunda projeye kaldığı yerden devam edebilmek için var.**
> Kod ne yaptığını söyler; bu belge **neden öyle olduğunu** söyler.
> Son güncelleme: 2026-08-23. Testler: **452 geçiyor, 5 atlanıyor.**

---

## 1. Nerede duruyoruz — dört cümle

Hat uçtan uca çalışıyor, gerçek Llama-2-7B'ye bağlı, ve artık **gerçek
ağırlıklar üzerinde iki ölçüm** var: dense perplexity (yayımlanmıştan 0.006
içinde) ve rotasyonun katman düzeyindeki değeri (**−70%**). M0'ın uçuş-öncesi
kalemlerinin dördü kapandı (`vq_bits`, Kapı B'nin gücü, transfer pilotu,
maliyet modeli). **Ama sıkıştırılmış modelin perplexity'si hâlâ hiç ölçülmedi**
— Kapı A'nın ve Kapı B'nin tek bir gerçek verisi yok. Ve önümüzdeki asıl engel
bilimsel değil, **hesaplama maliyeti**: M1 bu makinede 120 gün.

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

| T | toplam | codebook | cholesky |
|---|---|---|---|
| 1 | 12.6 h | 7.7 h | 4.8 h |
| 4 | **38.0 h** | 19.3 h | 18.7 h |
| 16 | 16.6 h | 9.5 h | 7.1 h |
| max | 7.6 h | 7.5 h | ~0 |

**M1 (3 bütçe × 7 tile × 5 çekiliş): 120 gün.** Örneklenmiş ölçekle 68 gün.
`τ` süpürmesi: **33 gün** (spec 25 *saat* diyordu).

Cholesky `(n_out/T)·k³` olduğu için **ızgaranın ince ucunda yoğunlaşıyor** —
yani granülerlik tezinin en çok veriye ihtiyaç duyduğu bölge, en pahalı bölge.
`affordable()` bir zaman bütçesine neyin sığdığını hesaplıyor ve `T=1` ile
`T=max`'ı asla düşürmüyor (onlar kapının tanımı).

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

**Cholesky yapısal.** Her tile'ın kendi sütun kümesi + kendi rotasyonu var,
dolayısıyla kendi alt-Hessian'ını faktorize etmesi gerekiyor. `T=1`'de bu
**satır başına** bir faktorizasyon — SparseGPT'nin var oluş sebebi olan
"row Hessian challenge"ı birebir yeniden üretiyoruz.

Rotasyon artık vazgeçilemez olduğuna göre (§5.1) geriye iki seçenek kalıyor:

| seçenek | beklenen | durum |
|---|---|---|
| Rotasyonu **8'lik gruplara blok-köşegen** kısıtla → paylaşılan Hessian faktorizasyonu kurtulur | Cholesky terimi çöker | **ÖLÇÜLMEDİ.** Incoherence processing'in ne kadarının hayatta kaldığı açık. Ucuz sınanır: `m0_rotation_value.py`'ye üçüncü bir kol |
| Donanım kirala | A100/H100 fp32 lineer cebirde ~10–20× | Birkaç yüz dolar |

**`fit_scale`'i örneklemek** ayrı bir kaldıraç (6×) ama ölçülmüş bir kalite
bedeli var: katman-başı ölçek %11 kötü. Doğrusu tile-başı ölçeği *örneklemek* —
`fit_scale(sample=N)` kodda var, **kalite etkisi ölçülmedi**.

---

## 7. Sırada ne var

### Bir sonraki oturumun ilk işi

**Blok-köşegen rotasyonu sına.** §6.3'ün tek ölçülmemiş seçeneği, ve tutarsa
hem maliyeti hem donanım bağımlılığını birlikte çözüyor. `m0_rotation_value.py`
zaten iki kol koşuyor; üçüncü kol eklemek küçük iş, GPU'da ~3 dakika.

### Ön-kaydı dondurmak için kalan iki kutu

İkisi de `tau_sweep.py`'ye bağlı: **`Δ(T)` tahmin eğrisi** ve **`T*_tahmin`**.
Süpürme 33 gün olduğu için ikisi de bloke. Maliyet düşmeden ön-kayıt donmaz,
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
- **`fit_scale` örneklemesinin kalite etkisi** — kod var, ölçüm yok

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

**Maliyet.** 120 gün bir dizüstünde koşulamaz. §6.3 çözülmezse proje ya donanım
kiralamaya ya da ızgarayı daraltmaya mecbur — ve daraltmak tam da kanıtın
olduğu yeri budar.

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
| `experiments/m0_rotation_value.py` | rotasyon gerçek katmanda kazandırıyor mu |

**Belgeler:** `spec_v7.md` (şartname) · `preregistration.md` (M1 ön-kaydı,
**dondurulmadı** — iki kutu kaldı) · `audit.md` (v6 denetimi, tarihsel kayıt) ·
`gate_a_dry_run.md` (literatür provası) · bu belge.

```bash
python -m pytest tests/ -q                         # 452 test, ~75 s
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
