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

**Protokol ayrımı dizi uzunluğuymuş.** Ölçümden önce hipotez olarak kaydedildi,
ölçümde tuttu. Kural "birini seç" değil "pencereyi sabitle".

---

## 6. Sırada ne var

### Hemen yapılabilir (model yerelde, hat çalışıyor)

1. **`tau_sweep.py`** — `Q(d)` ve `τ(T,d)` yüzeyleri. Ön-kaydın tahmin eğrisi
   buna dayanıyor. `Q` 5 nokta × 3 seed + `τ` 25 nokta × 1 seed.
   ⚠️ Maliyet ölçülmeli: bir dense eval 4 dk sürüyor ama her `τ` noktası
   kalibrasyon + budama + eval demek. Spec ~25 GPU-saat diyor; bu kartta
   **önce 1 nokta ölçülüp ekstrapole edilmeli.**

2. **Transfer pilotu** — `τ`'yu bir `(T,d)` noktasında hem quantization'sız hem
   E8P ile ölç. Ön-kaydın **toleransı** buradan türetilecek. Seed varyansından
   türetilirse prereg "tutmadı" dalına kilitlenir. ~2 GPU-saat.

3. **Kapı B'nin minimum saptanabilir farkı** — 5 çekiliş `T=4` ile `T=16`'yı
   ayırmaya yetiyor mu? Kapı A provası çapa 1'de tam o iki hücreyi belirsiz
   bırakmıştı, yani bu teorik bir kaygı değil. **M1'den önce.**

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

**En büyüğü: Kapı A'nın düşme olasılığı yüksek.** Prova (`gate_a_dry_run.md`)
GPTQ-4bit survivor'larla her satırın düştüğünü gösterdi. E8P aritmetiği
değiştiriyor ama **gösterilmedi**. Karar tablosunun `✗/✓` dalı hazır: proje
durmaz, çerçeve daralır.

**E8P varsayımı** (§3). Düşerse bant 1.83–2.83'e kayar ve tezin "2 bitin altı"
motivasyonu zayıflar.

**Kapı B'nin istatistiksel gücü.** 5 çekiliş yetmeyebilir. Yetmiyorsa ya daha
çok çekiliş ya da daha büyük etki gerekir.

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

**Belgeler:** `spec_v7.md` (şartname) · `preregistration.md` (M1 ön-kaydı,
**dondurulmadı**) · `audit.md` (v6 denetimi, tarihsel kayıt) ·
`gate_a_dry_run.md` (literatür provası) · bu belge.

**Çalıştırma:**
```bash
python -m pytest tests/ -q                    # 353 test
HF_HUB_DISABLE_XET=1 python experiments/m0_dense_ppl.py --seqlens 2048 4096 --device cuda
python experiments/m1_gates.py --synthetic --n-out 64 --n-in 128 --budgets 1.5
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
| *(bu tur)* | VQ maliyeti checkpoint'ten ölçüldü; SU/SV ayrışması bulundu |

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
