# Ön-kayıt — M1'in iki kapısı

**Durum: TASLAK. Henüz dondurulmadı.** §9'daki dondurma listesi tamamlanana
kadar bu belge değiştirilebilir. Liste tamamlandığı anda commit edilir ve
**bir daha değiştirilmez** (Spec v6 §7, tuzak 22). Sonradan ortaya çıkan her
sapma §10'a *eklenir*, yukarısı düzeltilmez.

Amaç: M1'in sonucunu görmeden önce, neyi ölçeceğimizi ve hangi sonucun neye
sayılacağını yazmak. Bunun sebebi kibar bir formalite değil — §7'de anlatılan
somut bir başarısızlık modu var: kapının tanımını sonuca bakarak seçmek.

---

## 1. Ne test ediliyor

İki **bağımsız** soru. Ayrı tutulmalarının sebebi, birinin düşmesinin diğerini
geçersiz kılmaması.

**Kapı A — uygulanabilirlik.** Test edilen seyrek konfigürasyonların en iyisi,
2 bitlik yoğun PTQ tabanını geçiyor mu?

**Kapı B — tez.** Optimum `T` **içeride mi, uçta mı?**

> Kapı B'yi *"tile > unstructured"* diye tanımlamak trivialdir ve yasaktır
> (Spec v6 §7, tuzak 7). Soru sıralama değil, iç optimumun varlığı.

---

## 2. Dondurulan deney ızgarası

**Model:** Llama-2-7B. **Eksen:** B (row-tile). **Survivor quantizer:** QuIP# E8P,
`vq_bits = 2.0`.

> **Ölçülen düzeltme ve neden ızgaraya girmiyor (2026-08-21).** Gerçek
> checkpoint'te kodsözcüğü yükü tam 2.000000, katman-başı yan bilgiyle birlikte
> 2.005204 (QTIP'te 2.006740). Bizim hattımızda karşılığı
> `accounting.rotation_side_bits` ile **0.0075–0.0085 bit/survivor** — `T` ile
> neredeyse hiç değişmiyor.
>
> **Izgara `vq_bits = 2.0`'da donduruluyor.** Gerekçe: düzeltme her hücrede
> **aynı göreli** miktarda (%0.26–0.42) yoğunluk düşürüyor — bütçeden bağımsız,
> `T`'den bağımsız — yani hücreler arası bütçe-eşleşmesini bozmuyor, sadece
> hepsini birlikte kaydırıyor. Buna karşılık tam 2, `B=1.5` ızgarasının
> tamamını tam dyadic kesirler yapıyor (1/4, 1/2, 5/8, 11/16, 23/32, 47/64,
> 3/4) ve `tests/golden.py`'nin bağımsız türetmesi buna dayanıyor.
>
> **Şart:** makalede raporlanan bit bütçesi düzeltmeyi **geri ekler**; yani
> `B = 1.50` hücresi "1.50 bit" değil, "1.50 + 0.008 ≈ 1.51 bit" olarak
> raporlanır. Bu satır dondurmadan önce yazıldı ki sonradan seçilmiş olmasın.

E8P ile canlı bant **1.40 – 1.80** (Spec v6 §3.5 filtresi). Birincil bütçeler
üçü de bu bandın içinde ve **tamamı 2 bitin altında** — yani yoğun PTQ'nun
cevabının olmadığı rejim.

Yoğunluklar muhasebeden türetilmiştir, elle yazılmamıştır
(`accounting.density_for_budget`, `n_idx = 11008`):

| B | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | T=max |
|---|---|---|---|---|---|---|---|
| 1.75 | 0.375000 | 0.625000 | 0.750000 | 0.812500 | 0.843750 | 0.859375 | 0.875000 |
| 1.60 | 0.300000 | 0.550000 | 0.675000 | 0.737500 | 0.768750 | 0.784375 | 0.800000 |
| 1.50 | 0.250000 | 0.500000 | 0.625000 | 0.687500 | 0.718750 | 0.734375 | 0.750000 |

`T` ızgarası: **{1, 2, 4, 8, 16, 32, max}**. Gerekçe yalnızca özdeşlikten:
kazanım oranı `1 − 1/T`, yani T=2 %50, T=4 %75, T=8 %87.5, T=16 %93.75,
T=32 %96.9. `{1, 16, max}` üçlüsü yanlış-durdurma taşırdı — `T=4` kazanımın
dörtte üçünü çok daha düşük kısıt bedeliyle veriyor.

`T=16` avantajı `(1−1/T)/W = 0.46875`, `B*(T=1) = 1.148962`. Bütün bütçeler
`B*`'ın üzerinde, yani özdeşliğin geçerlilik alanındayız.

**Hizalama:** LDLQ sekizli grup istediğinden survivor sayısı 8'in katına
yuvarlanır. `n_idx = 11008`'de bu %0.07'lik bir kayma, ama **gerçekleşen**
yoğunluk ve bit sayısı her satırda raporlanır; asla istenen değer raporlanmaz.

---

## 3. Çapa ve rakip

**Çapa kararı (2026-08-20): yetkin baseline.** Rakip QTIP/QuIP#'tir, GPTQ değil.

Gerekçe: GPTQ 3-bit sınıfının en zayıf üyesi. Ona çapalanırsa Kapı A kolay geçer
ama sonuç savunulamaz — §6 zaten yetkin baseline şart koşuyor. GPTQ ikincil
bağlam olarak raporlanır.

**PTQ tabanı referansı:** QTIP 2-bit. Bütçe-eşleşmiş **değildir** ve bu kasıtlı:
karşılaştırma *"2 bitin altında, 2 bite karşı"*. Her tabloda bu böyle yazılır.

---

## 4. Protokol

Llama-2-7B için literatürde **iki uyumsuz aile** var (dense 5.12 ve 5.47) ve
aynı yöntemin sayısı 0.47 ppl değişiyor — Kapı B'nin çözmeye çalıştığı etkiden
büyük. Ayrıntı: `docs/gate_a_dry_run.md`.

**Kural (2026-08-21'de ölçüldü):** ayrımın sebebi dizi uzunluğuymuş. Kendi
ölçümümüz seqlen 2048'de **5.4675**, 4096'da **5.1143** — ikisi de yayımlanmış
değerlerin 0.006 içinde. Yani iki aile de yeniden üretilebiliyor; kural "birini
seç" değil **"pencereyi sabitle"**.

Yayımlanmış bir sayı, ancak bizim **aynı `seqlen`'de** aldığımız bir sayının
yanına konabilir. `eval.perplexity.compare` protokoller uyuşmazsa hata fırlatır.

Ölçüm: WikiText-2 **ve** C4, `convention="gptq"`, **seqlen 4096 birincil**
(§9'da donduruldu; 2048 ikincil olarak raporlanır).
Zero-shot 5 görev ayrıca raporlanır ama kapılara girmez.

---

## 5. Tahmin — ön-kaydın çekirdeği

M0'ın iki yüzeyi:

- `Q(d)` — unstructured yozlaşma eğrisi (`T=1`), 3 seed
- `τ(T,d)` — **eş-yoğunlukta** granülerlik vergisi, 1 seed

Tahmin:

```
Δ(T)_tahmin = Q(d(T)) + τ(T, d(T))        # d(T) muhasebeden, Q ve τ M0'dan
T*_tahmin   = argmin_T Δ(T)_tahmin
```

**Ön-kaydedilenler:** ① `Δ(T)` eğrisinin tamamı (sayısal, her `T` için),
② `T*_tahmin`, ③ aşağıdaki ayrılabilirlik varsayımı, ④ tolerans (§9).

> **v5'in eşiği döngüseldi.** `beklenen_kazanç` cebirsel olarak sadeleşince tam
> da Kapı B'nin ölçtüğü farkı veriyordu; "ölçülen etkinin %50'si" eşiği kapının
> tanım gereği geçmesini garantiliyordu. Ayrıca boyutsuz orandan boyutlu sayı
> çıkarılıyordu. Bu yüzden eşik değil, **tahmin doğrulaması** yapılır.

### 5.1 Ayrılabilirlik varsayımı — ayrıca kaydedilir

`Δ = Q + τ` bir teorem değildir. Eş-yoğunlukta ölçülen verginin bütçe-eşleşmiş
ve quantize edilmiş ayara transfer olduğunu varsayar. **Test edilen şey budur.**

| Tahmin | `T*` | Okuma |
|---|---|---|
| Tuttu | içeride | **Tez destekleniyor**; `T*` anlamlı, ayrışma geçerli |
| Tuttu | uçta | Tez yanlış ama **model doğru**; `T*` anlamlı |
| Tutmadı | — | **Ayrışma geçersiz** — kendi başına bulgu. `T*` yorumlanamaz, ampirik raporlanır |

*"Tez yanlış"* ile *"model yanlış"* farklı şeylerdir ve karıştırılmayacaktır.

### 5.2 `τ`'nun seed'i eşleştirilmiştir

`τ`, bir **fark** ölçtüğü için tek seed yeterlidir — ama yalnızca `ppl(T,d)` ve
`ppl(1,d)` **aynı kalibrasyon çekilişinden** geliyorsa. Ortak gürültü ancak o
zaman sadeleşir.

> **Kural:** `τ`'nun seed'i, `Q`'nun üç seed'inden **biriyle aynı çekiliştir**
> ve `τ` o çekiliş üzerinde eşleştirilmiş fark olarak hesaplanır.

### 5.3 `wb × T` — birincil öngörü DEĞİL

Birinci koşul `[|Q'| + |∂_dτ|]/(T²W) = ∂_Tτ`. `W` küçüldüğünde sol taraf büyür
→ `T*` büyümeli. Ama düşük `W`'de `d` yüksektir (eğrinin düz kısmı, `|Q'|`
küçük), yani karşı yönde bir kuvvet var. **İşaret belirsizdir.**

Ayrıca `is_live` ile birleşince birincil bantta yalnızca iki `W` değeri canlı
kalıyor — 1.32× aralık. Log aralıklı `T` ızgarasında `T*`'ın kayması için etkinin
en az 2× olması gerekir.

> **Kayıt:** `wb × T` birincil öngörüden çıkarılmıştır. `Δ(T)` eğrileri `wb`
> başına raporlanır, `argmin` kayması **iddia edilmez**.

---

## 6. Kapı A — karar kuralı

**Geçer** ⟺ test edilen seyrek konfigürasyonların en iyisinin bootstrap güven
aralığı, PTQ tabanı referansının **tamamen altında** kalır.

Örtüşme geçiş sayılmaz.

**A fortiori kuralı geçerlidir:** Wanda ile kazanılırsa sonuç kesindir.
Kaybedilirse SparseGPT'ye yükseltilir ve tekrarlanır.

---

## 7. Kapı B — karar kuralı

Kapı B **argmin değildir.** Üç koruma birlikte çalışır; üçü de zorunludur.

**① Çekiliş sayısı ≥ 5.** Altında verdikt **"undetermined"**tir.

> Spec §6 "seed ≥ 3" diyor. Üç çekiliş ortalama raporlamaya yeter, **bu kapıyı
> karara bağlamaya yetmez**: percentile bootstrap üç sayıyı yeniden örnekler ve
> %95 aralığının %95 kapsaması olmaz. Bu, implementasyon sırasında somut olarak
> gözlendi — düzeltme öncesi `gate_b` saf gürültüde "interior" verdi.

**② Bonferroni.** `T*` iç adaylar arasından argmin ile seçilip *aynı*
çekilişlerle test edildiği için alpha, aday sayısına bölünür. Bu yapılmazsa aynı
veri hem seçim hem test için kullanılmış olur.

**③ Eşleştirilmiş bootstrap.** Her `T` aynı kalibrasyon çekilişini gördüğünden
fark üzerinde eşleştirilmiş yeniden örnekleme yapılır.

**Verdikt:**

| Koşul | Verdikt |
|---|---|
| `Δ(T*)` hem `Δ(1)`'den hem `Δ(max)`'tan CI ile ayrışıyor | **interior** |
| `Δ(T*) ≥ min(Δ(1), Δ(max))` | **edge** |
| Aksi | **undetermined** |

Uygulama: `experiments/m1_gates.gate_b`, testleri altı ayrı gürültü çekilişinde.

### 7.1 Minimum saptanabilir fark — ölçüldü (2026-08-21)

`experiments/m0_gate_b_power.py`, **gerçek `gate_b`'yi** çağırarak 600 denemelik
simülasyon (idealize bir z-testi değil; Bonferroni düzeltmesi ve `min_seeds`
kapısı dahil). Etki, eşleştirilmiş gürültü `σ` biriminde:

| çekiliş | %80 güç | %50 güç | δ=0'da yanlış "interior" |
|---|---|---|---|
| 3 | *(karar yok)* | *(karar yok)* | 0.000 |
| **5** | **2.29 σ** | 1.47 σ | 0.033 |
| 8 | 1.76 σ | 1.28 σ | 0.010 |
| 10 | 1.64 σ | 1.20 σ | 0.010 |
| 15 | 1.37 σ | 0.98 σ | 0.005 |
| 20 | 1.25 σ | 0.88 σ | 0.012 |
| 30 | 0.95 σ | 0.73 σ | 0.003 |

Tip-I hızı her yerde nominal %5'in altında — ①②③'ün üçü birlikte çalışıyor ve
kapı gürültüde şişmiyor.

**σ ölçümü** (sentetik katman, `B=1.5`, 8 çekiliş — büyüklük mertebesi, değer
değil): kalibrasyon ekseninde `σ = 0.00446`, hata seviyesinin **%1.41**'i.
Buna göre 5 çekilişte saptanabilir fark hata seviyesinin **%3.2**'si.

**Aynı katmanda ölçülen gerçek etki:** uçlar iç optimumdan `T=1` için **26.5 σ**,
`T=max` için **6.7 σ** uzakta. Bağlayıcı olan `T=max` ve o da eşiğin **3 katı**.

> **KARAR: Kapı B'nin verdikti için 5 çekiliş yeterlidir** ve `gate_b`'nin
> `min_seeds=5` eşiği korunur. Gerekçe yukarıdaki 3× marj.

### 7.2 `T*` bir nokta değil, bir kümedir

Aynı simülasyon ikinci ve daha sıkı bir kısıt buldu: **verdikt ile `T*` aynı
güvenilirlikte değil.** Optimumu *uçlardan* ayırmak büyük bir farktır; *komşu
tile'dan* ayırmak küçük. Düz bir iç bölgede (`spread=3`), `δ=1σ`, 20 çekiliş:
verdikt %77 doğru, **argmin yalnızca %41**.

Sentetik katmanda ölçülen komşu farkları bunu doğruluyor:

| karşılaştırma | fark | 5 çekilişte argmin doğru | %90 için gereken çekiliş |
|---|---|---|---|
| `T=4` ↔ `T=16` | 2.69 σ | ~%99.9 | 2 |
| **`T=4` ↔ `T=8`** | **0.31 σ** | **~%65** | **~53** |

> **KURAL:** `T*` **tek başına raporlanmaz.** `m1_gates.t_star_set` argmin'den
> ayrılamayan bütün iç tile'ları döndürür ve manşet o kümedir. Tek elemanlı bir
> küme granülerlik hakkında gerçek bir iddiadır; dört elemanlı bir küme
> *"eğri düz, optimum içeride ama yeri belirsiz"* demektir — ve bu da meşru bir
> sonuçtur, sahte bir kesinlik değil.

### 7.3 Çekiliş ekseni: kalibrasyon, rotasyon değil

Aynı koşuda ölçüldü: rotasyon seed'i ekseninde `σ = 0.00228`, kalibrasyon
ekseninin **yarısı** (oran 1.95×). `GateRun` bu ölçüme kadar çekilişleri
rotasyon seed'i üzerinden üretiyordu; öyle koşulsa Kapı B kanıtın izin
verdiğinden **iki kat kendinden emin** çıkacaktı.

> **KURAL:** Kapı B'nin CI'ları **kalibrasyon çekilişleri** üzerinedir.
> `GateRun.run` artık `LayerProblem` listesi alır. Tek problem verilirse
> rotasyon seed'ine düşer ve çıktı `draw_axis` ile **etiketlenir**; o etiketi
> taşıyan hiçbir sayı Kapı B kanıtı olarak raporlanamaz.

**Not — eşleştirmenin kazancı beklenenden küçük.** Ölçülen eşleştirme kazancı
yalnızca **1.16×**, yani kalibrasyon gürültüsü tile boyutları arasında büyük
ölçüde bağımsız çıktı. Bu, §5.2'nin *"ortak gürültü sadeleşiyor"* gerekçesini
zayıflatır **ama kuralı değiştirmez**: eşleştirme hiçbir koşulda zarar vermez,
ve sentetik kurgu ortak bileşeni yapısı gereği eksik ölçüyor — her çekiliş
**aynı** dağılımdan yeniden çiziliyor, oysa gerçek kalibrasyon çekilişleri
içerik olarak birbirinden farklı. 1.16× bir **alt sınır** olarak okunmalıdır.

### 7.4 M1'e girerken uyarlanabilir kontrol (önceden kaydedilir)

Yukarıdaki σ sentetiktir. Gerçek σ ilk M1 bütçesinden gelecek. Sonradan
seçilmiş görünmemesi için kural **şimdi** yazılıyor:

> İlk bütçe `n = 5` çekilişle koşulur. O koşudan `σ` ve `δ = min(Δ(1), Δ(max)) −
> Δ(T*)` hesaplanır. **`δ/σ < 2.29` ise** kalan bütçelere geçmeden önce çekiliş
> sayısı, yukarıdaki tablodan `δ/σ`'yı karşılayan değere yükseltilir ve **ilk
> bütçe de o sayıyla yeniden koşulur.** Çekiliş sayısı sonuca bakılarak
> ayarlanmaz — yalnızca ölçülen `δ/σ`'ya bakılarak ayarlanır.

**Saliency kuralı:** a fortiori Kapı B'ye **uygulanmaz.** SparseGPT'ye yükseltmek
farklı `T`'leri eşit iyileştirmez ve yönü öngörülemez. Bu yüzden Kapı B hem Wanda
hem SparseGPT ile koşulur; işaretler farklıysa **bu kendi başına bulgudur**.

---

## 8. Sonuçların okunması

| Kapı A | Kapı B | Anlamı | Aksiyon |
|---|---|---|---|
| ✓ | interior | En iyi senaryo | Devam |
| ✗ | interior | Granülerlik gerçek, dağıtım için ilgisiz | Çerçeveyi *"seyreklik ailesi içinde granülerlik"*e daralt; roofline'ı öne çıkar |
| ✓ | edge | **Granülerlik tezi öldü, proje ölmedi** | Kapı A sonucunu yaz: `T*=1` → *"2 bit altında 4-bit+unstructured yoğun VQ'yu yeniyor"*; `T*=max` → *"structured + VQ yeniyor"* (donanım dostu, daha da pratik) |
| ✗ | edge | — | **Dur** |

`T*` uçta çıkması projeyi bitirmez; Kapı A'nın bağımsız değeri vardır. İki kapıyı
ayırmanın bütün amacı buydu.

---

## 9. Dondurma listesi — hepsi dolmadan bu belge geçerli değil

- [ ] **Tolerans** `|Δ(T)_ölçülen − Δ(T)_tahmin| ≤ ___`
      → **transfer pilotundan** türetilecek, seed varyansından değil.
      Tahmin hatasına hakim olan şey seed gürültüsü değil, ayrılabilirlik
      varsayımının yanlılığıdır; seed varyansından türetilen tolerans neredeyse
      kesin aşılır ve prereg "tutmadı" dalına kilitlenir.
      *Pilot: `τ`'yu tek bir `(T,d)` noktasında hem quantization'sız hem E8P ile
      ölç, farkı tolerans olarak al. ~2 GPU-saat.*
- [x] **Minimum saptanabilir fark** (§7.1) — 2026-08-21 ölçüldü.
      5 çekiliş → **2.29 σ**; ölçülen etki `T=max` ucunda 6.7 σ, yani
      3× marj. `min_seeds=5` korunuyor. Ayrıca §7.2 (`T*` küme olarak),
      §7.3 (çekiliş ekseni kalibrasyon) ve §7.4 (uyarlanabilir kontrol).
- [ ] **`Δ(T)` tahmin eğrisi** — M0'ın `Q` ve `τ` yüzeyleri tamamlanınca
- [ ] **`T*_tahmin`**
- [x] **Dense ppl ölçümü ve protokol kimliği** (§4) — 2026-08-21, ölçüldü:
      seqlen 2048 → **5.4675** (aile `dense-5.47`), seqlen 4096 → **5.1143**
      (aile `dense-5.12`). İkisi de yayımlanmış değerden 0.006'dan az sapıyor,
      yani hattımız her iki aileyi de yeniden üretiyor.
- [x] **seqlen = 4096** (birincil). Gerekçe: `dense-5.12` ailesi hem budama
      baseline'larını (Wanda/SparseGPT %50, 2:4, 4:8) hem güçlü quantization
      baseline'larını (QTIP/QuIP# 2-3-4 bit) birlikte taşıyor — Kapı A'nın
      rakibi QTIP 2-bit de orada. seqlen 2048 ikincil olarak raporlanır;
      SliceGPT ve QuaRot karşılaştırmaları yalnızca orada geçerli.
- [x] **`vq_bits = 2.0` doğrulaması** — 2026-08-21, `experiments/m0_vq_bits.py`.
      Kodsözcüğü yükü **tam 2.000000**; katman-başı yan bilgi dahil edilince
      **2.005204**. Manifest aritmetiği ile toplam dosya boyutu birbirini tam
      tutturuyor. Fark %0.26 ve bütçeden bağımsız olarak yoğunluğu aynı oranda
      düşürüyor — `accounting.E8P_STORED_BITS`.
- [x] **QTIP checkpoint'inin ölçülen bit maliyeti** — aynı koşu:
      yük **tam 2.000000**, saklanan **2.006740**. Kapı A'nın rakibi 2 biti
      gerçekten 2 bitte tutuyor; provanın QTIP 2-bit = 5.86 satırı bütçe
      açısından dürüst.

### 9.1 Taşınan açık varsayım

> **E8P'nin kompaktlanmış survivor alt-matrisinde `vq_bits ≈ 2.0` kalitesini
> koruduğu VARSAYILIYOR; doğrulanmadı.**
>
> 2026-08-21 ölçümü **maliyeti** kapattı, kaliteyi değil. Açık olan tam olarak
> şu: 2 bit ödeniyor, ama kalın kuyruklu survivor alt-matrisinde 2 bitlik
> *kalite* alınıp alınmadığı bilinmiyor.

Survivor'lar tanım gereği dağılımın kalın kuyruğu, kafes quantizer ise Gauss'a
yakın girdi ister. Bu varsayımı sınayacak ucuz deney **bilinçli olarak atlandı**
(karar: 2026-08-20).

**Erken uyarı kuralı:** ilk katman E8P'den geçtiğinde katman-çıkışı MSE'si dense
E8P referansının 2 katını aşarsa varsayım düşmüş sayılır. O noktada geri dönüş
yolu rotasyon + GPTQ-3bit'tir (`W = 3.148`) ve bant 1.83–2.83'e kayar. Bu geri
dönüş kodda hazır tutulur.

---

## 10. Sapma günlüğü

Dondurulduktan sonra ortaya çıkan her sapma buraya **eklenir**; yukarısı
düzeltilmez. Boş olması iyiye işarettir, dolu olması dürüstlük işaretidir.

*(henüz yok)*
