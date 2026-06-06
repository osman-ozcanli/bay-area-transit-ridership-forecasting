# BART Project — Profesyonelleşme Yol Haritası & İlerleme

> Bu dosya projenin **tek doğru-zamanlı ilerleme kaydıdır** (single source of truth).
> Her adım tamamlandığında bu dosyaya "ne yaptık / ne kazandık / nasıl doğruladık" işlenir.
> Bir sonraki adıma geçmeden önce **onay** alınır.

---

## ▶️ DEVAM NOKTASI (yeni session burayı okusun)

- **Tamamlanan:** Adım 0, 1, 2, 3, 4, 5 ✅
- **Şu an:** **Adım 6 — Notebook'u anlatı (narrative) katmanına cilala** onay bekliyor.
- **Sıradaki ilk iş:** Kullanıcıdan Adım 6 onayı al; alınınca `bart_kaggle.ipynb`'yi final portföy notebook'una getir (EDA sorularının markdown anlatımı, grafik yorumları, akış). Eski dağınık notebook'un kaderine karar ver.
- **Kaggle çalıştırma modeli:** Notebook'ta **Adım 4 = eğitim ana akış** (çalışan kod, ~65-70 dk, GPU). "Modeli yükle" yolu **(Bilgi) markdown notu** olarak duruyor (eğitmeden devam etmek istenirse koda çevrilir). Model `models/bart_lgb_final.txt` (~43MB) repoda commit'li. Not: ileride Adım 6'da iki-notebook (train / analysis) ayrımı tartışılacak.
- **Çalışma kuralları:** (1) Adım adım ilerle, her adım sonunda onay iste. (2) Önce yerelde çalıştır+doğrula, sonra Kaggle sürümü ver. (3) **Git'i kullanıcı yapar — Claude git komutu çalıştırmaz**, sadece sıralı komutları yazar. (4) Path/parametreler hep `config.yaml`'dan. (5) **Her adımın iki çıktısı vardır:** yerel kod `src/` altına; Kaggle'da çalıştırılacak kısım **`notebooks/bart_kaggle.ipynb`'ye yeni hücre olarak EKLENİR** (üst üste birikir, sohbete yapıştırılmaz).
- **Kaggle notebook'u:** `notebooks/bart_kaggle.ipynb` — birikimli. Şu an: Kurulum (clone) + Adım 2 (veri yükleme). Kullanıcı bunu Kaggle'a import edip çalıştırır. (Eski/dağınık orijinal notebook `notebooks/bart-project-...ipynb` sadece referans.)
- **Ortam:** Python 3.10, yerel veri `data/raw/` (git dışı), örnek `data/sample/bart_sample.csv` (1.12M satır). Tam eğitim Kaggle'da (`environment: kaggle`, `use_sample: false`).
- **GitHub:** `bay-area-transit-ridership-forecasting` (MIT).
- **Kaggle stratejisi (kilitli):** GitHub'dan `git clone` + `sys.path` → modüler `src/` import. GPU = notebook'ta Accelerator toggle + Adım 4'te LightGBM `device: gpu`. Adım adım rehber: **`KAGGLE_GUIDE.md`** (clone/GPU adımları hazır; eğitim hücreleri Adım 4'te dolacak).
- **Adım 4'te yapılacak küçük iş:** `config.py`'ye `BART_ENV` env-var override ekle (Kaggle'da config.yaml elle düzenlemeden `kaggle` ortamına geçiş için).

---

## 0. Başlangıç Durumu (Baseline)

- **Varlıklar:** Tek notebook `bart-project-bay-area-transit-ridership.ipynb` (154 hücre: 117 kod, 37 markdown), boş `12 - BART Project/` klasörü, analiz dosyası.
- **Veri:** Yerelde YOK — notebook Kaggle path'lerinden okuyor (`/kaggle/input/bart-ridership/...`).
- **Analiz puanı:** 6.1 / 10. Kritik sorunlar: `cell-103` sessiz bug, temporal split yok (leakage), `cat.codes` ordinal sorunu, duplicate bloklar, CV yok, model kaydedilmemiş, fonksiyon/type hint/docstring yok.

### Kararlaştırılan hedef
- **Mimari:** Tam profesyonel repo (`src/` modülleri + `config.yaml` + notebook anlatı katmanı).
- **Veri:** Kullanıcı gerçek CSV'leri verir; içinden **temsili küçük örneklem (subset)** türetilir (gerçek şema/dağılım/edge-case korunur). Tam eğitim Kaggle'da.
- **Kapsam:** 9 must-fix + senior eklemeler (TimeSeriesSplit CV, lag/rolling features, feature importance yorumu, model kayıt, reproducibility).

---

## Yol Haritası (sıralı — her adım bir öncekini bozmaz)

| Adım | Başlık | Durum |
|------|--------|-------|
| 0 | Yol haritası + PROGRESS.md | ✅ Tamamlandı |
| 1 | Proje iskeleti + repo hijyeni | ✅ Tamamlandı |
| 2 | Gerçek veriden örneklem + veri yükleme modülü (`src/data`) | ✅ Tamamlandı |
| 3 | Feature engineering modülü (`src/features`) | ✅ Tamamlandı |
| 4 | Temporal split + model eğitim modülü (`src/models`) | ✅ Tamamlandı |
| 5 | Değerlendirme + yorumlama | ✅ Tamamlandı |
| 6 | Notebook'u anlatı (narrative) katmanına dönüştür | ⏳ Onay bekliyor |
| 7 | Dokümantasyon + final cila | ⬜ Planlandı |

---

### Adım 1 — Proje iskeleti + repo hijyeni
**Yapılacak:** Klasör yapısı (`src/`, `data/raw`, `data/processed`, `notebooks/`, `models/`, `reports/figures`, `tests/`), `config.yaml`, `requirements.txt`, `.gitignore`, `README` iskeleti, `git init`. Mevcut notebook `notebooks/` altına taşınır (kopya korunur).
**Kazanım:** Versiyon kontrolü, reproducibility temeli, temiz dizin. **Risk:** Çok düşük, hiçbir şeyi çalıştırmaz/bozmaz.

### Adım 2 — Gerçek veriden örneklem + veri yükleme modülü
**Yapılacak:** Kullanıcının verdiği gerçek CSV'ler `data/raw/` altına konur. İçinden temsili küçük örneklem türetilir (`data/sample/` — örn. zaman aralığı + istasyon dağılımını koruyan stratified subset; WSPR ve self-trip gibi edge-case'ler dahil). `src/data/load.py`: read→concat→station merge→WSPR ekleme, fonksiyonlaştırılmış, type hint + docstring, path'ler `config.yaml`'dan. Pipeline örneklemle çalıştırılıp doğrulanır.
**Kazanım:** Yerelde çalışabilir veri katmanı; gerçek dağılım korunur; WSPR referans-bütünlüğü fix'i korunur. **Çözülen:** hardcode path.

### Adım 3 — Feature engineering modülü
**Yapılacak:** `src/features/build_features.py`: datetime feature'ları, `Period`, `IsWeekend`, holiday (CA), haversine `dist_km`. **Self-trip** (Origin==Destination) kararı + belgeleme. **Encoding fix:** LightGBM'e `category` dtype (cat.codes ordinal sorunu çözülür). **Lag/rolling** feature'lar (leakage-safe).
**Kazanım:** Temiz feature pipeline, encoding fix, temporal feature'lar. **Çözülen:** 3.3, 3.5, FE eksikliği.

### Adım 4 — Temporal split + model eğitim
**Yapılacak:** Temporal split (2016 train / 2017 test). `TimeSeriesSplit` CV. LightGBM `categorical_feature` ile eğitim. Model `models/` altına kaydedilir (`save_model`).
**Kazanım:** Leakage'sız validasyon, CV variance, kaydedilen reproducible model. **Çözülen:** 3.1 (v3 bug tek temiz akışta yok), 3.2, 3.8, 3.9.

### Adım 5 — Değerlendirme + yorumlama
**Yapılacak:** `src/models/evaluate.py`: MAE, RMSE, R², residual analizi. Feature importance + **yorum**. Grafikler `reports/figures/`.
**Kazanım:** Güvenilir metrikler, analitik derinlik. **Çözülen:** 3.7.

### Adım 6 — Notebook'u anlatı (narrative) katmanına cilala
**Yapılacak:** Birikimli `notebooks/bart_kaggle.ipynb`'yi final portföy notebook'una getir: EDA sorularının markdown anlatımı, grafik yorumları, akış düzeni. Eski/dağınık orijinal notebook'un kaderine karar ver (arşivle/kaldır). Not: eski notebook'un `cell-48`/duplicate sorunları (3.4, 3.6, 3.10) yeni notebook'ta zaten yok — sıfırdan temiz kuruluyor.
**Kazanım:** Okunabilir, savunulabilir portföy notebook'u.

### Adım 7 — Dokümantasyon + final cila
**Yapılacak:** README (problem/veri/yaklaşım/sonuç/çalıştırma), requirements pin, docstring kontrolü, opsiyonel testler.
**Kazanım:** Tamamlanmış senior portföy projesi.

---

## İlerleme Günlüğü

### ✅ Adım 0 — Yol haritası (2026-06-04)
- Notebook ve analiz incelendi, analizdeki tüm kritik tespitler kod üzerinde **doğrulandı**.
- Hedef mimari/veri/kapsam kararları alındı (yukarıda).
- Bu PROGRESS.md oluşturuldu.

### ✅ Adım 1 — Proje iskeleti + repo hijyeni (2026-06-04)
- **Klasör yapısı:** `src/{data,features,models,utils}`, `data/{raw,sample,processed}`, `notebooks/`, `models/`, `reports/figures/`, `tests/`. Boş `12 - BART Project/` silindi.
- **Taşımalar:** 3 ham CSV → `data/raw/`; ana notebook → `notebooks/`.
- **config.yaml:** Tüm path/parametreler merkezi; `local`/`kaggle` ortam ayrımı (Kaggle'a geçişte sadece `environment` değişir).
- **src/config.py:** Ortam-bilinçli path çözümleyici (`lru_cache`, type hint + docstring). Yerelde **çalıştırılıp doğrulandı**.
- **Repo hijyeni:** `requirements.txt` (yerel sürümlerle), `.gitignore` (ham veri + kişisel analiz notu + model/figür git dışı), `README` iskeleti.
- **Bağımlılık:** `holidays` kuruldu (Adım 3 için).
- **GitHub:** Repo `bay-area-transit-ridership-forecasting` (MIT license) ile bağlandı, ilk push yapıldı.
- **Kazanım:** Versiyon kontrolü + reproducibility temeli, hardcode path kalktı, temiz dizin. **Hiçbir mevcut mantık bozulmadı** (notebook aynen duruyor).

### ✅ Adım 2 — Gerçek veriden örneklem + veri yükleme modülü (2026-06-04)
- **Veri keşfi (token-güvenli):** 2016 = 9.97M satır (tam yıl), **2017 = 3.31M satır (sadece 1 Oca–3 May!)**, toplam ~13.3M. 46 istasyon, stations dosyasında **WSPR eksik** (doğrulandı). Self-trip: ~276K. `Location` formatı `lon,lat,alt`.
  - ⚠️ **Adım 4 için not:** 2017 ~4 aylık → temporal test seti kısa, raporlanacak.
- **`src/data/load.py`:** `read_ridership` (2016+2017 concat, DateTime parse, sıralı), `read_sample`, `read_stations` (BOM-aware, WSPR config'ten eklenir), `merge_station_info` (origin+dest), `load_dataset` (orchestrator, `use_sample` bayrağına göre sample/full seçer). Type hint + docstring.
- **`src/data/make_sample.py`:** `by_station` stratejisi — en busy + zorunlu edge-case (WSPR) istasyonlar, **hem Origin hem Destination seçili set içinde** (kapalı mini-ağ, tam zaman çizgisi korunur → lag/rolling ve temporal split bozulmaz).
- **Boyut ayarı:** İlk denemede sadece Origin'e göre filtreleyince örnek **4.19M satır (%31.5)** çıktı — bir "küçük örnek" için fazla büyük olduğunu tespit ettik; hem Origin hem Destination'ı seçili sete kısıtlayıp (kapalı mini-ağ) **1.12M satıra (%8.4)** düşürdük.
- **Yerel doğrulama:** Örnek = **1.12M satır / %8.4 / 37MB**, tarih 2016-01-01 → 2017-05-03, 12 istasyon, WSPR + self-trip dahil. `load_dataset` → 1.12M×8, istasyon merge'de **0 null**, WSPR koordinatı çözüldü.
- **Kaggle sürümü:** Kod aynı; Kaggle'da sadece `config.yaml` → `environment: kaggle` + `use_sample: false` (tam 13.3M veri). Path farkı tamamen config ile yönetiliyor.
- **Çözülen:** Hardcode path (analiz: kod kalitesi), referans bütünlüğü korundu.

### ✅ Adım 3 — Feature engineering (2026-06-05)
- **`src/features/build_features.py`** (config-driven, type hint + docstring):
  - `add_time_features`: Hour, DayOfWeek, Month, IsWeekend, Period (orijinal mantık).
  - `add_holiday_feature`: `holidays.US(subdiv='CA')` → IsHoliday.
  - `add_distance_feature`: `Location`'ı `lon,lat`'a parse (2/3 parçaya dayanıklı, WSPR güvenli) + haversine `dist_km` (vectorized).
  - `handle_self_trips`: **policy="drop"** (config'ten) → analiz 3.5 çözüldü.
  - `add_lag_features`: OD-bazlı, zaman-sıralı `lag_1`, `lag_24`, `roll_mean_3` — **leakage-safe** (sadece geçmiş, shift'li). NaN'ler LightGBM'e bırakıldı.
  - `prepare_categoricals`: Origin/Destination/DayOfWeek/Period → `category` dtype → **analiz 3.3 (cat.codes ordinal) çözüldü**.
- **Yerel doğrulama (örnek):** 1.118M → **1.028M satır** (90.134 self-trip düştü, kalan 0). dist_km 0.52–51.96 km, 0 NaN. IsHoliday %3.4. lag_1 NaN = 132 = OD çifti sayısı (her grup ilk kaydı). **Leakage testi geçti** (lag_1 = bir önceki saat değeri).
- **Kaggle notebook:** `bart_kaggle.ipynb`'ye Adım 3 hücreleri eklendi (7 hücre).
- **Çözülen:** 3.3 (encoding), 3.5 (self-trip) + lag/rolling senior eklemesi.

### ✅ Adım 4 — Temporal split + model eğitim + kayıt (2026-06-05)
- **`config.py`:** `BART_ENV` env-var override eklendi (Kaggle'da `os.environ["BART_ENV"]="kaggle"` → config.yaml elle düzenlemeye gerek yok).
- **`config.yaml`:** `model` bölümü (device cpu/gpu, params, num_boost_round=2000 tavan, early_stopping=50, run_cv, cv_splits=3).
- **`src/models/train.py`:** `temporal_split` (2016→train/2017→test, analiz 3.2), `run_cv` (TimeSeriesSplit, analiz 3.9), `train_model` (LightGBM `categorical_feature` + early stopping + **GPU→CPU otomatik fallback**), `save_model` (analiz 3.8).
- **Yerel doğrulama (örnek, CPU):** train/test = 761.728/266.370. best_iter=700 (early stopping, 2000'e gitmedi). **Holdout MAE 6.63 / RMSE 14.07 / R² 0.9437.** CV MAE 6.60 ± 0.24 (düşük varyans). Model `models/bart_lgb_final.txt`'e yazıldı.
- **KRİTİK BULGU:** Doğru temporal split ile R² yine 0.94 → analiz 3.2'deki "leakage R²'yi şişirmiş olabilir" şüphesi **çürütüldü**, model gerçekten zamanda genelleşiyor.
- **KAGGLE TAM VERİ SONUCU (GPU, 2026-06-06):** train/test = 9.763.937 / 3.245.031. **Holdout MAE 3.1733 / RMSE 8.4799 / R² 0.9369.** CV MAE **3.177 ± 0.0914** (çok düşük varyans → stabil). Model `/kaggle/working/models/bart_lgb_final.txt`. (Yereldeki örnek MAE 6.63'tü; tam veri ile 3.17'ye düştü.) Tüm 4 tur ~65-70 dk.
- **Kaggle notebook:** Adım 4 hücreleri eklendi (9 hücre, `device=gpu`).
- **Çözülen:** 3.2 (temporal split), 3.8 (model kayıt), 3.9 (CV). Not: 3.1 (v3 bug) yeni tek-akış pipeline'da zaten yok.

### ✅ Adım 5 — Değerlendirme + feature importance YORUMU (2026-06-06)
- **`src/models/evaluate.py`:** `load_model` (repodan, PROJECT_ROOT-relative → local+Kaggle aynı), `evaluate_holdout` (2017 MAE/RMSE/R²), `feature_importance_df` (gain+split), `plot_feature_importance` + `plot_residuals` (reports/figures'a kayıt), `interpret_importance` (yazılı yorum).
- **Feature importance (tam-veri model):** `Throughput_lag_1` **%62.9** (ezici), Hour %8.8, Origin %7.4, roll_mean_3 %6.1, Destination %6.0, Period %5.0, dist_km %2.2, gerisi minik. **Lag+rolling birlikte %69** → talep zaman-otokorelasyonlu. **Analiz 3.7 (grafik var yorum yok) çözüldü** — yazılı yorum hem koda hem notebook markdown'ına eklendi.
- **Yerel doğrulama:** evaluate.py çalıştı, 2 grafik üretildi (feature_importance.png, residuals.png). (Yerel MAE 6.59 — örnek 12-istasyon alt kümesi; Kaggle tam veride ~3.17.)
- **Notebook'a Adım 5 eklendi (13 hücre):** Adım 4 **eğitim ana akışta kaldı** (çalışan kod); "modeli yükle" yaklaşımı **(Bilgi) markdown notu** olarak eklendi (eğitmeden devam isteğe bağlı). Adım 5 hücreleri: değerlendirme + feature importance + grafikler. Tüm kod hücreleri syntax-check'ten geçti.
- **⚠️ İYİLEŞTİRME NOTU (sonraki aşama için):** Adım 5'i Kaggle'da çalıştırırken çok uzun bekledik, çünkü `evaluate_holdout` 3.2M satırı **yeniden predict** ediyor (özellikle bellekteki GPU-modelinde yavaş). Halbuki canlı session'da holdout metrikleri eğitimden **zaten `results` içinde hazırdı** ve feature importance modelden **anında** okunuyor → yeniden predict gereksizdi. **Öneri (Adım 6'da uygula):** Adım 5'i "canlı session" için sadeleştir → metrikleri `results`'tan kullan + importance'ı modelden oku (saniyeler); ağır residual/predict'i **ayrı opsiyonel hücreye** al ya da portföy için bir kez üret. (Bu turda kod değiştirilmedi, sadece not.)
- **Sonraki:** Adım 6 için onay bekleniyor (orada iki-notebook [train/analysis] ayrımı kararı verilecek).
