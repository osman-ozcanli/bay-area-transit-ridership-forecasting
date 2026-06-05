# BART Project — Profesyonelleşme Yol Haritası & İlerleme

> Bu dosya projenin **tek doğru-zamanlı ilerleme kaydıdır** (single source of truth).
> Her adım tamamlandığında bu dosyaya "ne yaptık / ne kazandık / nasıl doğruladık" işlenir.
> Bir sonraki adıma geçmeden önce **onay** alınır.

---

## ▶️ DEVAM NOKTASI (yeni session burayı okusun)

- **Tamamlanan:** Adım 0, 1, 2 ✅
- **Şu an:** **Adım 3 — Feature engineering (`src/features`)** onay bekliyor.
- **Sıradaki ilk iş:** Kullanıcıdan Adım 3 onayı al; alınınca `src/features/build_features.py` yaz (datetime/holiday/haversine + encoding fix + self-trip kararı + lag/rolling).
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
| 3 | Feature engineering modülü (`src/features`) | ⏳ Onay bekliyor |
| 4 | Temporal split + model eğitim modülü (`src/models`) | ⬜ Planlandı |
| 5 | Değerlendirme + yorumlama | ⬜ Planlandı |
| 6 | Notebook'u anlatı (narrative) katmanına dönüştür | ⬜ Planlandı |
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
- **Sonraki:** Adım 3 için onay bekleniyor.
