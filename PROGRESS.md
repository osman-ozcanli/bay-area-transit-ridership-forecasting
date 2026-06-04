# BART Project — Profesyonelleşme Yol Haritası & İlerleme

> Bu dosya projenin **tek doğru-zamanlı ilerleme kaydıdır** (single source of truth).
> Her adım tamamlandığında bu dosyaya "ne yaptık / ne kazandık / nasıl doğruladık" işlenir.
> Bir sonraki adıma geçmeden önce **onay** alınır.

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
| 2 | Gerçek veriden örneklem + veri yükleme modülü (`src/data`) | ⏳ Onay bekliyor |
| 3 | Feature engineering modülü (`src/features`) | ⬜ Planlandı |
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

### Adım 6 — Notebook'u anlatı katmanına dönüştür
**Yapılacak:** Temiz notebook modülleri import eder, EDA sorularını cevaplar. `cell-48` fix, heatmap yorumları, duplicate temizlik.
**Kazanım:** Okunabilir, savunulabilir portföy notebook'u. **Çözülen:** 3.4, 3.6, 3.10.

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
- **Sonraki:** Adım 2 için onay bekleniyor.
