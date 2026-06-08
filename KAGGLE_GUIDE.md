# Kaggle'da Çalıştırma Rehberi (GPU)

Bu proje modüler (`src/`) yapıdadır ve Kaggle'a **GitHub'dan clone** ile taşınır.
Tek doğruluk kaynağı GitHub reposudur: yerelde `git push` → Kaggle'da yeniden
`clone` → kod her zaman güncel. Çalıştırılacak hazır notebook:
[`notebooks/bart_kaggle.ipynb`](notebooks/bart_kaggle.ipynb).

---

## 0. Ön koşul: Veri Kaggle'da dataset olarak ekli olmalı
Ham CSV'ler git'e girmez (çok büyük). Kaggle'da veriyi bir **Dataset** olarak
notebook'a attach et. Beklenen yol: `/kaggle/input/bart-ridership/` içinde
`date-hour-soo-dest-2016.csv`, `date-hour-soo-dest-2017.csv`, `station_info.csv`.
(Dataset slug farklıysa `config.yaml > paths > kaggle > raw_dir` güncellenir.)

## 1. Notebook ayarları (sağ panel)
- **Internet → On** (GitHub'dan clone için zorunlu)
- **Accelerator → GPU** (LightGBM GPU eğitimi için; sadece EDA'yı çalıştıracaksan gerekmez)

## 2. Notebook'u import et ve sırayla çalıştır
`notebooks/bart_kaggle.ipynb` zaten tüm hücreleri içerir; sıra:

| Hücre | Bölüm | Not |
|-------|-------|-----|
| Kurulum | `git clone` + `sys.path` + taze import self-check | Internet On |
| Veri | `load_dataset` (tam ~13.3M, `use_sample=False`) | — |
| **EDA** | `run_eda(df, cfg)` → 6 iş sorusu + 7 grafik + yorum | **Eğitimden ÖNCE**, GPU gerekmez |
| Feature | `build_features` (self-trip drop, lag/rolling, category) | — |
| Eğitim | `train_model` (temporal split + CV + GPU) → `save_model` | **~65–70 dk** |
| Değerlendirme | metrik (`results`'tan) + feature importance + grafik | yeniden predict yok |

> Notebook'taki kurulum hücresi `cfg["environment"]="kaggle"` ve `cfg["use_sample"]=False`'u
> kendisi ayarlar; `config.yaml`'ı elle düzenlemene gerek yok. (Alternatif:
> `os.environ["BART_ENV"]="kaggle"`.)

## 3. Eğitimi atlamak istersen (modeli yükle)
Repoda kayıtlı model (`models/bart_lgb_final.txt`) clone ile gelir. 70 dk'lık eğitimi
tekrarlamadan değerlendirmeye geçmek için, eğitim hücresi yerine:
```python
import os, lightgbm as lgb
from src.config import PROJECT_ROOT
model_path = os.path.join(str(PROJECT_ROOT), "models", cfg["files"]["final_model"])
model = lgb.Booster(model_file=model_path)
print("Model yuklendi:", model_path)
```
> Geliştirirken bunu kullan (hızlı). Final portföy sürümü için bir kez **Save & Run All
> (Commit)** ile eğitimi gerçekten çalıştır → yayınlanan sürüm tüm çıktıları taşır.

---

## Beklenen sonuçlar (tam veri, teyitli)
- Holdout 2017: **MAE 3.17 / RMSE 8.48 / R² 0.937**, CV MAE **3.18 ± 0.09**
- EDA: busiest **EMBR**, least **WSPR→SBRN**, en yoğun gün **Çarşamba**,
  gece **%1.2**, Berkeley→SF en iyi saat **04:00**

## Notlar
- Repo herkese açıksa clone için ek kimlik doğrulama gerekmez.
- `holidays` Kaggle'da kurulu değilse: `os.system("pip install holidays -q")`.
