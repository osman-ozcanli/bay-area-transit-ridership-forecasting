# Kaggle'da Çalıştırma Rehberi (GPU)

Bu proje modüler (`src/`) yapıdadır ve Kaggle'a **GitHub'dan clone** ile taşınır.
Tek doğruluk kaynağı GitHub reposudur; yerelde `git push` → Kaggle'da yeniden
`clone` → kod her zaman güncel.

> Durum: Clone + GPU kurulum adımları geçerli. **Eğitim (training) hücreleri
> Adım 4 tamamlanınca buraya eklenecek.**

---

## 0. Ön koşul: Veri Kaggle'da dataset olarak ekli olmalı
Ham CSV'ler git'e girmez (çok büyük). Kaggle'da veriyi bir **Dataset** olarak
notebook'a attach et. Beklenen yol: `/kaggle/input/bart-ridership/` içinde
`date-hour-soo-dest-2016.csv`, `date-hour-soo-dest-2017.csv`, `station_info.csv`.
(Dataset slug farklıysa `config.yaml > paths > kaggle > raw_dir` güncellenir.)

## 1. Notebook ayarları (sağ panel)
- **Accelerator → GPU** (LightGBM GPU eğitimi için)
- **Internet → On** (GitHub'dan clone için zorunlu)

## 2. İlk hücre — repoyu clone'la ve import yolunu aç
```python
import os, sys

REPO = "bay-area-transit-ridership-forecasting"
URL = "https://github.com/osman-ozcanli/bay-area-transit-ridership-forecasting.git"

# Repo yoksa clone'la (hücre tekrar çalıştırılırsa hata vermesin)
if not os.path.exists(REPO):
    os.system(f"git clone {URL}")

sys.path.insert(0, f"/kaggle/working/{REPO}")
```

## 3. İkinci hücre — Kaggle ortamını seç + bağımlılıklar
```python
# Kaggle ortamini aktif et (tam veri + /kaggle/input path'leri)
os.environ["BART_ENV"] = "kaggle"   # config.py bunu okuyacak (Adim 4'te eklenecek)

# holidays Kaggle'da kurulu degilse:
# os.system("pip install holidays -q")
```

## 4. Üçüncü hücre — veriyi yükle (mevcut, çalışır)
```python
from src.data.load import load_dataset
df = load_dataset()
print(df.shape, df["DateTime"].min(), "->", df["DateTime"].max())
```

## 5. Eğitim (Adım 4'te eklenecek)
> LightGBM `device: gpu` ile temporal split + TimeSeriesSplit CV + model kayıt.
> Somut hücreler ve beklenen GPU süresi Adım 4 bittiğinde buraya yazılacak.

---

## Notlar
- `config.yaml` repo içinde `environment: local` ile gelir; Kaggle'da
  `BART_ENV=kaggle` override'ı ile değiştirilecek (bu override Adım 4'te
  `config.py`'ye eklenecek — şimdilik gerekiyorsa config.yaml elle `kaggle`
  yapılır).
- Repo herkese açıksa clone için ek kimlik doğrulama gerekmez.
