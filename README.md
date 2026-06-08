# BART Bay Area Transit Ridership — ML Project

2016–2017 BART (Bay Area Rapid Transit) yolculuk verisi (~13.3M kayit) uzerinde,
herhangi iki istasyon arasinda **saatlik yolcu yogunlugunu (Throughput)** tahmin eden
bir LightGBM modeli + 6 is sorusuna cevap veren kesifsel veri analizi (EDA).

Tek bir daginik notebook'tan; moduler (`src/`) + config-driven + test edilen
profesyonel bir repo'ya donusturulmustur. Karar gunlugu: [PROGRESS.md](PROGRESS.md).

---

## 🎯 Sonuclar (tam veri, 2017 holdout)

| Metrik | Deger |
|--------|-------|
| **MAE**  | **3.17** |
| **RMSE** | **8.48** |
| **R²**   | **0.937** |
| TimeSeriesSplit CV (MAE) | **3.18 ± 0.09** (stabil, dusuk varyans) |

İki bagimsiz Kaggle egitimi neredeyse bire bir ayni sonucu verdi (MAE 3.1733 → 3.1725)
→ pipeline **tekrarlanabilir (reproducible)**. Egitim 2016 (~9.76M satir), test 2017 (~3.25M satir).

### En guclu sinyal — feature importance
`Throughput_lag_1` tek basina gain'in **~%63'u**; lag + rolling birlikte **~%69**.
Yani talep guclu sekilde **zaman-otokorelasyonlu**: yakin gecmis, yakin gelecegi belirliyor.
Kalan sinyal *ne zaman + nerede* (Hour / Origin / Destination / Period).

---

## 📊 EDA — 6 is sorusu (tam veri ile dogrulanmis cevaplar)

1. **En yogun istasyon?** → **EMBR** (Embarcadero) > MONT > POWL — SF finans bolgesi.
2. **En az populer rota?** → **WSPR → SBRN** (agin guney ucu, cok dusuk talep).
3. **En yogun gun?** → **Carsamba** (hafta ortasi); hafta sonu belirgin dusuk.
4. **Gece (LateNight) yolcu payi?** → toplam yolculugun ~**%1.2'si**.
5. **En populer rotalar?** → SF cekirdegi kisa koridorlari (POWL↔BALB/MONT/24TH)
   **ve** uzun banliyo→merkez rotalari (DUBL/FRMT/WOAK → EMBR) karisimi.
6. **Berkeley → SF'te koltuk icin en iyi saat?** → **04:00** (zirve 08–09; aksam donusu ~19:00).

Tum EDA mantigi `src/eda.py`'de modulerdir; her soru kod → grafik → yorum olarak
[`notebooks/bart_kaggle.ipynb`](notebooks/bart_kaggle.ipynb) icinde anlatilir.

---

## 🗂️ Proje Yapisi

```
.
├── config.yaml             # Tum path & parametreler (no hardcoding)
├── requirements.txt
├── data/
│   ├── raw/                # Ham CSV'ler (git'e girmez)
│   ├── sample/             # Gercek veriden turetilmis kucuk ornek (yerel test)
│   └── processed/
├── src/
│   ├── config.py           # config.yaml yukleyici (local/kaggle, BART_ENV override)
│   ├── data/               # veri yukleme (load.py) & ornekleme (make_sample.py)
│   ├── features/           # feature engineering (build_features.py)
│   ├── eda.py              # 6 is sorusu: hesap + grafik + yorum
│   └── models/             # train.py (temporal split + CV) , evaluate.py
├── notebooks/
│   ├── bart_kaggle.ipynb   # FINAL anlatim notebook'u (EDA → egitim → degerlendirme)
│   └── archive/            # eski/orijinal notebook (referans-only)
├── models/                 # kaydedilen model (bart_lgb_final.txt)
├── reports/figures/        # uretilen grafikler
└── tests/                  # pytest smoke testleri
```

---

## ⚙️ Kurulum

```bash
pip install -r requirements.txt
```

## 📦 Veri

Kaynak: Kaggle "BART Ridership" (`date-hour-soo-dest-2016/2017.csv`, `station_info.csv`).
Ham CSV'ler `data/raw/` altina konur. Yerel gelistirme icin gercek veriden zaman-bilincli
kucuk bir ornek (`data/sample/`, ~1.12M satir) turetilir; **tam egitim Kaggle'da** yapilir.

> Not: `station_info.csv` ridership'taki **WSPR** istasyonunu icermez; referans
> butunlugu icin config'ten manuel eklenir (`manual_stations`).

## ▶️ Calistirma

Ortam `config.yaml` → `environment` ile secilir (`local` ornek veri | `kaggle` tam veri).
`use_sample` bayragi yerel ornek ile tam veri arasinda gecis yapar.

**Yerel moduler pipeline (ornek veri ile):**
```bash
python -m src.eda             # EDA: 6 is sorusu + grafikler
python -m src.models.train    # load → feature → temporal split → egit → kaydet
python -m src.models.evaluate # holdout metrikleri + feature importance + grafikler
```

**Kaggle (tam veri):** [`notebooks/bart_kaggle.ipynb`](notebooks/bart_kaggle.ipynb)'i import et
→ repo `git clone` ile gelir, hucreler sirayla calisir (kurulum → veri → EDA → feature →
egitim → degerlendirme → sonuc). Adim adim rehber: [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md).

## ✅ Testler

```bash
python -m pytest tests/ -v
```
Smoke testler (yerel, ~30 sn): config cozumleme, leakage-safe lag + self-trip drop,
EDA'nin 6 soruyu uretmesi. Kaggle/egitim gerektirmez.

---

## 🧠 Metodolojik Notlar

- **Temporal split:** Zaman serisi oldugu icin 2016 → train, 2017 → test
  (random split kullanilmaz; **future leakage** onlenir). Dogru split ile R² yine ~0.94
  → leakage suphesi curutuldu, model gercekten zamanda genellesiyor.
- **Kategorik degiskenler:** Istasyonlar LightGBM'e `categorical_feature` olarak verilir
  (`cat.codes` ordinal varsayimindan kacinilir).
- **Leakage-safe lag/rolling:** OD-bazli, zaman-sirali, sadece **gecmis** kayit (shift'li).
- **Self-trip drop:** Origin==Destination kayitlari (turnike artefakti, dist_km=0) atilir.
- **Reproducibility:** sabit `random_state`, kaydedilen model, pinlenmis bagimliliklar.
- Detayli gerekce ve karar gunlugu: [PROGRESS.md](PROGRESS.md).
