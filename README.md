# BART Bay Area Transit Ridership — ML Project

2016–2017 BART (Bay Area Rapid Transit) yolculuk verisi (~13M kayit) uzerinde
istasyon-saat bazli yolcu yogunlugu (Throughput) tahmini ve EDA.

> **Durum:** Profesyonellesme refactor'u devam ediyor. Adim ilerlemesi icin
> [PROGRESS.md](PROGRESS.md) dosyasina bakin.

## Proje Yapisi

```
.
├── config.yaml             # Tum path & parametreler (no hardcoding)
├── requirements.txt
├── data/
│   ├── raw/                # Ham CSV'ler (git'e girmez)
│   ├── sample/             # Gercek veriden turetilmis kucuk ornek (yerel test)
│   └── processed/
├── src/
│   ├── config.py           # config.yaml yukleyici
│   ├── data/               # veri yukleme & ornekleme
│   ├── features/           # feature engineering
│   ├── models/             # egitim, degerlendirme, tahmin
│   └── utils/
├── notebooks/              # anlatim (narrative) notebook'u
├── models/                 # kaydedilen model artifact'leri
├── reports/figures/        # uretilen grafikler
└── tests/
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Veri

Kaynak: Kaggle "BART Ridership" (`date-hour-soo-dest-2016/2017.csv`, `station_info.csv`).
Ham CSV'ler `data/raw/` altina konur. Yerel gelistirme icin gercek veriden
zaman-bilincli kucuk bir ornek (`data/sample/`) turetilir; tam egitim Kaggle'da yapilir.

## Calistirma

`config.yaml` icindeki `environment` degeri ile ortam secilir:
- `local`  → yerel path'ler, ornek veri ile pipeline dogrulama
- `kaggle` → `/kaggle/input/...` path'leri, tam veri ile egitim

## Metodolojik Notlar

- **Temporal split:** Zaman serisi oldugu icin 2016 → train, 2017 → test
  (random split kullanilmaz; future leakage onlenir).
- **Kategorik degiskenler:** Istasyonlar LightGBM'e `categorical_feature` olarak
  verilir (cat.codes ordinal varsayimindan kacinilir).
- Detayli gerekce ve karar gunlugu: [PROGRESS.md](PROGRESS.md).
