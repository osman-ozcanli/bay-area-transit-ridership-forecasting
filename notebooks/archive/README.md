# Archive — referans (reference only)

`bart-project-bay-area-transit-ridership.ipynb` projenin **ilk/orijinal** dağınık
notebook'udur. Sadece **tarihsel referans** için tutulur — çalıştırılması/güncellenmesi
amaçlanmaz.

Final, savunulabilir portföy notebook'u: [`../bart_kaggle.ipynb`](../bart_kaggle.ipynb)
(modüler `src/` koduna dayanır; EDA + eğitim + değerlendirme + sonuç tek anlatı).

Orijinaldeki bilinen sorunlar (analizde tespit edilen) yeni pipeline'da çözüldü:
temporal split (leakage yok), category dtype (ordinal fix), CV, model kayıt,
feature importance yorumu, `cell-48` throughput-ağırlıklı saat dağılımı, duplicate blok yok.
