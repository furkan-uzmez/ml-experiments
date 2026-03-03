# Resolution Comparison Experiment on COVIDx Dataset

## Amaç

Bu deney, görüntü çözünürlüğünün derin öğrenme model performansı üzerindeki etkisini araştırmaktadır. İki farklı senaryo karşılaştırılmıştır:

1. **Orijinal Çözünürlük** — COVIDx veri setindeki orijinal boyuttaki görüntüler
2. **256×256 Yeniden Boyutlandırılmış** — Önceden 256×256 piksele küçültülmüş görüntüler

Her iki senaryoda da görüntüler eğitim sırasında 224×224 boyutuna yeniden ölçeklendirilmiştir (ResNet18 girdi boyutu).

## Deney Kurulumu

| Parametre | Değer |
|---|---|
| **Model** | ResNet18 (ImageNet ön-eğitimli) |
| **Veri Seti** | COVIDx (İkili sınıflandırma: Negatif / Pozitif) |
| **Optimizer** | Adam (lr=0.0001) |
| **Loss Fonksiyonu** | CrossEntropyLoss |
| **Epoch Sayısı** | 5 (max) |
| **Early Stopping Patience** | 3 |
| **Batch Size** | 32 |
| **Giriş Boyutu** | 224×224 |
| **Normalize** | ImageNet standartları ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) |

## Eğitim Sonuçları

### Senaryo 1: Orijinal Çözünürlük

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Durum |
|---|---|---|---|---|---|
| 1 | 0.1007 | 0.9719 | 0.1823 | 0.9462 | ✅ Best model kaydedildi |
| 2 | 0.0659 | 0.9822 | 0.2054 | 0.9441 | ❌ İyileşme yok (1/3) |
| 3 | 0.0526 | 0.9846 | 0.2126 | 0.9524 | ❌ İyileşme yok (2/3) |
| 4 | 0.0400 | 0.9871 | 0.2074 | 0.9533 | ❌ İyileşme yok (3/3) — Early Stopping |

- **En İyi Epoch:** 1
- **En İyi Val Loss:** 0.1823
- **Toplam Eğitim Süresi:** ~20 dakika (epoch başına ~5 dk)

### Senaryo 2: 256×256 Yeniden Boyutlandırılmış

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Durum |
|---|---|---|---|---|---|
| 1 | 0.0963 | 0.9739 | 0.2013 | 0.9476 | ✅ Best model kaydedildi |
| 2 | 0.0640 | 0.9826 | 0.1750 | 0.9528 | ✅ Best model kaydedildi |
| 3 | 0.0512 | 0.9850 | 0.1745 | 0.9482 | ✅ Best model kaydedildi |
| 4 | 0.0388 | 0.9878 | 0.2268 | 0.9516 | ❌ İyileşme yok (1/3) |
| 5 | 0.0272 | 0.9906 | 0.2773 | 0.9523 | ❌ İyileşme yok (2/3) |

- **En İyi Epoch:** 3
- **En İyi Val Loss:** 0.1745
- **Toplam Eğitim Süresi:** ~14 dakika (epoch başına ~3 dk)

## Test Sonuçları Karşılaştırması

| Metrik | Orijinal Çözünürlük | 256×256 Yeniden Boyutlandırılmış | Fark |
|---|---|---|---|
| **Accuracy** | 0.5396 | 0.5275 | -0.0121 |
| **Precision** | 0.5210 | 0.5142 | -0.0068 |
| **Recall** | 0.9818 | 0.9974 | +0.0156 |
| **F1 Score** | 0.6808 | 0.6785 | -0.0023 |
| **AUC** | 0.6989 | 0.6899 | -0.0090 |

## Sonuç ve Değerlendirme

İki senaryo arasında **anlamlı bir performans farkı gözlemlenmemiştir**:

- Tüm test metrikleri (Accuracy, Precision, Recall, F1, AUC) her iki senaryoda da birbirine çok yakın değerler üretmiştir.
- En büyük fark Recall metriğinde olup, yeniden boyutlandırılmış görüntülerle eğitilen model **+1.56%** daha yüksek recall elde etmiştir. Diğer tüm metrikler orijinal çözünürlük lehine olmakla birlikte, farklar **%1'in altında** kalmaktadır.
- Ancak 256×256 çözünürlükte eğitim, epoch başına süreyi yaklaşık **%40 azaltmıştır** (~5 dk → ~3 dk), bu da eğitim verimliliği açısından önemli bir avantajdır.

Bu sonuçlar, COVIDx veri seti üzerinde ResNet18 ile ikili sınıflandırma görevinde, görüntülerin önceden 256×256 boyutuna küçültülmesinin model performansını olumsuz yönde etkilemediğini göstermektedir. Her iki senaryoda da model 224×224 girdi boyutunu kullandığından, orijinal görüntülerin daha yüksek çözünürlükte olması ek bilgi sağlamamaktadır.

## Proje Yapısı

```
resolution_comparision_dl/
├── experiment.ipynb          # Ana deney notebook'u
├── README.md                 # Bu dosya
├── functions/
│   ├── dataset.py            # COVIDxResolutionDataset sınıfı
│   ├── evaluation.py         # Değerlendirme metrikleri ve grafik fonksiyonları
│   ├── logger.py             # Loglama yardımcı fonksiyonu
│   └── train.py              # Eğitim döngüsü (early stopping destekli)
├── logs/
│   ├── training_orig.log     # Orijinal çözünürlük eğitim logları
│   └── training_res.log      # 256×256 yeniden boyutlandırılmış eğitim logları
└── models/
    ├── best_model_orig.pth   # Orijinal çözünürlük en iyi model ağırlıkları
    └── best_model_res.pth    # 256×256 yeniden boyutlandırılmış en iyi model ağırlıkları
```
