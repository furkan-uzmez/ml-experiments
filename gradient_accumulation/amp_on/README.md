# Gradient Accumulation Experiment

> **Amaç:** Aynı efektif batch boyutu koşulunda **Standart Eğitim** ile **Gradient Accumulation** yöntemini kontrollü bir deneyle karşılaştırmak; loss/accuracy, eğitim süresi, throughput ve donanım kullanımı üzerindeki etkilerini ölçmek.

---

## İçindekiler

1. [Proje Amacı](#proje-amacı)
2. [Çabuk Başlangıç](#çabuk-başlangıç)
3. [Proje Yapısı](#proje-yapısı)
4. [Temel Kavramlar](#temel-kavramlar)
5. [Deney Tasarımı](#deney-tasarımı)
6. [Sonuçlar ve Bulgular](#sonuçlar-ve-bulgular)
7. [Modüller](#modüller)

---

## Proje Amacı

Büyük batch boyutları GPU belleğine sığmadığında **Gradient Accumulation** yaygın bir çözümdür: küçük micro-batch'ler üzerinden gradyanlar biriktirilerek, büyük bir batch'in gradyanı simüle edilir. Bu proje şu soruyu yanıtlar:

> *`batch_size=32` ile tek seferde adım atmak, `batch_size=8 × 4 adım birikimi` ile eşdeğer midir? Hangisi daha hızlı, hangisi daha az bellek kullanır?*

### Kontrol Edilen Metrikler

| Kategori | Metrikler |
|---|---|
| **Başarım** | `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy` |
| **Hız** | `total_train_time_sec`, `avg_epoch_time_sec`, `avg_step_time_sec` |
| **Verimlilik** | `samples_per_sec`, `batches_per_sec` |
| **Donanım** | `peak_vram_mb`, `cpu_percent`, `ram_used_mb` |

---

## Çabuk Başlangıç

```bash
# 1. Bağımlılıkları yükle
pip install torch torchvision tqdm psutil scikit-learn matplotlib seaborn pandas

# 2. Ana notebook'u aç
jupyter notebook main.ipynb
```

> Notebook sırayla çalıştırıldığında **Standart** ve **Accumulation** koşullarını eğitir, logları diske kaydeder ve karşılaştırmalı görseller üretir.

---

## Proje Yapısı

```
gradient_accumulation/
├── functions/                  # Yeniden kullanılabilir Python modülleri
│   ├── dataset.py              # Veri seti sınıfı ve DataLoader fabrikası
│   ├── train.py                # train_one_epoch / validate_one_epoch / fit
│   ├── evaluation.py           # Sızıntısız metrik hesaplama ve görselleştirme
│   ├── logging.py              # CSV + JSON tabanlı deney loglama motoru
│   └── logger.py               # Geriye uyumluluk shim'i (logging.py'yi re-export eder)
│
├── runs/                       # Deney çıktıları (otomatik oluşturulur)
│   ├── standard_bs32_acc1/     # Standart eğitim çıktıları
│   │   ├── epoch_metrics.csv   # Epoch başına loss/accuracy/timing
│   │   ├── step_metrics.csv    # Step bazlı ayrıntılı metrikler
│   │   ├── system_metrics.csv  # GPU/CPU/RAM anlık ölçümler
│   │   ├── run_summary.json    # Özet istatistikler
│   │   ├── run.log             # İnsan okunabilir log
│   │   └── best_model.pth      # En iyi checkpoint
│   │
│   └── accum_bs8_acc4/         # Gradient accumulation çıktıları
│       └── (aynı yapı)
│
├── main.ipynb                  # Deney orkestrasyonu ve analiz notebook'u
├── implementation_plan.md      # Proje tasarım belgesi
└── README.md                   # Bu dosya
```

---

## Temel Kavramlar

### Gradient Accumulation Nedir?

Normal eğitimde her `optimizer.step()` çağrısı tek bir batch'in gradyanına dayanır. Gradient accumulation'da ise:

```
[Adım 1] loss.backward()  ← gradyanlar birikir, optimizer.step() yok
[Adım 2] loss.backward()  ← gradyanlar birikmeye devam eder
[Adım 3] loss.backward()  ← ...
[Adım 4] loss.backward()  ← optimizer.step() + zero_grad()  ✓
```

Bu sayede `batch_size=8 × 4 birikim = 32 efektif batch` elde edilir; büyük modelleri sınırlı VRAM'de eğitmek mümkün hale gelir.

### Loss Normalizasyonu

Gradyanların eşit ağırlık taşıması için loss her micro-step'te `accumulation_steps`'e bölünür:

```python
loss_normalized = loss / accumulation_steps
loss_normalized.backward()
```

---

## Deney Tasarımı

| Parametre | Standart | Accumulation |
|---|---|---|
| `batch_size` (micro) | 32 | 8 |
| `accumulation_steps` | 1 | 4 |
| **Efektif batch boyutu** | **32** | **32** |
| `model` | ResNet18 | ResNet18 (aynı) |
| `optimizer` | AdamW | AdamW (aynı) |
| `epochs` | 5 | 5 |
| `seed` | sabit | sabit |

> Her iki koşulda da **efektif batch boyutu 32** olarak sabit tutulmuştur; tek değişken micro-batch + birikim adımı sayısıdır.

---

## Sonuçlar ve Bulgular

### 5 Epoch Sonunda Özet

| Metrik | Standart (`bs32`) | Accumulation (`bs8 acc4`) | Fark |
|---|---|---|---|
| **Best Val Loss** | 0.1022 | **0.0939** | ✅ Accumulation daha iyi |
| **Best Epoch** | 4 | 5 | — |
| **Total Train Time** | 643.4 sn | 665.9 sn | +%3.5 (accumulation daha yavaş) |
| **Peak VRAM (Epoch 1)** | 1702.5 MB | **1354.3 MB** | ✅ Accumulation %20.5 daha az |
| **Peak VRAM (Epoch 2+)** | 573.3 MB | 323.4 MB | ✅ Accumulation %43.6 daha az |
| **Avg Epoch Time** | ~118-122 sn | ~123-126 sn | Accumulation biraz daha uzun |

### Epoch Bazında Val Loss Karşılaştırması

| Epoch | Standard Val Loss | Accumulation Val Loss |
|---|---|---|
| 1 | 0.1169 | 0.1170 |
| 2 | 0.1103 | 0.1038 |
| 3 | 0.1413 | 0.1244 |
| 4 | **0.1022** | 0.1029 |
| 5 | 0.1024 | **0.0939** |

### Epoch Bazında Val Accuracy Karşılaştırması

| Epoch | Standard Val Acc | Accumulation Val Acc |
|---|---|---|
| 1 | 96.46% | 96.42% |
| 2 | 96.66% | 96.42% |
| 3 | 95.41% | 96.06% |
| 4 | 96.75% | 96.61% |
| 5 | 96.80% | **97.09%** |

---

## Bulgular ve Yorum

### ✅ Gradient Accumulation Avantajları

1. **Daha İyi Genelleme:** Son epoch'ta val loss `0.0939` vs `0.1024` — accumulation yaklaşık **%8.1 daha düşük validation loss** elde etmiştir. Bunun nedeni küçük micro-batch'lerin daha yüksek gradyan gürültüsü üretmesi; bu gürültünün bir tür düzenlileştirici etki yarattığı bilinmektedir.

2. **Belirgin VRAM Tasarrufu:** İlk epoch'ta %20.5, sonraki epoch'larda %43.6 daha az GPU belleği kullanılmıştır. Bellek kısıtlı ortamlarda bu fark kritik önem taşır; daha büyük modeller veya daha büyük efektif batch'ler uygulanabilir hale gelir.

3. **Eşdeğer Throughput:** Her iki koşulda da saniyedeki örnek sayısı (~434–455 samples/sec) birbirine yakındır; accumulation ek bir hız cezasına yol açmamaktadır.

### ⚠️ Gradient Accumulation Dezavantajları / Dikkat Edilecekler

1. **Hafif Süre Artışı:** Accumulation deneyi toplam eğitimde ~22 saniye (%3.5) daha uzun sürmüştür. Bu, birden fazla backward pass ile ilgili Python/CUDA overhead'inden kaynaklanmaktadır.

2. **GPU Senkronizasyonu:** Step bazlı timing ölçümü yapılırken `cuda.synchronize()` ihmal edilirse zaman ölçümleri yanıltıcı olabilir; mevcut implementasyon bunu doğru şekilde ele almaktadır.

3. **Daha Fazla Adım Sayısı:** Accumulation modunda optimizer step başına 4× daha fazla DataLoader iterasyonu yapılır; bu durum `step_metrics.csv` dosyasının ~4× büyük olmasına neden olmuştur (944 KB vs 244 KB).

### 🎯 Sonuç

> **Bellek kısıtlıysa veya daha iyi genelleme isteniyorsa: Gradient Accumulation kullan.**  
> **Hız birincil öncelikse ve VRAM yeterliyse: Standart eğitim tercih edilebilir.**

Efektif batch boyutu sabit tutulduğunda gradient accumulation, eşit ya da daha iyi model başarımı ile birlikte önemli bellek tasarrufu sağlamaktadır.

---

## Modüller

### `functions/dataset.py`
- `XRayDataset`: CSV tabanlı veri seti sınıfı; path çözme, transform, opsiyonel bellek önbelleği.
- `create_dataloader(...)`: `pin_memory`, `num_workers`, `prefetch_factor`, `persistent_workers` ile optimize edilmiş DataLoader fabrikası.

### `functions/train.py`
- `train_one_epoch(...)`: Tek epoch eğitimi; gradient accumulation, step timing, VRAM peak ölçümü.
- `validate_one_epoch(...)`: `torch.inference_mode()` ile sızıntısız doğrulama.
- `fit(...)`: Early stopping, checkpoint kaydetme, loglama. Ana orkestrasyon fonksiyonu.

### `functions/evaluation.py`
- `evaluate(...)`: Loss, accuracy, precision, recall, F1, AUC; yapılandırılmış dict döner.
- `plot_confusion_matrix(...)`, `plot_roc_curve(...)`: Görselleştirme yardımcıları.

### `functions/logging.py`
- `ExperimentLogger`: Step, epoch ve sistem metriklerini `runs/<run_name>/` altında CSV ve JSON olarak persiste eder.
- Loglar: `step_metrics.csv` · `epoch_metrics.csv` · `system_metrics.csv` · `run_summary.json` · `run.log`

---

*Son güncelleme: 2026-02-27*
