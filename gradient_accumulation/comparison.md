# AMP On vs AMP Off Karşılaştırması

Bu belge, `gradient_accumulation` klasöründeki `amp_on` ve `amp_off` çalıştırılmalarının sonuçlarını karşılaştırmaktadır. Karşılaştırma, Mixed Precision (AMP - Otomatik Karmaşıklık Hassasiyeti) kullanımının ve gradyan biriktirmenin (gradient accumulation) eğitim süresi, bellek tüketimi ve doğruluk üzerindeki etkilerini göstermektedir.

## 1. Özet Tablo

| Yapılandırma | AMP | Tamamlanan Epoch | En İyi Val Loss | Toplam Süre (s) | Ortalama Hız (SPS) | Tepe VRAM (MB)* |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard (BS=32, Acc=1)** | Açık (On) | 5 | 0.1084 | 426.91 | ~813 | 570.38 |
| **Standard (BS=32, Acc=1)** | Kapalı (Off)| 5 | 0.1033 | 408.70 | ~790 | 882.58 |
| **Accumulation (BS=8, Acc=4)** | Açık (On) | 5 | 0.1023 | 530.01 | ~540 | 324.25 |
| **Accumulation (BS=8, Acc=4)** | Kapalı (Off)| 4 (Early Stop) | 0.1101 | 446.17 | ~531 | 383.78 |

*(Yüksek bellek tüketen ilk epoch başlatma aşamasından sonraki kararlı durum tepe VRAM değerleri alınmıştır.)*

## 2. Temel Bulgular

### Bellek (VRAM) Tüketimi
* **AMP'nin Etkisi:** Beklendiği gibi, AMP (Mixed Precision) kullanımı VRAM tüketimini önemli ölçüde azaltmaktadır. Standart batch size (32) eğitiminde AMP açıkken VRAM **~882.5 MB'den ~570.3 MB'ye** düşmüştür.
* **Gradient Accumulation Etkisi:** Batch size'ı 8'e düşürüp 4 adımda birikim yapmak belleği daha da rahatlatmıştır. AMP ile birlikte kullanıldığında tepe VRAM kullanımı **324.2 MB** gibi çok düşük bir seviyeye inmiştir.

### Eğitim Hızı ve Süresi
* **Hız (SPS - Samples Per Second):** AMP açıkken saniyede işlenen örnek sayısı (SPS) genellikle daha yüksektir. Örneğin standart BS32 eğitiminde SPS ~790'dan ~813'e çıkmaktadır.
* **Toplam Süre:** İlginç bir şekilde, standart ayarlarda AMP açıkken (426.9s) AMP kapalı durumuna (408.7s) göre toplam süre biraz daha uzun sürmüştür. Model veya GPU (örneğin Tensor Core verimliliği) AMP'nin sağladığı işlem hızı artışından ziyade diğer darboğazlara (overhead vb.) takılmış olabilir.
* **Gradient Accumulation Süresi:** Gradyan biriktirme işlemi, sık forward/backward pass'lerden dolayı standart eğitime kıyasla genel olarak modeli daha yavaş eğitmektedir (~530s vs ~426s).

### Doğruluk / Kayıp (Loss)
* AMP Açık (On) ve Kapalı (Off) durumları arasında kayda değer bir performans bozulması yaşanmamıştır. Hatta AMP açık ve Gradient Accumulation kullanılan senaryoda ulaşılan `0.1023` val_loss değeri en iyi sonuç olmuştur.

## 3. Sonuç
AMP kullanımı, özellikle kısıtlı VRAM'e sahip ortamlarda büyük bir avantaj sağlamaktadır. Daha büyük modelleri sığdırmak için Gradient Accumulation ile AMP kombinasyonu oldukça etkilidir. Eğitim hızında hafif dalgalanmalar veya küçük overhead durumları yaşansa da (toplam sürenin hafif uzaması gibi), sağlanan %35-40'lık VRAM tasarrufu, bu kombinasyonun kullanılmasını ideal kılar.
