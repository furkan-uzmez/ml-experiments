# Husky vs. Wolf - Kısayol Öğrenimi (Shortcut Learning) ve Grad-CAM Analizi

Bu proje, Derin Öğrenme modellerinde sıkça karşılaşılan **"Kısayol Öğrenimi" (Shortcut Learning)** problemini, popüler "Husky (Sibirya Kurdu) vs. Wolf (Kurt)" veri seti üzerinde incelemeyi ve Grad-CAM (Gradient-weighted Class Activation Mapping) kullanarak görselleştirmeyi amaçlamaktadır.

## Proje Hakkında

Normal şartlarda bir modelden, bir resimdeki Sibirya Kurdu ile bir Kurt'u fiziksel özelliklerine (yüz hatları, kulakları, kürk yapısı vb.) bakarak ayırması beklenir. Ancak, bu veri setinde kasti bir **önyargı (bias)** bulunmaktadır:
*   **Kurt (Wolf)** resimlerinin büyük çoğunluğu arka planda **kar** bulunan ortamlarda çekilmiştir.
*   **Husky** resimlerinde ise kar yoktur; daha çok çimenlik, orman veya insan yapımı ortamlar (ev, traktör vb.) yer almaktadır.

Sıfırdan eğitilen (Pretrained olmayan) bir model, hayvanın kendisine odaklanmak gibi zor bir işi yapmak yerine, daha kolay bir "kısayol" bularak arka planı ezberler: **"Arka planda kar varsa Kurttur, yoksa Husky'dir."**

## Deney Bulguları

Projemizde ResNet18 mimarisi kullanılarak iki farklı durum test edilmiştir:

1.  **Pre-trained (Önceden Eğitilmiş) Model Analizi:**
    *   ImageNet gibi devasa veri setlerinde önceden eğitilmiş (`pretrained=True`) ağırlıklarla başlatılan model, nesne tespiti konusunda halihazırda iyi bir altyapıya sahiptir.
    *   Bu model Husky vs Wolf veri setine ince ayar (fine-tuning) yapıldığında, hedeflenen hayvanın fiziksel özelliklerine odaklanmayı başarıyor. Arka planda kar olsa bile Husky'yi doğru tanıyabiliyor veya hayvan silinmiş arka plan resminde yanılma payı daha düşük oluyor.

2.  **Sıfırdan Eğitilen (No Pre-trained) Model Analizi (Shortcut Learning Tuzağı):**
    *   Ağırlıkları sıfırdan başlatılan (`weights=None`) model, hiçbir ön bilgisi olmadığı için doğrudan bu kasti önyargılı veri setiyle eğitiliyor.
    *   Model, hayvanı öğrenmek yerine arka plan detaylarına odaklanıyor.
    *   **Test Aşaması (\`exchanged_background\`):** Modelleri test edebilmek için araştırmacılar bazı test resimlerinden Husky/Kurt fotoğraflarını dijital olarak **silip şeffaf (beyaz)** yapmışlardır. Geriye sadece karlı ortam veya çimenlik arka plan kalmıştır.
    *   Sıfırdan eğitilen model, resimde hiçbir hayvan olmamasına rağmen, **sadece arka plandaki traktöre / karlı ortama bakarak (Grad-CAM ile tespit edilmiştir) bu resmin bir Kurt (Wolf) olduğunu iddia etmektedir.** Resmin orijinalinde ise bir Husky vardı.

## Grad-CAM İşlevi

Projede `pytorch-grad-cam` kütüphanesi kullanılmıştır. Grad-CAM, tahmin sırasında modelin resmin **tam olarak neresine** (hangi piksellerine) daha çok odaklanarak o sınıfa karar verdiğini kırmızı/mavi ısı haritası şeklinde görselleştirir.

*   Bizim testlerimizde sıfırdan eğitilen moedel için gösterilen Grad-CAM haritasında, odak noktasının hayvanın üzerinde DEĞİL, traktör, otlar veya karlı arka plan üzerinde olduğu açıkça görülmektedir.

## Sonuç

*   **Pretrained (Önceden Eğitilmiş) modeller**, genel özellik çıkartma (feature extraction) kabiliyetleri gelişmiş olduğu için yeni bir göreve (task) uyarlanırken kısayol öğrenimine karşı daha dirençlidir ve mantıklı özelliklere odaklanarak doğru kararlar verebilir.
*   **Sıfırdan (From Scratch) eğitilen modeller** ise yeterli çeşitliliğe sahip olmayan veya önyargılı veri setlerinde kolaylıkla aldanabilir. Veri setinin özellikleri yerine, daha az maliyetli olan "kısayol sinyallerini" (arka plan, etiket izleri vb.) ezberleyerek yanlış sebeplerle doğru sonuçları verebilirler (Clever Hans Etkisi).

## Kullanım

Görselleştirme için aşağıdaki örnek fonksiyon jupyter notebook içerisinde mevcuttur:

```python
visualize_gradcam("/path/to/test/image.jpg", "gerçek_sinif", model, device)
```
