# 🚗 Araç Fiyat Tahmini

Türkiye ikinci el araç piyasası için makine öğrenmesi tabanlı fiyat tahmin uygulaması.

---

## 🚀 Kurulum (İlk Sefer)

### 1. Python kurulu mu kontrol et
```bash
python --version   # 3.8 veya üzeri olmalı
```
Kurulu değilse → https://www.python.org/downloads/

### 2. Projeyi bir klasöre koy
```
arac-fiyat-tahmini/
├── app.py
├── train.py
├── requirements.txt
├── utils/
│   └── predictor.py
└── data/          ← (boş, veri seti buraya gelecek)
```

### 3. Sanal ortam oluştur (tavsiye edilir)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 4. Kütüphaneleri yükle
```bash
pip install -r requirements.txt
```

---

## 📥 Veri Seti İndirme

**Kaggle'a kayıt gerekiyor (ücretsiz):**

1. https://www.kaggle.com adresine git, hesap aç
2. Şu linke git: https://www.kaggle.com/datasets/oguzarar/turkey-used-car-prices-august-2025
3. **Download** butonuna bas → `cars.csv` inecek
4. Bu dosyayı projenin `data/` klasörüne taşı

---

## 🤖 Model Eğitimi

```bash
python train.py
```

Çıktı:
```
==================================================
  Araç Fiyat Tahmin Modeli - Eğitim
==================================================
📂 Veri yükleniyor: data/cars.csv
   Ham veri: 52.256 satır, 23 kolon
🔧 Veri temizleniyor...
   Temiz veri: 51.450 satır
🤖 Model eğitiliyor...
  XGBoost         → R²=0.9120  MAE=45,230 ₺
  RandomForest    → R²=0.8980  MAE=52,100 ₺
✅ En iyi model: XGBoost
💾 Kaydedildi → model/
🎉 Eğitim tamamlandı!
```

---

## ▶️ Uygulamayı Başlat

```bash
streamlit run app.py
```

Tarayıcı otomatik açılır → http://localhost:8501

---

---

## 📁 Klasör Yapısı

```
arac-fiyat-tahmini/
├── app.py              ← Ana uygulama (Streamlit)
├── train.py            ← Model eğitim scripti
├── requirements.txt    ← Kütüphane listesi
├── README.md           ← Proje dokümanı
├── .gitignore          
├── utils/
│   ├── __init__.py
│   └── predictor.py    ← Tahmin yardımcı modülü
├── data/
│   └── cars.csv        ← (Kaggle'dan indirilecek)
└── model/              
    ├── model.pkl
    ├── encoders.pkl
    └── meta.json
```

---

## ⚙️ Özellikler

- ✅ XGBoost + RandomForest karşılaştırmalı eğitim
- ✅ Otomatik en iyi model seçimi
- ✅ Güven aralığı hesabı
- ✅ Gauge + histogram grafikler
- ✅ Türkçe arayüz
- ✅ İnternet bağlantısı gerekmez (sadece kurulumda)

---
