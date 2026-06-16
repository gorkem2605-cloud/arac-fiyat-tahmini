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
2. Şu linke git: https://www.kaggle.com/datasets/alpertemel/turkey-car-market-2020
3. **Download** butonuna bas → `turkey_car_market.csv` inecek
4. Bu dosyayı projenin `data/` klasörüne taşı

Alternatif veri seti (daha büyük):
- https://www.kaggle.com/datasets/omercolakoglu/turkish-market-sales-dataset-with-9000items

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
📂 Veri yükleniyor: data/turkey_car_market.csv
   Ham veri: 9,044 satır, 15 kolon
🔧 Veri temizleniyor...
   Temiz veri: 8,901 satır
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

## 🏫 Okulda Kullanım

Her seferinde:
```bash
# 1. Sanal ortamı aktif et (kurulum yaptıysan)
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# 2. Çalıştır
streamlit run app.py
```

Model zaten `model/` klasöründe olduğu için `train.py`'ı tekrar çalıştırmana gerek yok.

---

## 📁 Klasör Yapısı

```
arac-fiyat-tahmini/
├── app.py              ← Ana uygulama (Streamlit)
├── train.py            ← Model eğitim scripti
├── requirements.txt    ← Kütüphane listesi
├── README.md
├── utils/
│   ├── __init__.py
│   └── predictor.py    ← Tahmin yardımcı modülü
├── data/
│   └── turkey_car_market.csv   ← (Kaggle'dan indirilecek)
└── model/              ← (train.py sonrası otomatik oluşur)
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

## 🔧 Sorun Giderme

**`ModuleNotFoundError` hatası:**
```bash
pip install -r requirements.txt
```

**`FileNotFoundError: data/turkey_car_market.csv`:**
Veri setini `data/` klasörüne koyduğundan emin ol.

**Port meşgul hatası:**
```bash
streamlit run app.py --server.port 8502
```
