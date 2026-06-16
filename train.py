import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

DATA_PATH = "data/cars.csv"
MODEL_DIR = "model"

KATEGORIK = ["marka", "seri", "model", "vites_tipi", "yakit_tipi", "kasa_tipi", "renk"]
SAYISAL = ["yil", "kilometre", "arac_yasi", "boyali_parca", "degisen_parca", "hasar_kaydi"]
HEDEF = "fiyat"

def train_model():
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # DÜZELTME 1: Sütun adlarındaki Türkçe karakterleri tamamen İngilizceye çeviriyoruz
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df.columns = df.columns.str.replace('ı', 'i').str.replace('ğ', 'g').str.replace('ü', 'u').str.replace('ş', 's').str.replace('ö', 'o').str.replace('ç', 'c')
    
    # DÜZELTME 2: Gerekli ana sütunların veri setinde olup olmadığını kontrol et
    eksik_sutunlar = [col for col in [HEDEF, "yil", "kilometre"] if col not in df.columns]
    if eksik_sutunlar:
        print(f"❌ DİKKAT: Veri setinde şu kritik sütunlar eksik: {eksik_sutunlar}")
        return

    # Veri Temizleme (Fiyat, Kilometre ve Yaş)
    df['fiyat'] = pd.to_numeric(df['fiyat'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
    df['kilometre'] = pd.to_numeric(df['kilometre'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
    df['arac_yasi'] = 2026 - pd.to_numeric(df['yil'], errors='coerce') 
    
    # Ekspertiz sütunlarını hatasız oluşturma
    # DÜZELTME 3: Sütunlar yoksa çökmek yerine 0 atıyoruz
    if 'tramer' in df.columns:
        df['hasar_kaydi'] = pd.to_numeric(df['tramer'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce').fillna(0)
    else:
        df['hasar_kaydi'] = 0

    if 'boya_degisen' in df.columns:
        # DÜZELTME 4: Regex artık büyük/küçük harf veya Türkçe karakter umursamadan doğru sayıyı çekecek
        df['boyali_parca'] = df['boya_degisen'].astype(str).str.extract(r'(?i)(\d+)\s*boyal[ıi]').astype(float).fillna(0)
        df['degisen_parca'] = df['boya_degisen'].astype(str).str.extract(r'(?i)(\d+)\s*de[gğ]i[sş]en').astype(float).fillna(0)
    else:
        df['boyali_parca'] = 0
        df['degisen_parca'] = 0
        
    # Boş veya bozuk verileri at
    df = df.dropna(subset=[HEDEF] + SAYISAL)

    # --- DİNAMİK MENÜ HARİTASI OLUŞTURMA ---
    iliski_haritasi = {}
    
    grup_sutunlari = ["marka", "seri", "model", "yil", "yakit_tipi", "vites_tipi", "kasa_tipi", "renk"]
    benzersiz_df = df[grup_sutunlari].drop_duplicates()
    
    for _, row in benzersiz_df.iterrows():
        marka = str(row['marka']).strip()
        seri = str(row['seri']).strip()
        model_motor = str(row['model']).strip()
        yil = str(int(row['yil'])) if pd.notnull(row['yil']) else "2023"
        yakit = str(row['yakit_tipi']).strip()
        vites = str(row['vites_tipi']).strip()
        kasa = str(row['kasa_tipi']).strip()
        renk = str(row['renk']).strip()
        
        if marka not in iliski_haritasi: iliski_haritasi[marka] = {}
        if seri not in iliski_haritasi[marka]: iliski_haritasi[marka][seri] = {}
        if model_motor not in iliski_haritasi[marka][seri]: iliski_haritasi[marka][seri][model_motor] = {}
        if yil not in iliski_haritasi[marka][seri][model_motor]: iliski_haritasi[marka][seri][model_motor][yil] = {}
        if yakit not in iliski_haritasi[marka][seri][model_motor][yil]: iliski_haritasi[marka][seri][model_motor][yil][yakit] = {}
        if vites not in iliski_haritasi[marka][seri][model_motor][yil][yakit]: iliski_haritasi[marka][seri][model_motor][yil][yakit][vites] = {}
        if kasa not in iliski_haritasi[marka][seri][model_motor][yil][yakit][vites]: iliski_haritasi[marka][seri][model_motor][yil][yakit][vites][kasa] = []
        
        if renk not in iliski_haritasi[marka][seri][model_motor][yil][yakit][vites][kasa]:
            iliski_haritasi[marka][seri][model_motor][yil][yakit][vites][kasa].append(renk)
    
    # Kodlama (Encoders) - Kategorik kelimeleri XGBoost için sayılara çevir
    encoders = {}
    for col in KATEGORIK:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Model Eğitimi
    X = df[KATEGORIK + SAYISAL]
    y = df[HEDEF]
    model = XGBRegressor(random_state=42)
    model.fit(X, y)
    
    # Dosyaları Kaydet
    joblib.dump(model, f"{MODEL_DIR}/model.pkl")
    joblib.dump(encoders, f"{MODEL_DIR}/encoders.pkl")
    
    # Meta Dosyası (app.py için harita)
    meta = {
        "feature_cols": KATEGORIK + SAYISAL,
        "kategorik": KATEGORIK,
        "fiyat_istatistik": {"min": float(y.min()), "max": float(y.max()), "ort": float(y.mean())},
        "iliski_haritasi": iliski_haritasi
    }
    with open(f"{MODEL_DIR}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
        
    print("✅ Model başarıyla eğitildi ve dosyalar kaydedildi!")

if __name__ == "__main__":
    train_model()