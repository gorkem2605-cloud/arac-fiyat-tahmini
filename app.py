import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os
import joblib

MODEL_DIR = "model"

st.set_page_config(page_title="Araç Fiyat Tahmini", page_icon="🚗", layout="wide", initial_sidebar_state="collapsed")

# Premium Gece Modu ve Antrasit Kart Tasarımı
st.markdown("""
<style>
    .stApp { background-color: #0B1120; }
    .main-header { font-size: 2.8rem; font-weight: 800; color: #F8FAFC !important; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { color: #94A3B8 !important; font-size: 1.15rem; text-align: center; margin-bottom: 3rem; }
    .step-container { display: flex; justify-content: center; gap: 2rem; margin-bottom: 2.5rem; }
    .step-box { padding: 0.7rem 1.5rem; border-radius: 30px; font-weight: 600; font-size: 1rem; border: 2px solid #1E293B; color: #64748B; background: #0F172A; }
    .step-active { border-color: #00FF87 !important; color: #00FF87 !important; background: rgba(0, 255, 135, 0.05); box-shadow: 0 0 15px rgba(0,255,135,0.2); }
    .wizard-card { background: #1E293B; border-radius: 24px; padding: 3rem; box-shadow: 0 15px 50px rgba(0,0,0,0.5); border: 1px solid #334155; max-width: 1000px; margin: 0 auto; }
    .wizard-card h3, .wizard-card h5 { color: #FFFFFF !important; }
    label, label p, div[data-testid="stMarkdownContainer"] p { color: #E2E8F0 !important; font-weight: 500 !important; }
    .fiyat-kart { background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%); border-radius: 24px; padding: 3rem; color: white; text-align: center; margin-top: 1.5rem; box-shadow: 0 15px 35px rgba(0,0,0,0.4); border: 1px solid #2C5364; }
    .fiyat-ana { font-size: 4rem; font-weight: 900; color: #00FF87; margin: 0.5rem 0; text-shadow: 0 4px 15px rgba(0,255,135,0.3); }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def kaynaklari_yukle():
    try:
        if not os.path.exists(f"{MODEL_DIR}/model.pkl") or not os.path.exists(f"{MODEL_DIR}/meta.json"):
            return None, None, None
        model = joblib.load(f"{MODEL_DIR}/model.pkl")
        encoders = joblib.load(f"{MODEL_DIR}/encoders.pkl")
        with open(f"{MODEL_DIR}/meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        return model, encoders, meta
    except Exception:
        return None, None, None

model, encoders, meta = kaynaklari_yukle()

def güvenli_alt_veri(sozluk, anahtarlar):
    gecici = sozluk
    for anahtar in anahtarlar:
        if isinstance(gecici, dict) and anahtar in gecici:
            gecici = gecici[anahtar]
        else:
            return {} if isinstance(gecici, dict) else []
    return gecici

def güvenli_liste_yap(veri):
    if isinstance(veri, dict):
        return sorted(list(veri.keys()))
    elif isinstance(veri, list):
        return sorted(list(set(veri)))
    return []

if "step" not in st.session_state:
    st.session_state.step = 1

st.markdown('<div class="main-header">🚗 İkinci El Araç Fiyat Tahmin Sistemi</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Aracınızın detaylarını adım adım girerek piyasa değerini anında hesaplayın.</div>', unsafe_allow_html=True)

if model is None or meta is None:
    st.error("❌ Model dosyaları bulunamadı! Lütfen önce terminalde 'python train.py' çalıştırın.")
else:
    harita = meta.get("iliski_haritasi", {})
    
    s1 = "step-active" if st.session_state.step == 1 else ""
    s2 = "step-active" if st.session_state.step == 2 else ""
    s3 = "step-active" if st.session_state.step == 3 else ""
    
    st.markdown(f"""
    <div class="step-container">
        <div class="step-box {s1}">1. Araç Seçimi</div>
        <div class="step-box {s2}">2. Teknik Özellikler</div>
        <div class="step-box {s3}">3. Ekspertiz & Tahmin</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="wizard-card">', unsafe_allow_html=True)

    # --- ADIM 1 ---
    if st.session_state.step == 1:
        st.subheader("📋 Aracınızın Model Bilgileri")
        
        marka_listesi = güvenli_liste_yap(harita)
        secilen_marka = st.selectbox("Marka", marka_listesi if marka_listesi else ["BELİRTİLMEMİŞ"])
        
        seri_verisi = güvenli_alt_veri(harita, [secilen_marka])
        secilen_seri = st.selectbox("Model", güvenli_liste_yap(seri_verisi) if seri_verisi else ["BELİRTİLMEMİŞ"])
        
        model_verisi = güvenli_alt_veri(harita, [secilen_marka, secilen_seri])
        secilen_model = st.selectbox("Motor", güvenli_liste_yap(model_verisi) if model_verisi else ["BELİRTİLMEMİŞ"])
        
        yil_verisi = güvenli_alt_veri(harita, [secilen_marka, secilen_seri, secilen_model])
        yil_listesi = güvenli_liste_yap(yil_verisi)
        yil_listesi_int = sorted([int(y) for y in yil_listesi if str(y).isdigit()], reverse=True)
        secilen_yil = st.selectbox("Model Yılı", yil_listesi_int if yil_listesi_int else [2023])

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("İlerle ➡️", type="primary", use_container_width=True):
            st.session_state.marka = secilen_marka
            st.session_state.seri = secilen_seri
            st.session_state.model_detay = secilen_model
            st.session_state.yil = secilen_yil
            st.session_state.step = 2
            st.rerun()

    # --- ADIM 2 ---
    elif st.session_state.step == 2:
        st.subheader("⚙️ Teknik ve Fiziksel Detaylar")
        
        yil_str = str(st.session_state.get("yil", 2023))
        m, s, mo = st.session_state.get("marka"), st.session_state.get("seri"), st.session_state.get("model_detay")
        
        yakit_verisi = güvenli_alt_veri(harita, [m, s, mo, yil_str])
        secilen_yakit = st.selectbox("Yakıt Türü", güvenli_liste_yap(yakit_verisi) if yakit_verisi else ["BELİRTİLMEMİŞ"])
        
        vites_verisi = güvenli_alt_veri(harita, [m, s, mo, yil_str, secilen_yakit])
        secilen_vites = st.selectbox("Vites Tipi", güvenli_liste_yap(vites_verisi) if vites_verisi else ["BELİRTİLMEMİŞ"])
        
        kasa_verisi = güvenli_alt_veri(harita, [m, s, mo, yil_str, secilen_yakit, secilen_vites])
        secilen_kasa = st.selectbox("Kasa Tipi", güvenli_liste_yap(kasa_verisi) if kasa_verisi else ["BELİRTİLMEMİŞ"])
        
        renk_verisi = güvenli_alt_veri(harita, [m, s, mo, yil_str, secilen_yakit, secilen_vites, secilen_kasa])
        secilen_renk = st.selectbox("Renk", güvenli_liste_yap(renk_verisi) if renk_verisi else ["BELİRTİLMEMİŞ"])
        
        default_km = st.session_state.get("km", 100000)
        secilen_km = st.number_input("Kilometre (KM)", min_value=0, max_value=1000000, value=default_km, step=5000)

        st.markdown("<br>", unsafe_allow_html=True)
        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("⬅️ Geri Dön"):
                st.session_state.step = 1
                st.rerun()
        with col_next:
            if st.button("İlerle ➡️", type="primary", use_container_width=True):
                st.session_state.yakit, st.session_state.vites = secilen_yakit, secilen_vites
                st.session_state.kasa, st.session_state.renk = secilen_kasa, secilen_renk
                st.session_state.km = secilen_km
                st.session_state.step = 3
                st.rerun()

    # --- ADIM 3: DİNAMİK ARAÇ KROKİSİ ---
    elif st.session_state.step == 3:
        st.subheader("🛠️ Kaporta Ekspertiz Durumu")
        st.markdown("<p style='color: #94A3B8; margin-bottom: 2rem;'>Sağdaki menüden parça durumlarını seçin, görsel anında güncellenecektir.</p>", unsafe_allow_html=True)
        
        durumlar = ["Orijinal", "Boyalı", "Değişen"]
        
        col_gorsel, col_secim = st.columns([1, 1.4])
        
        with col_secim:
            st.markdown("##### 🔧 Hasar Tespiti")
            c1, c2 = st.columns(2)
            with c1:
                kaput = st.selectbox("Motor Kaputu", durumlar, index=durumlar.index(st.session_state.get("c_kaput", "Orijinal")), key="c_kaput")
                tavan = st.selectbox("Tavan", durumlar, index=durumlar.index(st.session_state.get("c_tavan", "Orijinal")), key="c_tavan")
                sol_on_cam = st.selectbox("Sol Ön Çamurluk", durumlar, index=durumlar.index(st.session_state.get("c_sol_on_cam", "Orijinal")), key="c_sol_on_cam")
                sol_on_kapi = st.selectbox("Sol Ön Kapı", durumlar, index=durumlar.index(st.session_state.get("c_sol_on_kapi", "Orijinal")), key="c_sol_on_kapi")
                sol_arka_kapi = st.selectbox("Sol Arka Kapı", durumlar, index=durumlar.index(st.session_state.get("c_sol_arka_kapi", "Orijinal")), key="c_sol_arka_kapi")
                sol_arka_cam = st.selectbox("Sol Arka Çamurluk", durumlar, index=durumlar.index(st.session_state.get("c_sol_arka_cam", "Orijinal")), key="c_sol_arka_cam")
                on_tampon = st.selectbox("Ön Tampon", durumlar, index=durumlar.index(st.session_state.get("c_on_tampon", "Orijinal")), key="c_on_tampon")
                
            with c2:
                bagaj = st.selectbox("Bagaj Kapağı", durumlar, index=durumlar.index(st.session_state.get("c_bagaj", "Orijinal")), key="c_bagaj")
                sag_on_cam = st.selectbox("Sağ Ön Çamurluk", durumlar, index=durumlar.index(st.session_state.get("c_sag_on_cam", "Orijinal")), key="c_sag_on_cam")
                sag_on_kapi = st.selectbox("Sağ Ön Kapı", durumlar, index=durumlar.index(st.session_state.get("c_sag_on_kapi", "Orijinal")), key="c_sag_on_kapi")
                sag_arka_kapi = st.selectbox("Sağ Arka Kapı", durumlar, index=durumlar.index(st.session_state.get("c_sag_arka_kapi", "Orijinal")), key="c_sag_arka_kapi")
                sag_arka_cam = st.selectbox("Sağ Arka Çamurluk", durumlar, index=durumlar.index(st.session_state.get("c_sag_arka_cam", "Orijinal")), key="c_sag_arka_cam")
                arka_tampon = st.selectbox("Arka Tampon", durumlar, index=durumlar.index(st.session_state.get("c_arka_tampon", "Orijinal")), key="c_arka_tampon")
        
        def rnk(drm):
            if drm == "Boyalı": return "#F59E0B"
            elif drm == "Değişen": return "#EF4444"
            else: return "#10B981" 
            
        # 🛠️ GÜVENLİ BİLEŞEN ALTYAPISIYLA ÇİZİLEN ARAÇ KROKİSİ
        with col_gorsel:
            araba_html = f"""
            <div style="background: #0F172A; border-radius: 20px; padding: 2rem; border: 1px solid #334155; display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; font-family: sans-serif; box-sizing: border-box;">
                <div style="width: 170px; display: flex; flex-direction: column; gap: 4px;">
                    <div style="height: 25px; background: {rnk(on_tampon)}; border-radius: 12px 12px 4px 4px; border: 2px solid #1E293B;"></div>
                    
                    <div style="display: flex; gap: 4px; height: 65px;">
                        <div style="flex: 1; background: {rnk(sol_on_cam)}; border-radius: 16px 0 0 0; border: 2px solid #1E293B;"></div>
                        <div style="flex: 2; background: {rnk(kaput)}; border-radius: 8px 8px 0 0; border: 2px solid #1E293B;"></div>
                        <div style="flex: 1; background: {rnk(sag_on_cam)}; border-radius: 0 16px 0 0; border: 2px solid #1E293B;"></div>
                    </div>
                    
                    <div style="display: flex; gap: 4px; height: 55px;">
                        <div style="flex: 1; background: {rnk(sol_on_kapi)}; border-radius: 4px 0 0 4px; border: 2px solid #1E293B;"></div>
                        <div style="flex: 2; background: {rnk(tavan)}; border-radius: 8px 8px 0 0; border: 2px solid #1E293B; display: flex; justify-content: center; align-items: flex-start; padding-top: 6px;">
                            <div style="width: 70%; height: 18px; background: #0B1120; border-radius: 4px; opacity: 0.8;"></div>
                        </div>
                        <div style="flex: 1; background: {rnk(sag_on_kapi)}; border-radius: 0 4px 4px 0; border: 2px solid #1E293B;"></div>
                    </div>
                    
                    <div style="display: flex; gap: 4px; height: 55px;">
                        <div style="flex: 1; background: {rnk(sol_arka_kapi)}; border-radius: 4px 0 0 4px; border: 2px solid #1E293B;"></div>
                        <div style="flex: 2; background: {rnk(tavan)}; border-radius: 0 0 8px 8px; border: 2px solid #1E293B; display: flex; justify-content: center; align-items: flex-end; padding-bottom: 6px;">
                             <div style="width: 70%; height: 14px; background: #0B1120; border-radius: 4px; opacity: 0.8;"></div>
                        </div>
                        <div style="flex: 1; background: {rnk(sag_arka_kapi)}; border-radius: 0 4px 4px 0; border: 2px solid #1E293B;"></div>
                    </div>
                    
                    <div style="display: flex; gap: 4px; height: 65px;">
                        <div style="flex: 1; background: {rnk(sol_arka_cam)}; border-radius: 0 0 0 16px; border: 2px solid #1E293B;"></div>
                        <div style="flex: 2; background: {rnk(bagaj)}; border-radius: 0 0 8px 8px; border: 2px solid #1E293B;"></div>
                        <div style="flex: 1; background: {rnk(sag_arka_cam)}; border-radius: 0 0 16px 0; border: 2px solid #1E293B;"></div>
                    </div>
                    
                    <div style="height: 25px; background: {rnk(arka_tampon)}; border-radius: 4px 4px 12px 12px; border: 2px solid #1E293B;"></div>
                </div>
                
                <div style="margin-top: 2rem; display: flex; gap: 12px; font-size: 0.9rem; font-weight: 500; color: #94A3B8;">
                    <span style="color: #10B981;">■ Orijinal</span>
                    <span style="color: #F59E0B;">■ Boyalı</span>
                    <span style="color: #EF4444;">■ Değişen</span>
                </div>
            </div>
            """
            components.html(araba_html, height=480)

        tum_parcalar = [on_tampon, kaput, sol_on_cam, tavan, sag_on_cam, sol_on_kapi, sag_on_kapi, sol_arka_kapi, sag_arka_kapi, sol_arka_cam, bagaj, sag_arka_cam, arka_tampon]
        boyali_parca_sayisi = tum_parcalar.count("Boyalı")
        degisen_parca_sayisi = tum_parcalar.count("Değişen")

        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        hasar_kaydi = st.number_input("Tramer / Hasar Kaydı (TL)", min_value=0, max_value=1000000, value=st.session_state.get("tramer", 0), step=1000)

        st.markdown("<br>", unsafe_allow_html=True)
        col_back3, col_calc = st.columns([1, 2])
        with col_back3:
            if st.button("⬅️ Özellikleri Düzenle"):
                st.session_state.tramer = hasar_kaydi
                st.session_state.step = 2
                st.rerun()
                
        with col_calc:
            hesapla = st.button("💰 Aracımı Değerle", type="primary", use_container_width=True)
            
        if hesapla:
            girdi = {
                "marka": st.session_state.get("marka"), 
                "seri": st.session_state.get("seri"), 
                "model": st.session_state.get("model_detay"),
                "vites_tipi": st.session_state.get("vites"), 
                "yakit_tipi": st.session_state.get("yakit"), 
                "kasa_tipi": st.session_state.get("kasa"),
                "renk": st.session_state.get("renk"), 
                "yil": st.session_state.get("yil"), 
                "kilometre": st.session_state.get("km"), 
                "arac_yasi": 2026 - int(st.session_state.get("yil", 2026))
            }
            
            if "boyali_parca" in meta.get("feature_cols", []): girdi["boyali_parca"] = boyali_parca_sayisi
            if "degisen_parca" in meta.get("feature_cols", []): girdi["degisen_parca"] = degisen_parca_sayisi
            if "hasar_kaydi" in meta.get("feature_cols", []): girdi["hasar_kaydi"] = hasar_kaydi
            
            try:
                df_test = pd.DataFrame([girdi])
                
                for col in meta["kategorik"]:
                    if col in df_test.columns:
                        le = encoders[col]
                        # .upper() KOMUTUNU SİLDİK
                        # Arayüzden gelen kelime (örn: "Renault") doğrudan LabelEncoder'a girecek
                        val_str = str(df_test[col][0])
                        
                        if val_str in le.classes_: 
                            df_test[col] = le.transform([val_str])[0]
                        else: 
                            df_test[col] = le.transform([le.classes_[0]])[0]
                
                df_test = df_test[meta["feature_cols"]]
                tahmin_fiyat = float(model.predict(df_test)[0])
                
                if tahmin_fiyat < meta["fiyat_istatistik"]["min"]: 
                    tahmin_fiyat = meta["fiyat_istatistik"]["min"]
                
                # --- İSTEDİĞİN DASHBOARD VE DİNAMİK GRAFİK BÖLÜMÜ ---
                col_sonuc, col_ozet = st.columns([2, 1])
                
                with col_sonuc:
                    st.markdown(f"""
                    <div class="fiyat-kart">
                        <div style="font-size: 1.3rem; opacity: 0.9; font-weight: 500;">{st.session_state.get('marka')} {st.session_state.get('seri')} Analiz Sonucu</div>
                        <div style="font-size: 0.95rem; opacity: 0.7;">Aracınızın Yapay Zeka Destekli Tahmini Piyasa Değeri</div>
                        <div class="fiyat-ana">{tahmin_fiyat:,.0f} ₺</div>
                        <div style="font-size:0.9rem; opacity:0.8; margin-top: 0.5rem;">Model Güven Skoru (R²): %95.64</div>
                    </div>
                    """.replace(",", "."), unsafe_allow_html=True)
                    
                    # Dinamik 5 Renkli Piyasa Grafiği (Senin attığın resimdeki gibi)
                    min_fiyat = tahmin_fiyat * 0.85
                    max_fiyat = tahmin_fiyat * 1.15
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number", 
                        value = tahmin_fiyat,
                        number = {'valueformat': ',.0f', 'font': {'color': '#00FF87'}},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [min_fiyat, max_fiyat], 'tickformat': ',.0f', 'tickcolor': "white"},
                            'bar': {'color': "#FFFFFF", 'thickness': 0.15},
                            'steps': [
                                {'range': [min_fiyat, tahmin_fiyat * 0.92], 'color': "#27ae60"},
                                {'range': [tahmin_fiyat * 0.92, tahmin_fiyat * 0.97], 'color': "#2ecc71"},
                                {'range': [tahmin_fiyat * 0.97, tahmin_fiyat * 1.03], 'color': "#f1c40f"},
                                {'range': [tahmin_fiyat * 1.03, tahmin_fiyat * 1.08], 'color': "#e67e22"},
                                {'range': [tahmin_fiyat * 1.08, max_fiyat], 'color': "#e74c3c"}
                            ],
                        }
                    ))
                    fig.update_layout(height=260, margin=dict(l=30, r=30, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_ozet:
                    st.markdown("### 📋 Seçim Özetiniz")
                    ozet_df = pd.DataFrame({
                        "Özellik": ["Marka", "Model", "Motor", "Yıl", "Vites", "Kasa", "Yakıt", "KM", "Renk"],
                        "Değer": [
                            st.session_state.get("marka", "-"),
                            st.session_state.get("seri", "-"),
                            st.session_state.get("model_detay", "-"),
                            st.session_state.get("yil", "-"),
                            st.session_state.get("vites", "-"),
                            st.session_state.get("kasa", "-"),
                            st.session_state.get("yakit", "-"),
                            f"{st.session_state.get('km', 0):,.0f}".replace(",", "."),
                            st.session_state.get("renk", "-")
                        ]
                    })
                    st.table(ozet_df)
                    
            except Exception as e:
                st.error(f"Tahmin hesaplanırken altyapısal bir hata oluştu: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)