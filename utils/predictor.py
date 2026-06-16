import joblib
import json
import os
import pandas as pd
import numpy as np

class CarPricePredictor:
    def __init__(self, model_dir="model"):
        self.model_dir = model_dir
        self.model = None
        self.encoders = None
        self.meta = None
        self.is_loaded = False
        self.load_model()

    def load_model(self):
        try:
            model_path = os.path.join(self.model_dir, "model.pkl")
            encoders_path = os.path.join(self.model_dir, "encoders.pkl")
            meta_path = os.path.join(self.model_dir, "meta.json")

            if not (os.path.exists(model_path) and os.path.exists(encoders_path) and os.path.exists(meta_path)):
                return False

            self.model = joblib.load(model_path)
            self.encoders = joblib.load(encoders_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            self.is_loaded = True
            return True
        except:
            return False

    def predict(self, input_data: dict) -> float:
        if not self.is_loaded:
            raise Exception("Model henüz yüklenmedi!")

        df_in = pd.DataFrame([input_data])
        df_in["arac_yasi"] = 2026 - df_in["model_yil"]

        for col in self.meta["kategorik"]:
            if col in df_in.columns:
                val = str(df_in[col][0]).strip().upper()
                le = self.encoders[col]
                if val in le.classes_:
                    df_in[col] = le.transform([val])[0]
                else:
                    df_in[col] = le.transform([le.classes_[0]])[0]

        feature_cols = self.meta["feature_cols"]
        df_in = df_in[feature_cols]
        pred = self.model.predict(df_in)[0]
        return float(pred)