import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

@st.cache_data
def load_data():
    train_df = pd.read_csv('fraudTrain.csv').sample(n=10000, random_state=42)
    test_df = pd.read_csv('fraudTest.csv').sample(n=2000, random_state=42)
    return train_df, test_df

st.title("💳 Deteksi Fraud")

train_df, test_df = load_data()

# 1. Tampilkan Data Train
st.subheader("Data Training (Sample)")
st.write(train_df.head())

# Training
target_col = 'is_fraud'
X = train_df.drop(target_col, axis=1).select_dtypes(include=['number'])
y = train_df[target_col]
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
model = RandomForestClassifier(n_estimators=50).fit(X_res, y_res)
st.success("Model berhasil dilatih!")

# 2. Tampilkan Hasil Prediksi dengan lebih jelas
if st.button("Prediksi Data Test"):
    X_test = test_df.drop(target_col, axis=1, errors='ignore').select_dtypes(include=['number'])
    pred = model.predict(X_test)
    
    # Membuat DataFrame agar rapi
    hasil_df = pd.DataFrame({'Hasil Prediksi': pred})
    st.subheader("Hasil Prediksi Data Test:")
    st.table(hasil_df.head(100)) # Menampilkan 10 hasil pertama dalam bentuk tabel
