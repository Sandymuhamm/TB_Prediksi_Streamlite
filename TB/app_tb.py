import streamlit as st
import pandas as pd
import joblib

# Load model yang sudah disimpan
model = joblib.load('model_tb.pkl')

st.title("Prediksi Risiko Tuberculosis (TB)")

# Input user
jenis_kelamin = st.selectbox("Jenis Kelamin", options=["Laki-Laki", "Perempuan"])
kontak_serumah = st.selectbox("Kontak Serumah dengan penderita TB?", options=["YA", "TIDAK"])
dm = st.selectbox("Apakah pasien menderita Diabetes Mellitus (DM)?", options=["YA", "TIDAK"])
lansia = st.selectbox("Apakah pasien berusia >60 tahun?", options=["YA", "TIDAK"])
ibu_hamil = st.selectbox("Apakah pasien ibu hamil?", options=["YA", "TIDAK"])
perokok = st.selectbox("Apakah pasien perokok?", options=["YA", "TIDAK"])

# Mapping input ke angka
map_jenis_kelamin = {"Laki-Laki":1, "Perempuan":0}
map_binary = {"YA":1, "TIDAK":0}

data_input = pd.DataFrame({
    'Jenis Kelamin': [map_jenis_kelamin[jenis_kelamin]],
    'Kontak Serumah': [map_binary[kontak_serumah]],
    'DM': [map_binary[dm]],
    'Lansia >60 th': [map_binary[lansia]],
    'Ibu Hamil': [map_binary[ibu_hamil]],
    'Perokok': [map_binary[perokok]],
})

if st.button("Prediksi"):
    prediksi = model.predict(data_input)[0]
    prob = model.predict_proba(data_input)[0][1] * 100

    hasil = "SAKIT" if prediksi == 1 else "SEHAT"
    st.write(f"**Hasil Prediksi:** {hasil}")
    st.write(f"**Peluang positif TB:** {prob:.2f}%")