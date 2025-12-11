import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
import joblib
from ANFIS import ANFIS
from ModeloSimple import ANN_Simple

# Configuración
st.set_page_config(page_title="Predicción Calidad Aire", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🌫️ Sistema Inteligente de Predicción PM2.5</h1>", unsafe_allow_html=True)

# PESTAÑAS
tab1, tab2 = st.tabs(["📊 Análisis Comparativo (Test)", "🧪 Predicción en Vivo (Manual)"])

# =========================================================
# TAB 1: GRÁFICAS COMPARATIVAS (LEE EL CSV)
# =========================================================
with tab1:
    try:
        df = pd.read_csv('resultados_test.csv')
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        st.subheader("Rendimiento del Modelo en Test Set")
        cols_modelos = [c for c in df.columns if c not in ['Fecha', 'Real']]
        seleccion = st.multiselect("Comparar Modelos:", cols_modelos, default=cols_modelos)
        
        # Gráfica
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Fecha'], y=df['Real'], mode='lines', name='Real', line=dict(color='black')))
        
        for mod in seleccion:
            fig.add_trace(go.Scatter(x=df['Fecha'], y=df[mod], mode='lines', name=mod, line=dict(dash='dot')))
            
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas
        if seleccion:
            cols = st.columns(len(seleccion))
            for idx, mod in enumerate(seleccion):
                rmse = np.sqrt(mean_squared_error(df['Real'], df[mod]))
                r2 = r2_score(df['Real'], df[mod])
                cols[idx].metric(label=f"RMSE {mod}", value=f"{rmse:.2f}", delta=f"R2: {r2:.3f}")
                
    except FileNotFoundError:
        st.error("⚠️ Ejecuta primero 'procesar_modelos.py' para generar los datos.")

# =========================================================
# TAB 2: PREDICCIÓN MANUAL EN VIVO (USA LOS .PTH)
# =========================================================
with tab2:
    st.subheader("Simulador de Predicción (t+1 hora)")
    
    # Verificar archivos necesarios
    try:
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
    except:
        st.error("Faltan los escaladores (.pkl). Ejecuta 'procesar_modelos.py'.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    
    # Inputs del Usuario (Datos Físicos Reales)
    with col1:
        pm10 = st.number_input("PM10 Actual", value=45.0)
        pm10_ant = st.number_input("PM10 (Hace 1h)", value=42.0)
        temp = st.number_input("Temperatura (°C)", value=22.5)
        
    with col2:
        ozono = st.number_input("Ozono (ppb)", value=30.0)
        co = st.number_input("Monóxido Carbono (ppb)", value=350.0)
        humedad = st.number_input("Humedad Relativa (%)", value=40.0)
        
    with col3:
        pm25_act = st.number_input("PM2.5 Actual", value=25.0)
        pm25_ant1 = st.number_input("PM2.5 (Hace 1h)", value=23.0)
        pm25_ant2 = st.number_input("PM2.5 (Hace 2h)", value=20.0)

    # Botón de Predicción
    if st.button("🔮 Calcular Predicción Futura", type="primary"):
        # 1. Calcular Variables Derivadas (Tu ingeniería de datos)
        prom_3h = (pm25_act + pm25_ant1 + pm25_ant2) / 3
        cambio_pm25 = pm25_act - pm25_ant1
        cambio_pm10 = pm10 - pm10_ant
        
        # 2. Crear Vector de Entrada (Orden exacto del entrenamiento)
        input_data = np.array([[
            pm10, ozono, co, temp, humedad, 
            pm25_act, prom_3h, cambio_pm25, cambio_pm10
        ]])
        
        # 3. Normalizar
        input_norm = scaler_X.transform(input_data)
        input_tensor = torch.tensor(input_norm, dtype=torch.float32)
        
        # 4. Cargar Modelos y Predecir
        resultados = {}
        
        # ANFIS
        try:
            modelo_anfis = ANFIS(n_entr=9, n_curvas=3, n_reglas=35)
            modelo_anfis.load_state_dict(torch.load('modelo_anfis.pth'))
            modelo_anfis.eval()
            with torch.no_grad():
                pred_anfis_norm = modelo_anfis(input_tensor).numpy()
                pred_anfis = scaler_y.inverse_transform(pred_anfis_norm)[0][0]
                resultados['ANFIS_EA'] = max(0, pred_anfis)
        except:
            resultados['ANFIS_EA'] = "Error cargando modelo"

        # ANN SIMPLE
        try:
            modelo_ann = ANN_Simple(n_entr=9)
            modelo_ann.load_state_dict(torch.load('modelo_ann.pth'))
            modelo_ann.eval()
            with torch.no_grad():
                pred_ann_norm = modelo_ann(input_tensor).numpy()
                pred_ann = scaler_y.inverse_transform(pred_ann_norm)[0][0]
                resultados['ANN_Simple'] = max(0, pred_ann)
        except:
            resultados['ANN_Simple'] = "Error cargando modelo"
            
        # 5. Mostrar Resultados
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        
        res_col1.success(f"### ANFIS Predice:\n# {resultados['ANFIS_EA']:.2f} ug/m3")
        res_col2.warning(f"### ANN Simple Predice:\n# {resultados['ANN_Simple']:.2f} ug/m3")