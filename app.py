import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
import joblib
from PIL import Image
import os
from ANFIS import ANFIS
from ModeloSimple import ANN_Simple

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Predicción de la Calidad del Aire", 
    layout="wide", 
    page_icon="🍃"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2em; 
        color: #3e4a57; 
        text-align: center; 
        font-weight: 800;
    }
            
    .sub-text {
        text-align: center; 
        color: #555; 
        font-style: italic;
    }
   
    .info-box {
        background: linear-gradient(135deg, #e0f7fa 0%, #80deea 100%);
        padding: 25px; 
        border-radius: 15px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #006064;
    }
    
    /* Tabla de calidad del aire */        
    .tg {border-collapse: collapse; width: 100%; border-radius: 10px; overflow: hidden; box-shadow: 0 0 20px rgba(0,0,0,0.1);}
    .tg th {background-color: #3e4a57; color: white; padding: 15px; text-align: center;}
    .tg td {padding: 15px; border-bottom: 1px solid #ddd;}
    .tg tr:hover {background-color: #f5f5f5;}
    </style>
""", unsafe_allow_html=True)

# --- TÍTULO PRINCIPAL ---
st.markdown('<div class="main-header">Sistema de Predicción de contaminantes en el aire</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Análisis comparativo: <b>Neuro-Difuso Evolutivo (ANFIS)</b> vs <b>Red Neuronal Clásica (ANN)</b></div>', unsafe_allow_html=True)
st.markdown("---")

# --- FUNCIÓN SEMÁFORO DE CALIDAD ---
def obtener_calidad(valor):
    if valor <= 12.0: return "Buena", "#2ECC71" 
    elif valor <= 35.4: return "Moderada", "#F1C40F" 
    elif valor <= 55.4: return "Mala", "#E67E22" 
    elif valor <= 150.4: return "Muy Mala", "#E74C3C" 
    else: return "Peligrosa", "#8E44AD" 

# --- PESTAÑAS DEL SISTEMA ---
tab_intro, tab_dashboard, tab_train, tab_live = st.tabs([
    "🏠 Inicio", 
    "📊 Dashboard Comparativo", 
    "🖼️ Gráficas de Entrenamiento", 
    "🧪 Simulador"
])

# ==============================================================================
# 1. PESTAÑA DE INTRODUCCIÓN 
# ==============================================================================
with tab_intro:
    # Título de la sección
    st.markdown("### Objetivo del Proyecto")
    st.markdown("""
    El presente proyecto tiene como finalidad el **diseño, implementación y evaluación comparativa** de un sistema híbrido de inteligencia artificial para el pronóstico de calidad del aire en la Ciudad de México.

    Específicamente, se propone un modelo **ANFIS (Adaptive Neuro-Fuzzy Inference System)** cuyos hiperparámetros y estructura de reglas han sido optimizados mediante **Algoritmos Genéticos (Computación Evolutiva)**.

    Este enfoque propuesto se confronta con un modelo benchmark de **Red Neuronal Artificial (ANN)** clásica para validar dos hipótesis:
    1.  **Precisión:** La capacidad del sistema híbrido para reducir el Error Cuadrático Medio (RMSE) en series de tiempo no lineales y caóticas.
    2.  **Interpretabilidad:** La ventaja de utilizar un modelo de *"Caja Blanca"* (reglas legibles) frente a la *"Caja Negra"* de las redes neuronales tradicionales para la toma de decisiones ambientales.
    """)
    
    st.markdown("---")

    # Sección de Modelos 
    st.markdown("### Arquitecturas Comparadas")
    
    c_info1, c_info2 = st.columns(2)
    
    with c_info1:
        st.markdown("#### 1. ANFIS (Neuro-Difuso)")
        st.info("""
        **"Adaptive Neuro-Fuzzy Inference System"**
        
        Es un modelo híbrido que combina:
        * **Lógica Difusa:** Maneja la incertidumbre y el razonamiento humano mediante reglas *"Si-Entonces"* (ej: *Si la Temperatura es Alta, entonces el Ozono sube*).
        * **Redes Neuronales:** Ajustan los parámetros de las funciones de pertenencia para minimizar el error.
        
        * **Ventaja:** Es un modelo de **"Caja Blanca"**. Podemos entender las reglas que generó para tomar decisiones.
        """)

    with c_info2:
        st.markdown("#### 2. ANN (Red Neuronal)")
        st.warning("""
        **"Artificial Neural Network (MLP)"**
        
        Es el estándar clásico del aprendizaje profundo:
        * Utiliza capas de neuronas interconectadas y funciones de activación no lineales (ReLU).
        * Aprende patrones matemáticos complejos mediante Backpropagation.
        
        * **Desventaja:** Es un modelo de **"Caja Negra"**. Aunque es muy potente, es difícil explicar matemáticamente por qué predijo un valor específico.
        """)

    st.markdown("---")

    # Sección de Variables Físicas
    st.markdown("### Variable Objetivo: PM2.5")
    
    st.markdown("""
    El sistema se centra en predecir el **Material Particulado Fino (PM2.5)**, definido como partículas con un diámetro aerodinámico $\le 2.5 \mu m$.
    
    * **¿Por qué es crítico?** A diferencia del PM10 (polvo), el PM2.5 es lo suficientemente pequeño para cruzar la barrera alvéolo-capilar y entrar al torrente sanguíneo, representando el mayor riesgo epidemiológico actual.
    * **Unidad de Medida:** $\mu g/m^3$ (Microgramos por metro cúbico). Indica la masa de partículas suspendidas en un volumen de aire unitario.
    """)

    st.markdown("---")

    # Tabla de Referencia (Semáforo)
    st.markdown("#### Escala de Referencia (Índice de Calidad del Aire)")
    
    tabla_html = """
    <table class="tg">
    <thead>
      <tr>
        <th class="tg-header">Concentración (<span style="text-transform:none">µg/m³</span>)</th>
        <th class="tg-header">Categoría</th>
        <th class="tg-header">Impacto Sanitario</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background-color:#D5F5E3;">
        <td style="text-align:center;font-weight:bold;">0 - 12.0</td>
        <td style="text-align:center;color:#196F3D;font-weight:bold;">BUENA</td>
        <td>Riesgo mínimo o nulo para la salud.</td>
      </tr>
      <tr style="background-color:#fcf0bd;">
        <td style="text-align:center;font-weight:bold;">12.1 - 35.4</td>
        <td style="text-align:center;color:#cfa913;font-weight:bold;">MODERADA</td>
        <td>Grupos sensibles deben considerar limitar el esfuerzo prolongado.</td>
      </tr>
      <tr style="background-color:#ffd491;">
        <td style="text-align:center;font-weight:bold;">35.5 - 55.4</td>
        <td style="text-align:center;color:#E67E22;font-weight:bold;">MALA</td>
        <td>Posibles efectos en salud de grupos sensibles (niños/asmáticos).</td>
      </tr>
      <tr style="background-color:#fcc6c2;">
        <td style="text-align:center;font-weight:bold;">55.5 - 150.4</td>
        <td style="text-align:center;color:#C0392B;font-weight:bold;">MUY MALA</td>
        <td>Efectos adversos en la salud de la población general.</td>
      </tr>
      <tr style="background-color:#f1d9fc;">
        <td style="text-align:center;font-weight:bold;">+ 150.5</td>
        <td style="text-align:center;color:#884EA0;font-weight:bold;">PELIGROSA</td>
        <td>Alerta sanitaria: Probabilidad alta de afectaciones graves.</td>
      </tr>
    </tbody>
    </table>
    """
    st.markdown(tabla_html, unsafe_allow_html=True)

# ==============================================================================
# 2. PESTAÑA DE DASHBOARD
# ==============================================================================
with tab_dashboard:
    try:
        df = pd.read_csv('resultados_test.csv')
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        c1, c2 = st.columns([3, 1])
        
        with c1:
            st.markdown("### Serie de Tiempo Comparativa (Test Set)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Fecha'], y=df['Real'], mode='lines', name='Sensor Real (Target)', line=dict(color='black', width=2)))
            fig.add_trace(go.Scatter(x=df['Fecha'], y=df['ANFIS_EA'], mode='lines', name='ANFIS', line=dict(color='#2ECC71', width=1.5)))
            fig.add_trace(go.Scatter(x=df['Fecha'], y=df['ANN_Simple'], mode='lines', name='ANN', line=dict(color='#E74C3C', width=1.5, dash='dot')))
            fig.update_layout(height=400, template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("### 🏆 Métricas")
            rmse_anfis = np.sqrt(mean_squared_error(df['Real'], df['ANFIS_EA']))
            r2_anfis = r2_score(df['Real'], df['ANFIS_EA'])
            rmse_ann = np.sqrt(mean_squared_error(df['Real'], df['ANN_Simple']))
            r2_ann = r2_score(df['Real'], df['ANN_Simple'])
            
            st.markdown("**ANFIS**")
            st.metric("RMSE", f"{rmse_anfis:.2f}", delta=f"R²: {r2_anfis:.3f}")
            st.markdown("---")
            st.markdown("**ANN**")
            st.metric("RMSE", f"{rmse_ann:.2f}", delta=f"R²: {r2_ann:.3f}", delta_color="inverse")
            
    except FileNotFoundError:
        st.error("⚠️ Falta 'resultados_test.csv'. Ejecuta 'procesar_modelos.py'.")

# ==============================================================================
# 3. PESTAÑA DE ENTRENAMIENTO
# ==============================================================================
with tab_train:
    st.markdown("### Evidencia Visual del Entrenamiento")
    st.info("Estas gráficas se generaron automáticamente durante la fase de aprendizaje de cada uno de los modelos.")
    
    col1, col2 = st.columns(2)
    ruta_anfis = "Resultados_Modelo_ANFIS_Optimizado"
    ruta_ann = "Resultados_ANN-MLP"

    with col1:
        st.subheader("🟢 Modelo ANFIS (Propuesto)")
        if os.path.exists(ruta_anfis):
            try:
                st.image(os.path.join(ruta_anfis, "1_Curva_Aprendizaje.png"), caption="Curva de Convergencia (ANFIS)", use_container_width=True)
                st.image(os.path.join(ruta_anfis, "3_Dispersion_Regresion.png"), caption="Regresión Lineal (ANFIS)", use_container_width=True)
                st.image(os.path.join(ruta_anfis, "4_Histograma_Errores.png"), caption="Distribución de Errores (ANFIS)", use_container_width=True)
                st.image(os.path.join(ruta_anfis, "5_Zoom_Detalle.png"), caption="Zoom a Detalles (ANFIS)", use_container_width=True)
            except:
                st.warning("Faltan algunas imágenes en la carpeta de ANFIS.")
        else:
            st.error(f"No existe la carpeta {ruta_anfis}. Ejecuta ANFIS.py primero.")

    with col2:
        st.subheader("🔴 Modelo ANN (Benchmark)")
        if os.path.exists(ruta_ann):
            try:
                st.image(os.path.join(ruta_ann, "1_Curva_ANN.png"), caption="Curva de Aprendizaje (ANN)", use_container_width=True)
                st.image(os.path.join(ruta_ann, "3_Dispersion_ANN.png"), caption="Regresión Lineal (ANN)", use_container_width=True)
                st.image(os.path.join(ruta_ann, "4_Errores_ANN.png"), caption="Distribución de Errores (ANN)", use_container_width=True)
                st.image(os.path.join(ruta_ann, "5_Zoom_ANN.png"), caption="Zoom a Detalles (ANN)", use_container_width=True)
            except:
                st.warning("Faltan algunas imágenes en la carpeta de ANN.")
        else:
            st.error(f"No existe la carpeta {ruta_ann}. Ejecuta ModeloSimple.py primero.")

# ==============================================================================
# 4. PESTAÑA DE SIMULADOR
# ==============================================================================
with tab_live:
    st.markdown("### Simulador de Predicción (+1 hora)")
    st.markdown("Introduce los valores actuales y el historial reciente para predecir la contaminación futura.")
    
    try:
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
    except:
        st.error("⚠️ Faltan escaladores (.pkl). Ejecuta 'procesar_modelos.py'.")
        st.stop()

    # Inputs organizados
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**1. Atmósfera**")
            pm10 = st.number_input("PM10 Actual", value=45.0)
            ozono = st.number_input("Ozono (ppb)", value=30.0)
            co = st.number_input("CO (ppb)", value=350.0)
        with c2:
            st.markdown("**2. Ambiente**")
            temp = st.number_input("Temperatura (°C)", value=22.5)
            humedad = st.number_input("Humedad (%)", value=40.0)
            pm10_ant = st.number_input("PM10 (Hace 1h)", value=42.0)
        with c3:
            st.markdown("**3. Historial PM2.5**")
            pm25_act = st.number_input("PM2.5 Actual", value=25.0)
            pm25_ant1 = st.number_input("PM2.5 (Hace 1h)", value=23.0)
            pm25_ant2 = st.number_input("PM2.5 (Hace 2h)", value=20.0)

    if st.button("Calcular Pronóstico Futuro", type="primary"):
        prom_3h = (pm25_act + pm25_ant1 + pm25_ant2) / 3
        cambio_pm25 = pm25_act - pm25_ant1
        cambio_pm10 = pm10 - pm10_ant
        
        input_data = np.array([[pm10, ozono, co, temp, humedad, pm25_act, prom_3h, cambio_pm25, cambio_pm10]])
        input_norm = scaler_X.transform(input_data)
        input_tensor = torch.tensor(input_norm, dtype=torch.float32)
        
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        
        # --- RESULTADO ANFIS ---
        try:
            modelo_anfis = ANFIS(n_entr=9, n_curvas=3, n_reglas=35)
            modelo_anfis.load_state_dict(torch.load('modelo_anfis.pth'))
            modelo_anfis.eval()
            with torch.no_grad():
                pred = scaler_y.inverse_transform(modelo_anfis(input_tensor).numpy())[0][0]
                val_anfis = max(0, pred)
            
            calidad, color  = obtener_calidad(val_anfis)
            
            with res_col1:
                st.markdown(f"<div style='background-color:{color}20; padding:15px; border-radius:10px; border-left:5px solid {color};'>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color:{color}; margin:0;'>🟢 Predicción ANFIS</h3>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='margin:0;'>{val_anfis:.2f} <span style='font-size:16px'>ug/m3</span></h1>", unsafe_allow_html=True)
                st.markdown(f"**Calidad:** {calidad}")
                st.markdown("</div>", unsafe_allow_html=True)
        except:
            res_col1.error("Error ANFIS")

        # --- RESULTADO ANN ---
        try:
            modelo_ann = ANN_Simple(n_entr=9)
            modelo_ann.load_state_dict(torch.load('modelo_ann.pth'))
            modelo_ann.eval()
            with torch.no_grad():
                pred = scaler_y.inverse_transform(modelo_ann(input_tensor).numpy())[0][0]
                val_ann = max(0, pred)
            
            calidad, color  = obtener_calidad(val_ann)

            with res_col2:
                st.markdown(f"<div style='background-color:{color}20; padding:15px; border-radius:10px; border-left:5px solid {color};'>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color:{color}; margin:0;'>🔴 Predicción ANN</h3>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='margin:0;'>{val_ann:.2f} <span style='font-size:16px'>ug/m3</span></h1>", unsafe_allow_html=True)
                st.markdown(f"**Calidad:** {calidad}")
                st.markdown("</div>", unsafe_allow_html=True)
        except:
            res_col2.error("Error ANN")