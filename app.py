import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np

# Configuración
st.set_page_config(page_title="Monitor Calidad Aire - CDMX", layout="wide")

st.markdown("""
    <style>
    .big-font {font-size:30px !important; font-weight: bold; color: #2E86C1;}
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">🌫️ Sistema Inteligente de Predicción PM2.5</p>', unsafe_allow_html=True)
st.markdown("Comparativa Técnica: **Neuro-Difuso Evolutivo (ANFIS-EA)** vs **Red Neuronal Clásica (ANN)**")

# CARGA DE DATOS
try:
    df = pd.read_csv('resultados_test.csv')
    df['Fecha'] = pd.to_datetime(df['Fecha'])
except FileNotFoundError:
    st.error("⚠️ Falta el archivo 'resultados_test.csv'. Ejecuta primero 'procesar_modelos.py'.")
    st.stop()

# INTERFAZ
col_control, col_grafica = st.columns([1, 3])

with col_control:
    st.header("⚙️ Configuración")
    cols_pred = [c for c in df.columns if c not in ['Fecha', 'Real']]
    seleccion = st.multiselect("Seleccionar Modelos:", cols_pred, default=cols_pred)
    
    st.markdown("---")
    st.subheader("📊 Métricas (Test Set)")
    
    if seleccion:
        for modelo in seleccion:
            rmse = np.sqrt(mean_squared_error(df['Real'], df[modelo]))
            r2 = r2_score(df['Real'], df[modelo])
            
            st.markdown(f"**{modelo}**")
            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{rmse:.2f}")
            col2.metric("R²", f"{r2:.4f}")
            st.markdown("---")

# GRÁFICA PRINCIPAL
with col_grafica:
    st.subheader("📈 Series de Tiempo: Predicción vs Realidad")
    
    fig = go.Figure()
    
    # Real
    fig.add_trace(go.Scatter(
        x=df['Fecha'], y=df['Real'], mode='lines', name='Sensor Real (Target)',
        line=dict(color='black', width=2)
    ))
    
    # Modelos
    colores = {'ANFIS_EA (Propuesto)': '#2ECC71', 'ANN_Simple (Benchmark)': '#E74C3C'}
    
    for modelo in seleccion:
        color = colores.get(modelo, 'blue')
        fig.add_trace(go.Scatter(
            x=df['Fecha'], y=df[modelo], mode='lines', name=modelo,
            line=dict(color=color, width=2, dash='dot')
        ))
        
    fig.update_layout(
        height=500,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Tiempo",
        yaxis_title="PM2.5 [ug/m3]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# SECCIÓN COMPARATIVA (Zoom y Errores)
st.subheader("🔍 Análisis Detallado de Errores")
tab1, tab2 = st.tabs(["📉 Distribución de Errores (Residuos)", "🔭 Zoom (Últimas 48 horas)"])

with tab1:
    fig_hist = go.Figure()
    for modelo in seleccion:
        residuo = df['Real'] - df[modelo]
        fig_hist.add_trace(go.Histogram(
            x=residuo, name=f'Error {modelo}', opacity=0.6
        ))
    fig_hist.update_layout(barmode='overlay', title="Histograma de Residuos (Cero es perfecto)")
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    # Filtramos las ultimas 48 datos para ver detalle
    df_zoom = df.tail(48)
    fig_zoom = go.Figure()
    fig_zoom.add_trace(go.Scatter(x=df_zoom['Fecha'], y=df_zoom['Real'], mode='lines+markers', name='Real', line=dict(color='black')))
    for modelo in seleccion:
        fig_zoom.add_trace(go.Scatter(x=df_zoom['Fecha'], y=df_zoom[modelo], mode='lines', name=modelo))
    fig_zoom.update_layout(title="Comportamiento en las últimas 48 horas registradas")
    st.plotly_chart(fig_zoom, use_container_width=True)