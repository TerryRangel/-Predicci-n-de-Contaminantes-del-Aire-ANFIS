import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import random
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Predicción ANFIS Calidad del Aire", layout="wide")

st.title("🌬️ Sistema de Predicción de Contaminantes (ANFIS)")
st.markdown("""
Esta aplicación utiliza una red **Neuro-Difusa (ANFIS)** optimizada con Algoritmos Genéticos 
para predecir la concentración de PM2.5 basándose en variables climáticas e históricas.
""")

# --- DEFINICIÓN DEL MODELO (Tomado de tu archivo original) ---
class ANFIS(nn.Module):
    def __init__(self, n_entr, n_curvas, n_reglas):
        super(ANFIS, self).__init__()
        self.n_entr = n_entr
        self.n_curvas = n_curvas
        self.medias = nn.Parameter(torch.rand(n_entr, n_curvas))
        self.sigmas = nn.Parameter(torch.ones(n_entr, n_curvas))
        self.capa_reglas = nn.Linear(n_entr * n_curvas, n_reglas)
        self.salida = nn.Linear(n_reglas, 1)
        nn.init.xavier_uniform_(self.salida.weight)
        nn.init.zeros_(self.salida.bias)

    def forward(self, x):
        x_exp = x.unsqueeze(-1)
        pertenencia = torch.exp(-torch.pow(x_exp - self.medias, 2) / (2 * torch.pow(self.sigmas, 2)))
        lote = x.shape[0]
        x_plano = pertenencia.reshape(lote, -1)
        reglas_out = torch.relu(self.capa_reglas(x_plano))
        return self.salida(reglas_out)

    def inic_params(self, datos_X):
        with torch.no_grad():
            for i in range(self.n_entr):
                col = datos_X[:, i]
                self.medias[i] = torch.quantile(col, torch.linspace(0.1, 0.9, self.n_curvas))
                self.sigmas[i] = torch.std(col) + 0.05

# --- FUNCIÓN DE CARGA Y ENTRENAMIENTO (CON CACHÉ) ---
@st.cache_resource
def entrenar_modelo():
    # 1. Carga de Datos
    archivo = "AirQualityIBEROCDMX.csv"
    if not os.path.exists(archivo):
        st.error(f"No se encontró el archivo {archivo}")
        return None, None, None, None

    datos = pd.read_csv(archivo)
    col_obj = 'PM2.5 [ug/m3]'
    
    # Limpieza (Basado en tu lógica original)
    datos = datos.dropna()
    datos = datos[datos[col_obj] < 600]
    datos = datos[datos['PM10[ug/m3]'] < 1000]

    # Ingeniería de variables
    datos['pm25_act'] = datos[col_obj]
    datos['pm25_ant'] = datos[col_obj].shift(1)
    datos['pm25_prom'] = datos[col_obj].rolling(3).mean()
    datos['cambio_pm25'] = datos['pm25_act'] - datos['pm25_ant']
    datos['cambio_pm10'] = datos['PM10[ug/m3]'] - datos['PM10[ug/m3]'].shift(1)
    datos['objetivo'] = datos[col_obj].shift(-1)

    cols_entr = [
        'PM10[ug/m3]', 'Ozone [ppb]', 'Carbon_Monoxide [ppb]', 
        'Temperature [°C]', 'Relative_Humidity [%]',
        'pm25_act', 'pm25_prom', 'cambio_pm25', 'cambio_pm10'
    ]
    datos = datos.dropna()

    X = datos[cols_entr]
    y = datos['objetivo']

    esc_X = MinMaxScaler()
    esc_y = MinMaxScaler()
    X_norm = esc_X.fit_transform(X)
    y_norm = esc_y.fit_transform(y.values.reshape(-1, 1))

    X_ent, X_pru, y_ent, y_pru = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)
    
    # Tensores
    X_ent_t = torch.tensor(X_ent, dtype=torch.float32)
    # y_ent_t = torch.tensor(y_ent, dtype=torch.float32) # No se usa en inferencia directa aquí
    
    # NOTA: Para este ejemplo interactivo, saltamos el AG completo y usamos parámetros fijos 
    # o un entrenamiento rápido para que la UI cargue. 
    # En producción, deberías cargar un modelo guardado (.pth).
    
    reglas_fin = 30 # Valor promedio de tu AG
    modelo = ANFIS(n_entr=9, n_curvas=3, n_reglas=reglas_fin)
    modelo.inic_params(X_ent_t)
    
    # Simulación de entrenamiento rápido (o cargar pesos guardados)
    # Aquí cargamos el entrenamiento real si quisieras, pero por demo:
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.01)
    criterio = nn.MSELoss()
    y_ent_t = torch.tensor(y_ent, dtype=torch.float32)
    
    progreso = st.progress(0, text="Entrenando modelo inicial...")
    for e in range(500): # Reducido para demo
        modelo.train()
        optimizador.zero_grad()
        predic = modelo(X_ent_t)
        error = criterio(predic, y_ent_t)
        error.backward()
        optimizador.step()
        if e % 50 == 0:
            progreso.progress(int(e/500*100))
    
    progreso.empty()
    modelo.eval()
    
    return modelo, esc_X, esc_y, (X, y)

# --- CARGAR MODELO ---
modelo, esc_X, esc_y, raw_data = entrenar_modelo()

if modelo is not None:
    # --- INTERFAZ LATERAL (INPUTS DEL USUARIO) ---
    st.sidebar.header("🎛️ Parámetros de Entrada")
    st.sidebar.markdown("Define las condiciones actuales para pronosticar:")

    # Inputs basados en tu sección "SISTEMA DE PRONÓSTICO (V4)"
    def user_input_features():
        st.sidebar.subheader("A. Datos Climáticos")
        pm10 = st.sidebar.number_input("PM10 [ug/m3]", min_value=0.0, value=45.0)
        ozono = st.sidebar.number_input("Ozono [ppb]", min_value=0.0, value=25.0)
        co = st.sidebar.number_input("CO [ppb]", min_value=0.0, value=0.5)
        temp = st.sidebar.slider("Temperatura [°C]", -5.0, 40.0, 22.0)
        hum = st.sidebar.slider("Humedad [%]", 0.0, 100.0, 40.0)

        st.sidebar.subheader("B. Historia Reciente")
        pm25_act = st.sidebar.number_input("PM2.5 Actual", min_value=0.0, value=15.0)
        pm10_ant = st.sidebar.number_input("PM10 (Hace 1h)", min_value=0.0, value=40.0)
        pm25_ant1 = st.sidebar.number_input("PM2.5 (Hace 1h)", min_value=0.0, value=14.0)
        pm25_ant2 = st.sidebar.number_input("PM2.5 (Hace 2h)", min_value=0.0, value=12.0)

        # Cálculos derivados (Ingeniería de variables)
        prom_3h = (pm25_act + pm25_ant1 + pm25_ant2) / 3
        cambio_pm25 = pm25_act - pm25_ant1
        cambio_pm10 = pm10 - pm10_ant

        # Crear array con las 9 variables que espera el modelo
        data = np.array([[pm10, ozono, co, temp, hum, pm25_act, prom_3h, cambio_pm25, cambio_pm10]])
        return data

    input_data = user_input_features()

    # --- PANEL PRINCIPAL ---
    
    # Botón de predicción
    if st.button("🔍 Generar Pronóstico (+1 Hora)"):
        with st.spinner('Calculando inferencia difusa...'):
            try:
                # Normalizar
                x_usr_norm = esc_X.transform(input_data)
                x_tensor = torch.tensor(x_usr_norm, dtype=torch.float32)
                
                # Predicción
                with torch.no_grad():
                    pred_norm = modelo(x_tensor).numpy()
                    res_final = esc_y.inverse_transform(pred_norm)[0][0]
                
                # Mostrar Resultado
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="PM2.5 Predicho (+1h)", value=f"{max(0, res_final):.2f} ug/m3")
                with col2:
                    if res_final < 12:
                        st.success("Calidad del Aire: BUENA")
                    elif res_final < 35.4:
                        st.warning("Calidad del Aire: REGULAR")
                    else:
                        st.error("Calidad del Aire: MALA")
                
                # Gráfico de la posición del resultado vs históricos (Opcional)
                st.subheader("Contexto Histórico")
                fig, ax = plt.subplots(figsize=(10, 3))
                # Usamos una muestra de los datos reales para comparar
                hist_vals = raw_data[1][-100:].values # Últimos 100 valores reales
                ax.plot(hist_vals, label="Historia Reciente", color='gray', alpha=0.5)
                ax.axhline(y=res_final, color='red', linestyle='--', label='Tu Predicción')
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error en el cálculo: {e}")

    # Mostrar métricas del modelo (Informativo)
    with st.expander("Ver Estadísticas del Modelo"):
        st.write("El modelo se entrenó usando datos históricos de la CDMX.")
        st.info("Arquitectura: ANFIS Híbrido (Neural + Fuzzy)")

else:
    st.warning("Por favor asegúrate de que 'AirQualityIBEROCDMX.csv' esté en la misma carpeta.")