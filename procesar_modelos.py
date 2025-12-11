import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# CONFIGURACIÓN
FILE_PATH = 'AirQualityIBEROCDMX.csv'
TARGET_COL = 'PM2.5 [ug/m3]'
EPOCAS_ANFIS = 500  # <--- Sube esto a 3000 para el resultado final
EPOCAS_ANN = 500    # <--- Sube esto a 2500 para el resultado final

# ==========================================
# 1. CLASE ANFIS (Tu código optimizado)
# ==========================================
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

# ==========================================
# 2. CLASE ANN SIMPLE (Tu Benchmark)
# ==========================================
class ANN_Simple(nn.Module):
    def __init__(self, n_entr):
        super(ANN_Simple, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(n_entr, 16),  
            nn.ReLU(),              
            nn.Linear(16, 8),      
            nn.ReLU(),
            nn.Linear(8, 1)        
        )
    
    def forward(self, x):
        return self.red(x)

# ==========================================
# 3. PIPELINE DE EJECUCIÓN
# ==========================================
def ejecutar_pipeline():
    print(f"--- [1/4] Procesando: {FILE_PATH} ---")
    
    if not os.path.exists(FILE_PATH):
        print("❌ ERROR: No se encuentra el archivo CSV.")
        return

    datos = pd.read_csv(FILE_PATH)
    
    # Limpieza
    datos = datos.dropna()
    datos = datos[datos[TARGET_COL] < 600]
    datos = datos[datos['PM10[ug/m3]'] < 1000]

    # Ingeniería de Variables
    datos['pm25_act'] = datos[TARGET_COL]
    datos['pm25_ant'] = datos[TARGET_COL].shift(1)
    datos['pm25_prom'] = datos[TARGET_COL].rolling(3).mean()
    datos['cambio_pm25'] = datos['pm25_act'] - datos['pm25_ant']
    datos['cambio_pm10'] = datos['PM10[ug/m3]'] - datos['PM10[ug/m3]'].shift(1)
    datos['objetivo'] = datos[TARGET_COL].shift(-1)

    cols_entr = [
        'PM10[ug/m3]', 'Ozone [ppb]', 'Carbon_Monoxide [ppb]', 
        'Temperature [°C]', 'Relative_Humidity [%]',
        'pm25_act', 'pm25_prom', 'cambio_pm25', 'cambio_pm10'
    ]

    datos = datos.dropna()
    
    # Guardar fechas para la gráfica final (importante para series de tiempo)
    fechas = pd.to_datetime(datos['Timestamp'], dayfirst=True)

    X = datos[cols_entr].values
    y = datos['objetivo'].values

    # Normalización
    esc_X = MinMaxScaler()
    esc_y = MinMaxScaler()
    X_norm = esc_X.fit_transform(X)
    y_norm = esc_y.fit_transform(y.reshape(-1, 1))

    # Split (Indices=True para recuperar fechas)
    X_ent, X_pru, y_ent, y_pru, idx_ent, idx_pru = train_test_split(
        X_norm, y_norm, np.arange(len(datos)), test_size=0.2, random_state=42
    )

    # Tensores
    X_ent_t = torch.tensor(X_ent, dtype=torch.float32)
    y_ent_t = torch.tensor(y_ent, dtype=torch.float32)
    X_pru_t = torch.tensor(X_pru, dtype=torch.float32)

    # ---------------------------------------------------------
    print("--- [2/4] Entrenando ANFIS (Modelo Propuesto) ---")
    modelo_anfis = ANFIS(n_entr=9, n_curvas=3, n_reglas=35) # Reglas promedio del GA
    modelo_anfis.inic_params(X_ent_t)
    opt_anfis = torch.optim.Adam(modelo_anfis.parameters(), lr=0.005)
    crit = nn.MSELoss()

    modelo_anfis.train()
    for e in range(EPOCAS_ANFIS):
        opt_anfis.zero_grad()
        loss = crit(modelo_anfis(X_ent_t), y_ent_t)
        loss.backward()
        opt_anfis.step()
        if e % 100 == 0: print(f"   ANFIS Epoca {e}: Error {loss.item():.5f}")

    # Predicción ANFIS
    modelo_anfis.eval()
    with torch.no_grad():
        pred_anfis = modelo_anfis(X_pru_t).numpy()
    pred_anfis_real = np.maximum(esc_y.inverse_transform(pred_anfis), 0).flatten()

    # ---------------------------------------------------------
    print("--- [3/4] Entrenando ANN Simple (Benchmark) ---")
    modelo_ann = ANN_Simple(n_entr=9)
    opt_ann = torch.optim.SGD(modelo_ann.parameters(), lr=0.01, momentum=0.9)
    
    modelo_ann.train()
    for e in range(EPOCAS_ANN):
        opt_ann.zero_grad()
        loss = crit(modelo_ann(X_ent_t), y_ent_t)
        loss.backward()
        opt_ann.step()
        if e % 100 == 0: print(f"   ANN Epoca {e}: Error {loss.item():.5f}")

    # Predicción ANN
    modelo_ann.eval()
    with torch.no_grad():
        pred_ann = modelo_ann(X_pru_t).numpy()
    pred_ann_real = np.maximum(esc_y.inverse_transform(pred_ann), 0).flatten()
    
    # ---------------------------------------------------------
    print("--- [4/4] Generando archivo para interfaz ---")
    y_real_val = esc_y.inverse_transform(y_pru).flatten()
    fechas_test = fechas.iloc[idx_pru].values
    
    df_res = pd.DataFrame({
        'Fecha': fechas_test,
        'Real': y_real_val,
        'ANFIS_EA (Propuesto)': pred_anfis_real,
        'ANN_Simple (Benchmark)': pred_ann_real
    })
    
    df_res = df_res.sort_values('Fecha')
    df_res.to_csv('resultados_test.csv', index=False)
    print("✅ ¡Listo! Archivo 'resultados_test.csv' generado correctamente.")

if __name__ == "__main__":
    ejecutar_pipeline()