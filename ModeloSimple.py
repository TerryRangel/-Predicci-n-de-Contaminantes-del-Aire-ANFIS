import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import os

# ==============================================================================
# 1. ARQUITECTURA ANN (CLASE EXPORTABLE)
# ==============================================================================
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

# ==============================================================================
# 2. BLOQUE DE EJECUCIÓN 
# ==============================================================================
if __name__ == "__main__":
    
    # CONFIGURACIÓN INICIAL
    CARPETA_SALIDA = "Resultados_ANN-MLP"
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)

    # PROCESAMIENTO DE DATOS 
    print("--- [1/4] Procesando Datos ---")

    archivo = "AirQualityIBEROCDMX.csv"
    if not os.path.exists(archivo):
        print("ERROR: Archivo CSV no encontrado.")
        exit()

    datos = pd.read_csv(archivo)
    col_obj = 'PM2.5 [ug/m3]'

    datos = datos.dropna()
    datos = datos[datos[col_obj] < 600]
    datos = datos[datos['PM10[ug/m3]'] < 1000]

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

    X_ent_t = torch.tensor(X_ent, dtype=torch.float32)
    y_ent_t = torch.tensor(y_ent, dtype=torch.float32)
    X_pru_t = torch.tensor(X_pru, dtype=torch.float32)
    y_pru_t = torch.tensor(y_pru, dtype=torch.float32)

    # ENTRENAMIENTO 
    print("\n--- [2/4] Entrenando Modelo Simple ---")

    modelo = ANN_Simple(n_entr=9)

    optimizador = torch.optim.SGD(modelo.parameters(), lr=0.01, momentum=0.9) 
    criterio = nn.MSELoss()
    hist_err = []

    EPOCAS = 2500 

    for e in range(EPOCAS):
        modelo.train()
        optimizador.zero_grad()
        loss = criterio(modelo(X_ent_t), y_ent_t)
        loss.backward()
        optimizador.step()
        hist_err.append(loss.item())
        
        if (e+1) % 500 == 0:
            print(f"Época {e+1}: Error {loss.item():.5f}")

    # RESULTADOS Y GUARDADO DE GRÁFICAS
    print("\n--- [3/4] Generando Gráficas de Ann ---")

    modelo.eval()
    with torch.no_grad():
        pred_norm = modelo(X_pru_t).numpy()

    pred_real = np.maximum(esc_y.inverse_transform(pred_norm), 0)
    y_real = esc_y.inverse_transform(y_pru)
    residous = y_real - pred_real

    rmse = np.sqrt(mean_squared_error(y_real, pred_real))
    r2 = r2_score(y_real, pred_real)
    mape = np.mean(np.abs((y_real - pred_real) / (y_real + 0.1))) * 100

    print(f"\n>>> RESULTADOS ANN(SIMPLE) <<<")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}  ")
    print(f"MAPE: {mape:.2f}%")

    # --- GENERACIÓN DE GRÁFICAS ---

    # 1. Curva de Aprendizaje
    plt.figure(figsize=(10, 6))
    plt.plot(hist_err, color='gray', linewidth=2, linestyle='--')
    plt.title('1. Aprendizaje ANN Simple')
    plt.xlabel('Épocas')
    plt.ylabel('Error MSE')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CARPETA_SALIDA, "1_Curva_ANN.png"), dpi=300)
    plt.close()

    # 2. Serie de Tiempo
    plt.figure(figsize=(12, 6))
    plt.plot(y_real, label='Realidad', color='black', alpha=0.7, linewidth=1)
    plt.plot(pred_real, label='ANN Simple', color='red', alpha=0.6, linewidth=1)
    plt.title(f'2. Predicción ANN (R2={r2:.2f})')
    plt.legend()
    plt.savefig(os.path.join(CARPETA_SALIDA, "2_Serie_ANN.png"), dpi=300)
    plt.close()

    # 3. Dispersión
    plt.figure(figsize=(8, 8))
    plt.scatter(y_real, pred_real, alpha=0.4, s=10, c='red')
    plt.plot([0, y_real.max()], [0, y_real.max()], 'k--', linewidth=2)
    plt.title('3. Regresión ANN Simple')
    plt.xlabel('Real')
    plt.ylabel('Predicho')
    plt.savefig(os.path.join(CARPETA_SALIDA, "3_Dispersion_ANN.png"), dpi=300)
    plt.close()

    # 4. Histograma Errores
    plt.figure(figsize=(10, 6))
    plt.hist(residous, bins=50, color='gray', alpha=0.7, edgecolor='black')
    plt.title('4. Errores ANN Simple')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.savefig(os.path.join(CARPETA_SALIDA, "4_Errores_ANN.png"), dpi=300)
    plt.close()

    # 5. Zoom Detalle
    plt.figure(figsize=(12, 6))
    muestras = 100
    plt.plot(y_real[:muestras], label='Real', color='black', marker='o', markersize=3)
    plt.plot(pred_real[:muestras], label='Benchmark', color='red', linewidth=2, linestyle='--')
    plt.title('5. Detalle ANN Simple')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CARPETA_SALIDA, "5_Zoom_ANN.png"), dpi=300)
    plt.close()