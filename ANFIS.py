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

# ==============================================================================
# 1. ARQUITECTURA ANFIS (CLASE EXPORTABLE)
# ==============================================================================
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

# ==============================================================================
# 2. BLOQUE DE EJECUCIÓN 
# ==============================================================================
if __name__ == "__main__":
    
    # ---- CONFIGURACIÓN INICIAL -----
    CARPETA_SALIDA = "Resultados_Modelo_ANFIS_Optimizado"
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)

    # ---- INGENIERÍA DE DATOS ----
    print("--- [1/6] Procesando Datos y Calculando Física ---")

    archivo = "AirQualityIBEROCDMX.csv"
    if not os.path.exists(archivo):
        print("ERROR CRÍTICO: Archivo CSV no encontrado.")
        exit()

    datos = pd.read_csv(archivo)
    col_obj = 'PM2.5 [ug/m3]'

    # Limpieza
    datos = datos.dropna()
    datos = datos[datos[col_obj] < 600]
    datos = datos[datos['PM10[ug/m3]'] < 1000]

    # Variables
    datos['pm25_act'] = datos[col_obj]
    datos['pm25_ant'] = datos[col_obj].shift(1)
    datos['pm25_prom'] = datos[col_obj].rolling(3).mean() # Suavizado

    # Velocidad de cambio 
    datos['cambio_pm25'] = datos['pm25_act'] - datos['pm25_ant']
    datos['cambio_pm10'] = datos['PM10[ug/m3]'] - datos['PM10[ug/m3]'].shift(1)

    # Objetivo (Futuro t+1)
    datos['objetivo'] = datos[col_obj].shift(-1)

    cols_entr = [
        'PM10[ug/m3]', 'Ozone [ppb]', 'Carbon_Monoxide [ppb]', 
        'Temperature [°C]', 'Relative_Humidity [%]',
        'pm25_act', 'pm25_prom', 'cambio_pm25', 'cambio_pm10'
    ]

    datos = datos.dropna()

    X = datos[cols_entr]
    y = datos['objetivo']

    # Normalización
    esc_X = MinMaxScaler()
    esc_y = MinMaxScaler()

    X_norm = esc_X.fit_transform(X)
    y_norm = esc_y.fit_transform(y.values.reshape(-1, 1))

    # División
    X_ent, X_pru, y_ent, y_pru = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)

    # Tensores
    X_ent_t = torch.tensor(X_ent, dtype=torch.float32)
    y_ent_t = torch.tensor(y_ent, dtype=torch.float32)
    X_pru_t = torch.tensor(X_pru, dtype=torch.float32)
    y_pru_t = torch.tensor(y_pru, dtype=torch.float32)

    print("Datos listos.")

    # ---- ALGORITMO GENÉTICO ----
    print("\n--- [2/6] Ejecutando Algoritmo Genético ---")

    def evaluar(n_reglas, tasa):
        modelo = ANFIS(n_entr=9, n_curvas=3, n_reglas=int(n_reglas))
        modelo.inic_params(X_ent_t)
        criterio = nn.MSELoss()
        optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa)
        modelo.train()
        for _ in range(30): 
            optimizador.zero_grad()
            pred = modelo(X_ent_t)
            error = criterio(pred, y_ent_t)
            error.backward()
            optimizador.step()
        modelo.eval()
        with torch.no_grad():
            loss_pru = criterio(modelo(X_pru_t), y_pru_t)
        return loss_pru.item()

    poblacion = [{'reglas': random.randint(20, 50), 'tasa': random.uniform(0.001, 0.015)} for _ in range(6)]
    mejor_gen = None
    mejor_puntaje = float('inf')

    for gen in range(4): 
        puntajes = []
        for indiv in poblacion:
            pt = evaluar(indiv['reglas'], indiv['tasa'])
            puntajes.append((pt, indiv))
            if pt < mejor_puntaje:
                mejor_puntaje = pt
                mejor_gen = indiv
        
        print(f"Gen {gen+1}: Error Mínimo = {mejor_puntaje:.5f}")
        
        puntajes.sort(key=lambda x: x[0])
        padres = [x[1] for x in puntajes[:2]]
        nueva_pob = padres[:]
        while len(nueva_pob) < 6:
            padre = random.choice(padres)
            hijo = {'reglas': int(padre['reglas'] + random.randint(-3, 3)), 'tasa': padre['tasa'] * random.uniform(0.9, 1.1)}
            hijo['reglas'] = max(15, min(60, hijo['reglas']))
            nueva_pob.append(hijo)
        poblacion = nueva_pob

    print(f"GANADOR: {mejor_gen['reglas']} Reglas | Tasa: {mejor_gen['tasa']:.5f}")

    # ---- ENTRENAMIENTO FINAL ----
    print("\n--- [3/6] Iniciando Entrenamiento Final ---")

    reglas_fin = mejor_gen['reglas']
    tasa_fin = mejor_gen['tasa']
    EPOCAS = 3000

    modelo = ANFIS(n_entr=9, n_curvas=3, n_reglas=reglas_fin)
    modelo.inic_params(X_ent_t)

    optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_fin)
    regulador = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode='min', factor=0.5, patience=100)
    criterio = nn.MSELoss()

    hist_error = [] 

    for e in range(EPOCAS):
        modelo.train()
        optimizador.zero_grad()
        predic = modelo(X_ent_t)
        error = criterio(predic, y_ent_t)
        error.backward()
        optimizador.step()
        regulador.step(error)
        hist_error.append(error.item())
        
        if (e+1) % 500 == 0:
            print(f"Época {e+1}: Error {error.item():.5f}")

    # ---- GENERACIÓN DE GRÁFICAS ----
    print("\n--- [4/6] Generando Gráficas en Carpeta ---")

    modelo.eval()
    with torch.no_grad():
        pred_norm = modelo(X_pru_t).numpy()

    pred_real = esc_y.inverse_transform(pred_norm)
    y_real = esc_y.inverse_transform(y_pru)
    pred_real = np.maximum(pred_real, 0)
    residous = y_real - pred_real

    # Métricas Numéricas
    rmse = np.sqrt(mean_squared_error(y_real, pred_real))
    r2 = r2_score(y_real, pred_real)
    mape = np.mean(np.abs((y_real - pred_real) / (y_real + 0.1))) * 100

    print(f"\n>>> MÉTRICAS FINALES <<<")
    print(f"RMSE: {rmse:.2f} | R2: {r2:.4f} | MAPE: {mape:.2f}%")

    # Gráfica 1: Curva de Aprendizaje
    plt.figure(figsize=(10, 6))
    plt.plot(hist_error, color='blue', linewidth=2)
    plt.title('1. Curva de Convergencia (MSE Loss)')
    plt.xlabel('Épocas')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CARPETA_SALIDA, "1_Curva_Aprendizaje.png"), dpi=300)
    plt.close()

    # Gráfica 2: Comparación Lineal
    plt.figure(figsize=(12, 6))
    plt.plot(y_real, label='Realidad', color='black', alpha=0.7, linewidth=1)
    plt.plot(pred_real, label='Predicción IA', color='orange', alpha=0.7, linewidth=1)
    plt.title(f'2. Serie de Tiempo Completa (R2={r2:.2f})')
    plt.legend()
    plt.savefig(os.path.join(CARPETA_SALIDA, "2_Serie_Completa.png"), dpi=300)
    plt.close()

    # Gráfica 3: Dispersión
    plt.figure(figsize=(8, 8))
    plt.scatter(y_real, pred_real, alpha=0.4, s=10, c='purple')
    plt.plot([0, y_real.max()], [0, y_real.max()], 'k--', linewidth=2)
    plt.title('3. Análisis de Regresión (Scatter)')
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Predicho')
    plt.savefig(os.path.join(CARPETA_SALIDA, "3_Dispersion_Regresion.png"), dpi=300)
    plt.close()

    # Gráfica 4: Histograma
    plt.figure(figsize=(10, 6))
    plt.hist(residous, bins=50, color='green', alpha=0.7, edgecolor='black')
    plt.title('4. Distribución de Errores (Residuos)')
    plt.xlabel('Magnitud del Error (ug/m3)')
    plt.ylabel('Frecuencia')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.savefig(os.path.join(CARPETA_SALIDA, "4_Histograma_Errores.png"), dpi=300)
    plt.close()

    # Gráfica 5: Zoom
    plt.figure(figsize=(12, 6))
    muestras = 100 
    plt.plot(y_real[:muestras], label='Real', color='black', marker='o', markersize=3)
    plt.plot(pred_real[:muestras], label='Predicción', color='red', linewidth=2)
    plt.title('5. Zoom de Detalle (Primeras 100 horas)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CARPETA_SALIDA, "5_Zoom_Detalle.png"), dpi=300)
    plt.close()

    # ---- PRONÓSTICO MANUAL (INPUT DE USUARIO) ----
    print("\n" + "="*50)
    print("   SISTEMA DE PRONÓSTICO (V4)")
    print("="*50)

    try:
        print("\n--- A. DATOS CLIMÁTICOS ACTUALES ---")
        pm10 = float(input("PM10: "))
        ozono = float(input("Ozono: "))
        co = float(input("CO: "))
        temp = float(input("Temp: "))
        hum = float(input("Humedad: "))
        
        print("\n--- B. HISTORIA RECIENTE ---")
        pm10_ant = float(input("PM10 (Hace 1h): "))
        pm25_act = float(input("PM2.5 (Actual): "))
        pm25_ant1 = float(input("PM2.5 (Hace 1h): "))
        pm25_ant2 = float(input("PM2.5 (Hace 2h): "))
        
        prom_3h = (pm25_act + pm25_ant1 + pm25_ant2) / 3
        cambio_pm25 = pm25_act - pm25_ant1
        cambio_pm10 = pm10 - pm10_ant
        
        x_usr = np.array([[pm10, ozono, co, temp, hum, pm25_act, prom_3h, cambio_pm25, cambio_pm10]])
        x_usr_norm = esc_X.transform(x_usr)
        x_tensor = torch.tensor(x_usr_norm, dtype=torch.float32)
        
        modelo.eval()
        with torch.no_grad():
            res_final = esc_y.inverse_transform(modelo(x_tensor).numpy())[0][0]
        
        print("\n" + "*"*35)
        print(f" PRONÓSTICO (+1 HORA): {max(0, res_final):.2f} ug/m3")
        print("*" * 35)

    except Exception as e:
        print(f"Error: {e}")