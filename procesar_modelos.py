import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib # Para guardar los escaladores
import os

# IMPORTAMOS TUS CLASES
from ANFIS import ANFIS  
from ModeloSimple import ANN_Simple 

FILE_PATH = 'AirQualityIBEROCDMX.csv'
TARGET_COL = 'PM2.5 [ug/m3]'

def ejecutar_pipeline():
    print(f"--- [1/5] Procesando Datos ---")
    if not os.path.exists(FILE_PATH):
        print("❌ ERROR: No se encuentra el CSV.")
        return

    datos = pd.read_csv(FILE_PATH)
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
    fechas = pd.to_datetime(datos['Timestamp'], dayfirst=True)

    X = datos[cols_entr].values
    y = datos['objetivo'].values

    # NORMALIZACIÓN Y GUARDADO DE ESCALADORES
    print("--- [2/5] Guardando Escaladores para la Interfaz ---")
    esc_X = MinMaxScaler()
    esc_y = MinMaxScaler()
    X_norm = esc_X.fit_transform(X)
    y_norm = esc_y.fit_transform(y.reshape(-1, 1))
    
    # Guardamos esto para usarlo en la app.py
    joblib.dump(esc_X, 'scaler_X.pkl')
    joblib.dump(esc_y, 'scaler_y.pkl')

    X_ent, X_pru, y_ent, y_pru, idx_ent, idx_pru = train_test_split(
        X_norm, y_norm, np.arange(len(datos)), test_size=0.2, random_state=42
    )

    X_ent_t = torch.tensor(X_ent, dtype=torch.float32)
    y_ent_t = torch.tensor(y_ent, dtype=torch.float32)
    X_pru_t = torch.tensor(X_pru, dtype=torch.float32)

    # --- ANFIS ---
    print("--- [3/5] Entrenando y Guardando ANFIS ---")
    modelo_anfis = ANFIS(n_entr=9, n_curvas=3, n_reglas=35)
    modelo_anfis.inic_params(X_ent_t)
    opt_anfis = torch.optim.Adam(modelo_anfis.parameters(), lr=0.005)
    crit = nn.MSELoss()

    modelo_anfis.train()
    for e in range(500): # Sube las épocas si quieres
        opt_anfis.zero_grad()
        loss = crit(modelo_anfis(X_ent_t), y_ent_t)
        loss.backward()
        opt_anfis.step()
    
    # Guardamos el modelo entrenado
    torch.save(modelo_anfis.state_dict(), 'modelo_anfis.pth')

    modelo_anfis.eval()
    with torch.no_grad():
        pred_anfis = np.maximum(esc_y.inverse_transform(modelo_anfis(X_pru_t).numpy()), 0).flatten()

    # --- ANN SIMPLE ---
    print("--- [4/5] Entrenando y Guardando ANN Simple ---")
    modelo_ann = ANN_Simple(n_entr=9)
    opt_ann = torch.optim.SGD(modelo_ann.parameters(), lr=0.01, momentum=0.9)
    
    modelo_ann.train()
    for e in range(500): # Sube las épocas si quieres
        opt_ann.zero_grad()
        loss = crit(modelo_ann(X_ent_t), y_ent_t)
        loss.backward()
        opt_ann.step()

    # Guardamos el modelo entrenado
    torch.save(modelo_ann.state_dict(), 'modelo_ann.pth')

    modelo_ann.eval()
    with torch.no_grad():
        pred_ann = np.maximum(esc_y.inverse_transform(modelo_ann(X_pru_t).numpy()), 0).flatten()
    
    # --- RESULTADOS ---
    print("--- [5/5] Generando CSV ---")
    y_real_val = esc_y.inverse_transform(y_pru).flatten()
    fechas_test = fechas.iloc[idx_pru].values
    
    df_res = pd.DataFrame({
        'Fecha': fechas_test,
        'Real': y_real_val,
        'ANFIS_EA': pred_anfis,
        'ANN_Simple': pred_ann
    })
    
    df_res = df_res.sort_values('Fecha')
    df_res.to_csv('resultados_test.csv', index=False)
    print("✅ ¡Listo! Modelos (.pth), Escaladores (.pkl) y Resultados (.csv) generados.")

if __name__ == "__main__":
    ejecutar_pipeline()