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

# =============================================================================
# SECCIÓN 1: LECTURA Y PREPROCESAMIENTO DE DATOS

print("--- [1/6] Cargando y Procesando Datos ---")

filename = "AirQualityIBEROCDMX.csv"
if not os.path.exists(filename):
    print(f"ERROR: Archivo '{filename}' no encontrado.")
    exit()

df = pd.read_csv(filename)
features = ['PM10[ug/m3]', 'Ozone [ppb]', 'Carbon_Monoxide [ppb]', 'Temperature [°C]', 'Relative_Humidity [%]']
target = 'PM2.5 [ug/m3]'

# --- LIMPIEZA ---
df = df.dropna()
df = df[df[target] < 600]        # Eliminar errores extremos
df = df[df['PM10[ug/m3]'] < 1000]

X = df[features]
y = df[target]

# --- NORMALIZACIÓN ---
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y.values.reshape(-1, 1))

# --- SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)

# Tensores
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

print("Datos listos.")

# =============================================================================
#  ARQUITECTURA ANFIS


class ANFIS_Global(nn.Module):
    def __init__(self, num_inputs, num_mf, num_rules):
        super(ANFIS_Global, self).__init__()
        self.num_inputs = num_inputs
        self.num_mf = num_mf
        
        self.means = nn.Parameter(torch.rand(num_inputs, num_mf))
        self.sigmas = nn.Parameter(torch.ones(num_inputs, num_mf))
        self.rule_connection = nn.Linear(num_inputs * num_mf, num_rules)
        self.consequence = nn.Linear(num_rules, 1)
        
        nn.init.xavier_uniform_(self.consequence.weight)
        nn.init.zeros_(self.consequence.bias)

    def forward(self, x):
        # Fuzzificación
        x_unsqueeze = x.unsqueeze(-1)
        pertenencia = torch.exp(-torch.pow(x_unsqueeze - self.means, 2) / (2 * torch.pow(self.sigmas, 2)))
        
        # Reglas
        batch_size = x.shape[0]
        x_flat = pertenencia.reshape(batch_size, -1)
        rules_output = torch.relu(self.rule_connection(x_flat))
        
        # Defuzzificación
        prediction = self.consequence(rules_output)
        return prediction

    def inicializar_parametros(self, X_data):
        with torch.no_grad():
            for i in range(self.num_inputs):
                columna = X_data[:, i]
                self.means[i] = torch.quantile(columna, torch.linspace(0.1, 0.9, self.num_mf))
                self.sigmas[i] = torch.std(columna) + 0.05

# =============================================================================
# ALGORITMO GENÉTICO

print("\n--- [2/6] Optimizando Hiperparámetros (Genético) ---")

def evaluar_cromosoma(num_rules, lr):
    model = ANFIS_Global(num_inputs=5, num_mf=3, num_rules=int(num_rules))
    model.inicializar_parametros(X_train_tensor)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for _ in range(30): 
        optimizer.zero_grad()
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_tensor)
        test_loss = criterion(test_pred, y_test_tensor)
    return test_loss.item()

# Población pequeña 
poblacion = [{'rules': random.randint(10, 30), 'lr': random.uniform(0.001, 0.01)} for _ in range(5)]
best_genes = None
best_score = float('inf')

for gen in range(3):
    scores = []
    for individuo in poblacion:
        score = evaluar_cromosoma(individuo['rules'], individuo['lr'])
        scores.append((score, individuo))
        if score < best_score:
            best_score = score
            best_genes = individuo
    
    print(f"Gen {gen+1}: Mejor Error = {best_score:.5f}")
    
    scores.sort(key=lambda x: x[0])
    padres = [x[1] for x in scores[:2]]
    nueva_poblacion = padres[:]
    while len(nueva_poblacion) < 5:
        padre = random.choice(padres)
        hijo = {'rules': int(padre['rules'] + random.randint(-2, 2)), 'lr': padre['lr'] * random.uniform(0.9, 1.1)}
        hijo['rules'] = max(5, min(40, hijo['rules']))
        nueva_poblacion.append(hijo)
    poblacion = nueva_poblacion

print(f"GANADOR: {best_genes['rules']} Reglas | LR: {best_genes['lr']:.5f}")

# =============================================================================
#  ENTRENAMIENTO 
print("\n--- [3/6] Entrenando Modelo Final ---")

final_rules = best_genes['rules']
final_lr = best_genes['lr']
EPOCHS = 1500

model = ANFIS_Global(num_inputs=5, num_mf=3, num_rules=final_rules)
model.inicializar_parametros(X_train_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=final_lr)
criterion = nn.MSELoss()

loss_history = [] # Para la gráfica 1

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    
    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}: Error {loss.item():.5f}")

# =============================================================================
# VISUALIZACIÓN Y REPORTES(GRÁFICAS)

print("\n--- [4/6] Generando Gráficas y Métricas ---")

model.eval()
with torch.no_grad():
    pred_norm = model(X_test_tensor).numpy()

pred_real = scaler_y.inverse_transform(pred_norm)
y_real = scaler_y.inverse_transform(y_test)
pred_real = np.maximum(pred_real, 0)

# --- CLASIFICACIÓN ---
def clasificar_riesgo(valor):
    if valor < 12: return "BAJO (Bueno)"
    elif valor < 35.5: return "MEDIO (Moderado)"
    elif valor < 55.5: return "ALTO (Dañino Sensible)"
    elif valor < 150.5: return "MUY ALTO (Dañino)"
    else: return "PELIGROSO"

# --- MÉTRICAS ---
rmse = np.sqrt(mean_squared_error(y_real, pred_real))
r2 = r2_score(y_real, pred_real)
mape = np.mean(np.abs((y_real - pred_real) / (y_real + 0.1))) * 100

print(f"\nRMSE: {rmse:.2f} | R2: {r2:.4f} | MAPE: {mape:.2f}%")

# --- GENERACIÓN DE 3 GRÁFICAS ---
plt.figure(figsize=(18, 5))

# GRÁFICA 1: Curva de Aprendizaje
plt.subplot(1, 3, 1)
plt.plot(loss_history, color='blue', linewidth=2)
plt.title('1. Curva de Aprendizaje (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Error (MSE)')
plt.grid(True, alpha=0.3)

# GRÁFICA 2: Predicción vs Realidad (Scatter)
plt.subplot(1, 3, 2)
scatter = plt.scatter(y_real, pred_real, c=pred_real, cmap='RdYlGn_r', alpha=0.6, s=15)
plt.plot([0, y_real.max()], [0, y_real.max()], 'k--', lw=1.5, label='Ideal')
plt.colorbar(scatter, label='Nivel PM2.5')
plt.title(f'2. Precisión (R2: {r2:.2f})')
plt.xlabel('Valor Real')
plt.ylabel('Predicción IA')
plt.legend()
plt.grid(True, alpha=0.3)

# GRÁFICA 3: Zoom a Muestras (Serie de Tiempo)
plt.subplot(1, 3, 3)
muestras = 50 # Solo las primeras 50 para ver detalle
plt.plot(y_real[:muestras], label='Realidad', color='blue', marker='o', markersize=4, alpha=0.7)
plt.plot(pred_real[:muestras], label='Predicción', color='red', linestyle='--', linewidth=2)
plt.title('3. Detalle (Zoom primeros 50 datos)')
plt.xlabel('Muestra')
plt.ylabel('PM2.5')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show() # Muestra todas las gráficas juntas

# Guardar CSV
df_res = pd.DataFrame({'Real': y_real.flatten(), 'Prediccion': pred_real.flatten()})
df_res['Riesgo'] = df_res['Prediccion'].apply(clasificar_riesgo)
df_res.to_csv('resultados.csv', index=False)
print("Archivo 'resultados.csv' guardado.")

# INGRESO DE DATOS 1 VEZ=============================================================================


print("\n" + "="*50)
print("   SISTEMA DE PREDICCIÓN - CONSULTA ")
print("="*50)

try:
    print("\nIntroduce los valores actuales del clima:")
    pm10 = float(input(" -> PM10 [ug/m3]: "))
    o3 = float(input(" -> Ozono [ppb]: "))
    co = float(input(" -> CO [ppb]: "))
    temp = float(input(" -> Temperatura [°C]: "))
    hum = float(input(" -> Humedad [%]: "))
    
    # Proceso
    input_data = np.array([[pm10, o3, co, temp, hum]])
    input_norm = scaler_X.transform(input_data)
    input_tensor = torch.tensor(input_norm, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        pred_norm = model(input_tensor)
    
    pred_final = scaler_y.inverse_transform(pred_norm.numpy())[0][0]
    pred_final = max(0, pred_final)
    nivel = clasificar_riesgo(pred_final)
    
    print("\n" + "-"*30)
    print("      RESULTADO DEL ANÁLISIS")
    print("-" * 30)
    print(f"ESTIMACIÓN PM2.5:   {pred_final:.2f} ug/m3")
    print(f"NIVEL DE RIESGO:    {nivel}")
    print("-" * 30)

except ValueError:
    print("\nERROR: Debes introducir números válidos (ej: 25.5).")
except Exception as e:
    print(f"\nERROR INESPERADO: {e}")
