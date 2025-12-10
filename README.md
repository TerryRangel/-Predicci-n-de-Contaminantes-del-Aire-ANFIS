# -Predicci-n-de-Contaminantes-del-Aire-ANFIS
Este proyecto busca predecir la concentración futura de un  contaminante objetivo (PM2.5 o PM10) utilizando la  información histórica de la propia serie y variables  meteorológicas y de otros contaminantes (variables  exógenas).


# REQUERIMIENTOS E INTALACIONES

-Python 3.10, 3.11
-librerias

pandas: Para leer el Excel (CSV) y limpiar datos.

numpy: Para cálculos matemáticos rápidos.

torch(PyTorch): El motor de Inteligencia Artificial para crear la red neuronal.

matplotlib: Para generar las gráficas de resultados.

scikit-learn: Para normalizar datos y calcular métricas de error (R2, RMSE).

COMANDO: pip install pandas numpy torch matplotlib scikit-learn

# Descripción del Proyecto
Este proyecto implementa un modelo ANFIS (Adaptive Neuro-Fuzzy Inference System) desarrollado en PyTorch para predecir la concentración de partículas PM2.5 calculando en variables climáticas y otros contaminantes.

Para garantizar el máximo rendimiento, el sistema integra un Algoritmo Genético (GA) que busca automáticamente la mejor arquitectura (número de reglas y tasa de aprendizaje) antes de iniciar el entrenamiento final.

 Características principales
*Arquitectura Híbrida: Combina la interpretabilidad de la Lógica Difusa con la capacidad de aprendizaje de las Redes Neuronales.

*Optimización Evolutiva: Ajuste automático de hiperparámetros mediante Algoritmos Genéticos.

*Limpieza Inteligente de Datos: Filtros estadísticos para eliminar errores de medición en sensores (outliers).

*Clasificación de Riesgo: Traduzca la predicción numérica a niveles de alerta (Bajo, Medio , Alto ).

*Interfaz Interactiva: Permite al usuario ingresar datos manualmente para realizar predicciones en tiempo real.

*Arquitectura del Modelo
El sistema procesa 5 variables de entrada para generar una predicción precisa:

1. Entradas (Características)
PM10 [ug/m3]

Ozono [ppb]

Monóxido de Carbono (CO) [ppb]

Temperatura [°C]

Humedad Relativa [%]

2. Estructura ANFIS (PyTorch)
Capa de Fuzzificación: Convierte los valores numéricos en grados de pertenencia usando funciones Gaussianas aprendibles.

Capa de Reglas: Evalúa las combinaciones lógicas de las entradas (Antecedentes).

Capa de Defuzzificación: Calcula la salida numérica final (Consecuente) mediante una combinación lineal ponderada.

3. Fase Genética
El Algoritmo Genético evoluciona una población de configuraciones durante varias generaciones para minimizar el error (MSE), seleccionando la mejor combinación de:

Número de Reglas(Complejidad del modelo).

Learning Rate(Velocidad de aprendizaje).