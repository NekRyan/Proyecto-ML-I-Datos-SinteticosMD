import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split

# === 1. Crear carpeta para visualizaciones si no existe ===
output_folder = "05_visualizaciones_finales"
os.makedirs(output_folder, exist_ok=True)

# === 2. Cargar datos ===
df = pd.read_csv("../01_generacion_datos/clientes_sinteticos.csv")
X = df[["presupuesto_mensual", "visitas_web_mensuales", "interacciones_redes_mensuales",
        "num_empleados", "satisfacción_cliente", "duración_interacción_meses"]]
y = df["monto_ventas_futuras"]

# === 3. Dividir datos ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Cargar modelos entrenados ===
modelo_lineal = joblib.load("../04_modelado_regresion/modelo_lineal.pkl")
modelo_poly = joblib.load("../04_modelado_regresion/modelo_regresion.pkl")

# === 5. Predicciones ===
y_pred_lineal = modelo_lineal.predict(X_test)
y_pred_poly = modelo_poly.predict(X_test)

# === 6. Evaluación de métricas ===
def evaluar_modelo(y_true, y_pred, nombre):
    print(f"--- {nombre} ---")
    print("MAE :", mean_absolute_error(y_true, y_pred))
    print("MSE :", mean_squared_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R²  :", r2_score(y_true, y_pred))
    print()

evaluar_modelo(y_test, y_pred_lineal, "Regresión Lineal")
evaluar_modelo(y_test, y_pred_poly, "Regresión Polinómica")

# === 7. Visualización: Gráficos de dispersión ===
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_lineal, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Regresión Lineal: y_test vs y_pred")
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=y_pred_poly, alpha=0.6, color="orange")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Regresión Polinómica: y_test vs y_pred")
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")

plt.tight_layout()

# === 8. Guardar gráfico ===
plt.savefig(os.path.join(output_folder, "comparacion_modelos_regresion.png"))
plt.show()