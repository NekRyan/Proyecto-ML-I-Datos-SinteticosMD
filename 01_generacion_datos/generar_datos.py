# generar_datos.py

import pandas as pd
import numpy as np
import random
from sklearn.datasets import make_classification, make_regression
import os

# Para reproducibilidad
np.random.seed(42)

# Cantidad de registros
n = 1000

# Variables categóricas
rubros = ['Retail', 'Salud', 'Educación', 'Tecnología', 'Gastronomía', 'Otros']
tamaños = ['Pequeña', 'Mediana', 'Grande']
tipos_servicio = ['Publicidad', 'Redes Sociales', 'Branding', 'SEO/SEM', 'Consultoría', 'Diseño']
zonas = ['Lima', 'Norte', 'Sur', 'Centro', 'Oriente']
frecuencias_contacto = ['Baja', 'Media', 'Alta']
niveles_digitalizacion = ['Bajo', 'Medio', 'Alto']

# Generación de datos
data = pd.DataFrame({
    'cliente_id': range(1, n + 1),
    'rubro_cliente': np.random.choice(rubros, n),
    'tamaño_empresa': np.random.choice(tamaños, n, p=[0.5, 0.35, 0.15]),
    'presupuesto_mensual': np.round(np.random.uniform(200, 20000, n), 2),
    'tipo_servicio_solicitado': np.random.choice(tipos_servicio, n),
    'duración_interacción_meses': np.random.randint(0, 37, n),
    'visitas_web_mensuales': np.random.randint(100, 50001, n),
    'interacciones_redes_mensuales': np.random.randint(10, 10001, n),
    'num_empleados': np.random.randint(1, 1001, n),
    'zona_geográfica': np.random.choice(zonas, n),
    'frecuencia_contacto': np.random.choice(frecuencias_contacto, n, p=[0.3, 0.5, 0.2]),
    'nivel_digitalizacion': np.random.choice(niveles_digitalizacion, n, p=[0.2, 0.5, 0.3]),
    'satisfacción_cliente': np.random.randint(1, 11, n)
})

# Variable objetivo de clasificación
data['contrató_servicio'] = (
    (data['presupuesto_mensual'] > 5000).astype(int) +
    (data['satisfacción_cliente'] > 7).astype(int) +
    (data['nivel_digitalizacion'] == 'Alto').astype(int)
)
data['contrató_servicio'] = (data['contrató_servicio'] >= 2).astype(int)

# Variable objetivo de regresión
ventas = []
for i in range(n):
    if data.loc[i, 'contrató_servicio'] == 1:
        base = data.loc[i, 'presupuesto_mensual'] * np.random.uniform(2, 5)
        mod = base + np.random.normal(0, 5000)
        ventas.append(np.round(min(mod, 100000), 2))  # máximo 100,000
    else:
        ventas.append(0.0)

data['monto_ventas_futuras'] = ventas

# Guardar en CSV
data.to_csv("clientes_sinteticos.csv", index=False)

print("✅ Datos generados y guardados correctamente en 'clientes_sinteticos.csv'")
