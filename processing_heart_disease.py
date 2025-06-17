
"""
processing_heart_disease.py

Transforma el dataset heart_2020_cleaned.csv:
- Aplica StandardScaler a variables numéricas
- Codifica variables categóricas binarias a 0/1
- Aplica one-hot encoding a categóricas de 3+ categorías
Guarda el resultado en processing_heart_disease.csv

Fecha: 17 de junio de 2025
Autor: Bernardo Corona D.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Cargar datos
df = pd.read_csv('heart_2020_cleaned.csv')

# Identificar variables numéricas y categóricas
num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(exclude='number').columns.tolist()

# 1. Escalar variables numéricas
scaler = StandardScaler()
df_num = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

# 2. Procesar variables categóricas
df_cat = pd.DataFrame()

for col in cat_cols:
    categorias = df[col].unique()
    n_categorias = len(categorias)
    if n_categorias == 2:
        # Codificación binaria (0/1)
        df_cat[col] = (df[col] == categorias[1]).astype(int)
    else:
        # One-hot encoding (drop_first evita multicolinealidad)
        df_onehot = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df_cat = pd.concat([df_cat, df_onehot], axis=1)

# 3. Concatenar el dataframe final
df_final = pd.concat([df_num, df_cat], axis=1)

# 4. Guardar archivo resultado
df_final.to_csv('processing_heart_disease.csv', index=False)
