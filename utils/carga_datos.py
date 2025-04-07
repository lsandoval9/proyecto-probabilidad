import pandas as pd

def cargar_datos(ruta):
    """Carga el dataset y muestra información básica."""
    df = pd.read_csv(ruta, sep=';', decimal=',', encoding='ISO-8859-1')
    print("=== Resumen inicial del dataset ===")
    print("\n=== Tipos de datos ===")
    print(df.dtypes)
    print("\n=== Valores faltantes ===")
    print(df.isnull().sum())
    return df