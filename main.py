from utils.carga_datos import cargar_datos
from utils.preprocesamiento import limpiar_datos, preparar_datos
from modelos.entrenamiento import (
    entrenar_modelo_lineal,
    entrenar_ridge,
    entrenar_lasso,
    entrenar_elastic_net,
    entrenar_stepwise
)
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

def main():

    # Configuración de la semilla para reproducibilidad
    seed = 2022
    np.random.seed(seed)
    random.seed(seed)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # Cargar datos
    df = cargar_datos('data/Ab19selec.csv')

    # Limpiar datos
    df_limpio = limpiar_datos(df)

    # Nombre tecnico de la variable objetivo
    target = 'kg/ADT'

    # Preparar datos
    X_train, X_test, y_train, y_test, _, feature_names  = preparar_datos(df_limpio, target)


    # Obtener correlaciones
    df_train = pd.DataFrame(X_train, columns=feature_names)

    df_train['target'] = y_train.values  # Añade el target

    # Correlación de Pearson entre cada feature y el target
    correlaciones_target = df_train.corr()['target'].drop('target').sort_values(ascending=False)

    print("\n=== Correlaciones (Pearson) con el target ===")
    print(correlaciones_target)

    # Entrenar todos los modelos
    modelo_lineal, mse_lineal, r2_lineal, coef_lineal  = entrenar_modelo_lineal(X_train, y_train, X_test, y_test)

    modelo_ridge, mse_ridge, r2_ridge, coef_ridge = entrenar_ridge(X_train, y_train, X_test, y_test)

    modelo_lasso, mse_lasso, r2_lasso, coef_lasso = entrenar_lasso(X_train, y_train, X_test, y_test)

    modelo_enet, mse_enet, r2_enet, coef_enet = entrenar_elastic_net(X_train, y_train, X_test, y_test)

    modelo_stepwise, mse_stepwise, r2_stepwise, _, coef_stepwise_full = entrenar_stepwise(X_train, y_train, X_test, y_test)

    # Mostrar resultados
    print("\n=== Resultados Comparativos ===")
    print(f"Regresión Lineal    - MSE: {mse_lineal:.2f}, R²: {r2_lineal:.2f}")
    print(f"Ridge               - MSE: {mse_ridge:.2f}, R²: {r2_ridge:.2f}")
    print(f"Lasso               - MSE: {mse_lasso:.2f}, R²: {r2_lasso:.2f}")
    print(f"Elastic Net         - MSE: {mse_enet:.2f}, R²: {r2_enet:.2f}")
    print(f"Stepwise            - MSE: {mse_stepwise:.2f}, R²: {r2_stepwise:.2f}")

    plt.figure(figsize=(12, 8))
    correlaciones_target.plot(kind='bar')
    plt.title(f'Correlación de Features con el Target ({target})')
    plt.xlabel('Features')
    plt.ylabel('Correlación de Pearson')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

    model_coefs = {
        "Regresión Lineal": coef_lineal,
        "Ridge": coef_ridge,
        "Lasso": coef_lasso,
        "Elastic Net": coef_enet,
        "Stepwise": coef_stepwise_full
    }

    for model_name, coefficients in model_coefs.items():
        if coefficients is not None:

            if len(coefficients) == len(feature_names):
                try:
                    coef_series = pd.Series(coefficients, index=feature_names)

                    coef_series_sorted = coef_series.abs().sort_values(ascending=False)

                    coef_series_to_plot = coef_series[coef_series_sorted.index]

                    plt.figure(figsize=(15, 8))
                    coef_series_to_plot.plot(kind='bar', width=0.8)
                    plt.title(f'Coeficientes de Regresión - Modelo: {model_name}')
                    plt.xlabel('Features')
                    plt.ylabel('Valor del Coeficiente')
                    plt.xticks(rotation=90, fontsize=8)
                    plt.grid(axis='y', linestyle='--')
                    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Error al generar gráfico para el modelo '{model_name}': {e}")
            else:

                print(f"Advertencia: No se puede graficar '{model_name}'. "
                      f"Longitud de coeficientes ({len(coefficients)}) no coincide con "
                      f"longitud de feature_names ({len(feature_names)}).")
        else:
            print(f"Advertencia: Coeficientes no disponibles para el modelo '{model_name}'.")

if __name__ == "__main__":
    main()