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

def main():

    # Configuración de la semilla para reproducibilidad
    seed = 2022
    np.random.seed(seed)
    random.seed(seed)

    # Cargar datos
    df = cargar_datos('data/Ab19selec.csv')

    # Limpiar datos
    df_limpio = limpiar_datos(df)

    # Nombre tecnico de la variable objetivo
    target = 'kg/ADT'

    # Preparar datos
    X_train, X_test, y_train, y_test, _ = preparar_datos(df_limpio, target)

    # Entrenar todos los modelos
    modelo_lineal, mse_lineal, r2_lineal = entrenar_modelo_lineal(X_train, y_train, X_test, y_test)

    modelo_ridge, mse_ridge, r2_ridge = entrenar_ridge(X_train, y_train, X_test, y_test)

    modelo_lasso, mse_lasso, r2_lasso = entrenar_lasso(X_train, y_train, X_test, y_test)

    modelo_enet, mse_enet, r2_enet = entrenar_elastic_net(X_train, y_train, X_test, y_test)

    modelo_stepwise, mse_stepwise, r2_stepwise, _ = entrenar_stepwise(X_train, y_train, X_test, y_test)

    # Mostrar resultados
    print("\n=== Resultados Comparativos ===")
    print(f"Regresión Lineal    - MSE: {mse_lineal:.2f}, R²: {r2_lineal:.2f}")
    print(f"Ridge               - MSE: {mse_ridge:.2f}, R²: {r2_ridge:.2f}")
    print(f"Lasso               - MSE: {mse_lasso:.2f}, R²: {r2_lasso:.2f}")
    print(f"Elastic Net         - MSE: {mse_enet:.2f}, R²: {r2_enet:.2f}")
    print(f"Stepwise            - MSE: {mse_stepwise:.2f}, R²: {r2_stepwise:.2f}")

if __name__ == "__main__":
    main()