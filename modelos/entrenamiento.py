from sklearn.linear_model import  RidgeCV, LassoCV, ElasticNetCV, LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
import numpy as np

def entrenar_modelo_lineal(X_train, y_train, X_test, y_test):
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    mse = mean_squared_error(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)
    return modelo, mse, r2, modelo.coef_

def entrenar_ridge(X_train, y_train, X_test, y_test):

    alphas = np.logspace(-6, 6, 13)

    modelo = RidgeCV(alphas=alphas, store_cv_results=True)
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    mse = mean_squared_error(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)
    print(f"  Ridge - Best alpha found: {modelo.alpha_:.4f}")
    return modelo, mse, r2, modelo.coef_


def entrenar_lasso(X_train, y_train, X_test, y_test, n_alphas=100, random_state=2022):
    modelo = LassoCV(n_alphas=n_alphas, random_state=random_state, cv=5, tol=0.001, max_iter=20000)
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    mse = mean_squared_error(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)
    print(f"  Lasso - Best alpha found: {modelo.alpha_:.4f}")
    return modelo, mse, r2, modelo.coef_


def entrenar_elastic_net(X_train, y_train, X_test, y_test, l1_ratios=[.1, .5, .7, .9, .95, .99, 1], n_alphas=100, random_state=2022):

    modelo = ElasticNetCV(l1_ratio=l1_ratios, n_alphas=n_alphas, random_state=random_state, cv=5,
                          max_iter=20000, tol=0.001)
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    mse = mean_squared_error(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)
    print(f"  ElasticNet - Best alpha: {modelo.alpha_:.4f}, Best l1_ratio: {modelo.l1_ratio_:.2f}")
    return modelo, mse, r2, modelo.coef_

def entrenar_stepwise(X_train, y_train, X_test, y_test, max_iter=100):

    # Entrenar modelo final con sklearn
    base_model = LinearRegression()

    selector = SequentialFeatureSelector(
        base_model,
        direction="forward",
        scoring="r2",
        cv=5,
        n_features_to_select="auto"  # Selección automática
    )

    selector.fit(X_train, y_train)

    selected_features = selector.get_support(indices=True)

    modelo = LinearRegression()

    modelo.fit(X_train[:, selected_features], y_train)

    # Métricas
    predicciones = modelo.predict(X_test[:, selected_features])
    mse = mean_squared_error(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)

    print(f"  Stepwise - Features seleccionadas: {len(selected_features)}")
    return modelo, mse, r2, selected_features, modelo.coef_
