from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import random

def limpiar_datos(df):
    """Elimina columnas no relevantes y filas duplicadas."""

    # Eliminar filas duplicadas
    df = df.drop_duplicates()

    imputer = SimpleImputer(strategy='mean')

    df_imputado = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    return df_imputado

def preparar_datos(df, target_col, test_size=0.2, random_state=2022):

    cols_to_drop = [
        target_col,

        # --- Variables Mecánicas/Control Operativo ---
        "CE05_IP21_547WI1750",   # Torque Motor Hidráulico preblanqueo
        "CE05_IP21_547WI3250",   # Torque Motor Hidraulico D0
        "CE05_IP21_547WI3650",   # Torque Motor Hidraulico D1
        "CE05_IP21_547WIC1450",  # Torque maximo prensa EOP
        "CE05_IP21_547WIC1850",  # Torque maximo prensa D2
        "CE05_IP21_547XI1215D",  # Producción prensa D0
        "CE05_IP21_547XI1415D",  # Producción prensa Eop
        "CE05_IP21_547XI1615D",  # Producción prensa D1
        "CE05_IP21_547XI1815D",  # Producción prensa D2
        "CE05_IP21_547XI1994D",  # Produccion Prensa Pre-Blanq L1

        # --- Variables de Nivel/Posición Redundantes ---
        "CE05_IP21_547LI1201",   # Nivel Etapa DO
        "CE05_IP21_547LI1401",   # Nivel Etapa EOP
        "CE05_IP21_547LI1601",   # Nivel Etapa D1
        "CE05_IP21_547LI1801",   # Nivel Etapa D2
        "CE05_IP21_547LIC1200",  # Nivel Etapa D0 _2
        "CE05_IP21_547LIC1400",  # Nivel Etapa Eop_2
        "CE05_IP21_547LIC1800",  # Nivel Etapa D2_2

        # --- Variables Constantes/Blancuras Fijas ---
        "CE05_L21_612AI547.29.4",  # Blancura D0 (Lab) L1 (51.5% fijo)
        "CE05_L21_612AI547.29.7",  # Blancura D1 (Lab) L1 (88.2% fijo)
        "CE05_L21_612AI547.29.8",  # Blancura D2 (Lab) L1 (89.7% fijo)
        "CE05_L21_615AI556.2.2",   # Concentracion CLO2 (~9.16 g/l fijo)

        # --- Variables Redundantes/Consumos Parciales ---
        "CE05_IP21_547_CLO2_TOTAL",  # Consumo Total ClO2 (kg/s)
        "CE05_IP21_547_CLO2_D0_L1",  # Consumo ClO2 Etapa D0 L1
        "CE05_IP21_547_CLO2_D1_L1",  # Consumo ClO2 Etapa D1 L1
        "CE05_IP21_547_CLO2_D2_L1",  # Consumo ClO2 Etapa D2 L1
        "CE05_IP21_547FI1930",       # Totalizador CIO2 L1

        # --- Variables de Control/Identificadores Técnicos ---
        "CE05_IP21_547FI1019",       # Identificador técnico
        "CE05_IP21_547NIC1007",      # Control Cs entrada P. Pre-Blanq
        "CE05_IP21_547NIC1207",      # Control Cs entrada prensa D0
        "CE05_IP21_547NIC1407",      # Control Cs entrada prensa EOP
        "CE05_IP21_547NIC1607",      # Control Cs entrada prensa D1
        "CE05_IP21_547NIC1807",      # Control Cs entrada prensa D2

        # --- Variables Indirectas/PH Redundantes ---
        "CE05_IP21_547AI1245",       # PH etapa D0 indirecta
        "CE05_IP21_547AI1445",       # PH Etapa EOP indirecta
        "CE05_IP21_547AI1645",       # PH etapa D1 indirecta
        "CE05_IP21_547AI1845",       # PH etapa D2 indirecta

        # --- Otras No Relevantess ---
        "CE05_IP21_547CI1772",       # Conductividad Prensa PreBlanqueo (0 en todos los datos)
        "CE05_IP21_547PI1336",       # Presion OP-Reactor _2 (redundante)
    ]

    cols_to_drop = [col for col in cols_to_drop if col in df.columns]

    cols_to_drop += df.columns[df.isnull().all()].tolist()

    X = df.drop(columns=cols_to_drop)

    feature_names = X.columns.tolist()

    y = df[target_col]

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    numeric_cols_train = X_train.select_dtypes(include=np.number).columns

    numeric_cols_test = X_test.select_dtypes(include=np.number).columns

    imputer = SimpleImputer(strategy='mean')

    X_train[numeric_cols_train] = imputer.fit_transform(X_train[numeric_cols_train])

    X_test[numeric_cols_test] = imputer.transform(X_test[numeric_cols_test])

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names