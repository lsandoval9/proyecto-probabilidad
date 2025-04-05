import pandas as pd


def limpiar_datos(df):
    df = df.dropna()
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
        
    return df

def cargar_datos(ruta_archivo):
    # Lee el archivo CSV y devuelve un DataFrame
    try:
        
        df = pd.read_csv(ruta_archivo, encoding = "ISO-8859-1", on_bad_lines='skip')
        df = limpiar_datos(df)
        df = df.drop_duplicates()
        return df
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

if __name__ == "__main__":
    
    ruta_csv = "Ab19selec.csv"
    
    datos = cargar_datos(ruta_csv)
    
    if datos is not None:
        print(datos.head())
