from pathlib import Path
from typing import Tuple #describe que se envía y que puede devolver una funcion
import pandas as pd
from sklearn.compose import ColumnTransformer #Lo que haces es tranformas las columnas que yo le diga de acuerdo a tales parametros
from sklearn.impute import SimpleImputer # Manejar valores faltantes
from sklearn.pipeline import Pipeline  # encadena pasos: paso1 → paso2 → paso3 (ej. imputer → scaler)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler # Se usa para poner las variables en un rango similar


# URL directa al CSV (para descargar si no hay caché local)
DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
)


def load_data(
    data_url: str = DATA_URL,
    cache_path: Path = Path("data/diabetes.csv"),
    ) -> pd.DataFrame:

    if cache_path.exists():
        return pd.read_csv(cache_path, sep=",")
    
    df = pd.read_csv(data_url, sep=",")
    
    # Si la carpeta data no existiera la crea por defecto
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(cache_path, index=False, sep=",")
    
    #Retorna finalmente el dataframe
    return df


#Funcion de separación de caracteristicas y el target osea la "y"
def split_features_target(
    df: pd.DataFrame, 
    target: str = "Outcome"
    ) -> Tuple[pd.DataFrame, pd.Series]:
    
    x = df.drop(columns=[target])
    y = df[target]
    
    return x, y


#Funcion para construir el Pipeline
def built_pipeline(numeric_features: list[str]) -> Pipeline:
    
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer", SimpleImputer(strategy="median")
            ),
            (
                "scaler", StandardScaler()
            ),
                
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            )
        ],
        remainder="drop"  # Columnas que no tienen importancia, eliminarlas
    )
    
    #clase de scikit-learn para clasificación binaria, Predice probabilidades y luego una clase (por ejemplo 0 o 1).
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,  # inverso de la regularización; menor = más regularización
    )
    
    pipeline = Pipeline(
        steps=[
        ("preprocess", preprocessor),
        ("model", model)
        ]
    )
    
    return pipeline

    

