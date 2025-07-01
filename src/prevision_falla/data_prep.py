import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime

def add_date_features(df):
    '''
    Agrega características temporales sin eliminar la columna 'date'.

    Args:
        df (pd.DataFrame): DataFrame con la columna 'date' en formato datetime.

    Returns:
        pd.DataFrame: copia de df con las siguientes columnas nuevas:
            - day_of_week: día de la semana (0=lunes … 6=domingo)
            - day_of_month: día del mes (1–31)
            - is_weekend: indicador de fin de semana (1 = sábado/domingo, 0 = día hábil)
            - month: mes del año (1–12)
            - week: número de semana ISO
        conservando la columna original 'date'.
    '''
    df = df.copy()
    df['day_of_week']  = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
    df['month']        = df['date'].dt.month
    df['week']         = df['date'].dt.isocalendar().week
    
    return df


def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df = add_date_features(df)
    df = df[df['device'] != 'Z1F2']
    df['target'] = df.groupby('device')['failure'].shift(-1)
    df['device_model']=df['device'].apply(lambda x : x[:4])
    df.drop(['device', 'date', 'attribute8'], axis=1, inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    return df



def undersample(X, y, random_state=42):
    '''
    Aplica undersampling para balancear las clases.

    Args:
        X: DataFrame con las variables predictoras.
        y: Series con la variable objetivo.
        random_state: semilla para reproducibilidad.

    Returns:
        X_res, y_res: arrays re-muestreados con la clase mayoritaria reducida.
    '''
    rus = RandomUnderSampler(random_state=random_state)
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res


def split_and_normalize(X, y, test_size=0.2, random_state=42):
    '''
    Divide los datos en entrenamiento y prueba, y estandariza las características.

    Args:
        X: array o DataFrame con las variables predictoras (ya balanceadas).
        y: array o Series con la variable objetivo (ya balanceada).
        test_size: proporción del conjunto para prueba.
        random_state: semilla para reproducibilidad.

    Returns:
        x_train_norm, x_test_norm, y_train, y_test: arrays listos para entrenamiento.
    '''
    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_test_norm  = scaler.transform(x_test)
    return x_train_norm, x_test_norm, y_train, y_test

def prepare_features_targets(df):
    X = df.copy().dropna()
    y = X['target']
    X = X.drop('target', axis=1)
    return X, y
