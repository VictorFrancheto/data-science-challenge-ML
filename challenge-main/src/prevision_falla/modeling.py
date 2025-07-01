import pandas as pd
import optuna
import inspect

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_classifiers(x_train, y_train, x_test, y_test, classifiers=None):
    '''
    Treina e avalia vários classificadores no hold-out, retornando métricas.

    Args:
        x_train, y_train: treino.
        x_test, y_test: teste.
        classifiers: dict opcional {'nome': modelo}. Se None, usa defaults.

    Returns:
        DataFrame com Accuracy, Precision, Recall e F1-score, ordenado por F1.
    '''
    if classifiers is None:
        classifiers = {
            'GradientBoost': GradientBoostingClassifier(),
            'RandomForest':  RandomForestClassifier(),
            'BernoulliNB':   BernoulliNB(),
            'Regressão Log': LogisticRegression(),
            'SVM': SVC(),
            'KNN': KNeighborsClassifier()}

    results = []
    for name, clf in classifiers.items():
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        results.append({
            'Model': name,
            'Accuracy':  accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall':    recall_score(y_test, y_pred, zero_division=0),
            'F1-score':  f1_score(y_test, y_pred, zero_division=0)
        })

    return (pd.DataFrame(results)
            .sort_values('F1-score', ascending=False)
            .reset_index(drop=True))


def create_study(objective, n_trials=50, direction='maximize'):
    '''
    Crea y ejecuta un estudio de Optuna para optimizar hiperparámetros.

    Args:
        objective: función objetivo que recibe un trial y devuelve la métrica a maximizar.
        n_trials: número de iteraciones de búsqueda.
        direction: 'maximize' o 'minimize'.

    Returns:
        best_params: diccionario con los mejores hiperparámetros.
        best_value: valor óptimo alcanzado de la métrica objetivo.
    '''
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def objective_gb(trial, x_train, y_train, x_test, y_test):
    '''
    Función objetivo para optimizar GradientBoostingClassifier con Optuna.

    Args:
        trial: objeto Trial de Optuna.
        x_train, y_train: datos de entrenamiento.
        x_test, y_test: datos de prueba.

    Returns:
        F1-score en el conjunto de prueba.
    '''
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 50, 500),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth':         trial.suggest_int('max_depth', 3, 8),
        'min_samples_split': trial.suggest_float('min_samples_split', 0.1, 1.0),
    }
    clf = GradientBoostingClassifier(**params, random_state=42)
    clf.fit(x_train, y_train)
    return f1_score(y_test, clf.predict(x_test))


def objective_rf(trial, x_train, y_train, x_test, y_test):
    '''
    Función objetivo para optimizar RandomForestClassifier con Optuna.

    Args:
        trial: objeto Trial de Optuna.
        x_train, y_train: datos de entrenamiento.
        x_test, y_test: datos de prueba.

    Returns:
        F1-score en el conjunto de prueba.
    '''
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 10, 200),
        'max_depth':         trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_float('min_samples_split', 0.1, 1.0),
        'min_samples_leaf':  trial.suggest_float('min_samples_leaf', 0.1, 0.5),
        'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2']),
    }
    clf = RandomForestClassifier(**params, random_state=42)
    clf.fit(x_train, y_train)
    return f1_score(y_test, clf.predict(x_test))


def objective_bnb(trial, x_train, y_train, x_test, y_test):
    '''
    Función objetivo para optimizar BernoulliNB con Optuna.

    Args:
        trial: objeto Trial de Optuna.
        x_train, y_train: datos de entrenamiento.
        x_test, y_test: datos de prueba.

    Returns:
        F1-score en el conjunto de prueba.
    '''
    params = {
        'alpha':     trial.suggest_loguniform('alpha', 1e-6, 1.0),
        'binarize':  trial.suggest_float('binarize', 0.0, 1.0),
        'fit_prior': trial.suggest_categorical('fit_prior', [True, False]),
    }
    clf = BernoulliNB(**params)
    clf.fit(x_train, y_train)
    return f1_score(y_test, clf.predict(x_test))


def train_model(params, model_cls, x_train, y_train, x_test, y_test, random_state=42):
    '''
    Ajusta y evalúa un clasificador, pasando random_state si aplica.

    Args:
        params: diccionario de parámetros para el modelo.
        model_cls: clase del modelo a instanciar.
        x_train, y_train: datos de entrenamiento.
        x_test, y_test: datos de prueba.
        random_state: semilla para reproducibilidad.

    Returns:
        model: instancia del modelo entrenado.
        f1: F1-score en el conjunto de prueba.
    '''
    sig = inspect.signature(model_cls.__init__).parameters
    if 'random_state' in sig:
        model = model_cls(**params, random_state=random_state)
    else:
        model = model_cls(**params)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return model, f1_score(y_test, preds)


def build_voting(models_dict, voting='hard'):
    '''
    Crea un VotingClassifier a partir de un dict de modelos.

    Args:
        models_dict: diccionario {'nombre': modelo}.
        voting: 'hard' o 'soft'.

    Returns:
        VotingClassifier configurado.
    '''
    return VotingClassifier(estimators=list(models_dict.items()), voting=voting)

def tune_and_train_models(x_train, y_train, x_test, y_test):
    gb_params, gb_f1 = create_study(lambda t: objective_gb(t, x_train, y_train, x_test, y_test), n_trials=50)
    rf_params, rf_f1 = create_study(lambda t: objective_rf(t, x_train, y_train, x_test, y_test), n_trials=50)
    bnb_params, bnb_f1 = create_study(lambda t: objective_bnb(t, x_train, y_train, x_test, y_test), n_trials=50)

    gb_model, _ = train_model(gb_params, GradientBoostingClassifier, x_train, y_train, x_test, y_test)
    rf_model, _ = train_model(rf_params, RandomForestClassifier, x_train, y_train, x_test, y_test)
    bnb_model, _ = train_model(bnb_params, BernoulliNB, x_train, y_train, x_test, y_test)

    models = [
        ('GradientBoosting', gb_model, gb_f1),
        ('RandomForest', rf_model, rf_f1),
        ('BernoulliNB', bnb_model, bnb_f1)
    ]
    return sorted(models, key=lambda tpl: tpl[2], reverse=True)