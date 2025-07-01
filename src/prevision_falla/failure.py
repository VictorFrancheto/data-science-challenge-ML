import pandas as pd
from data_loader import load_data, save_predictions
from data_prep import preprocess_data, prepare_features_targets, undersample, split_and_normalize
from modeling import build_voting, tune_and_train_models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class FailurePredictionPipeline:
    def __init__(self, file_name='full_devices.csv'):
        self.file_name = file_name
        self.df = None
        self.X = None
        self.y = None
        self.best_model = None
        self.scaler = None

    def run(self):
        self._load_and_prepare_data()
        self._train_models()
        self._final_predictions()

    def _load_and_prepare_data(self):
        self.df = load_data(filename=self.file_name)
        self.df = preprocess_data(self.df)
        self.X, self.y = prepare_features_targets(self.df)

    def _train_models(self):
        X_res, y_res = undersample(self.X, self.y)
        x_train, x_test, y_train, y_test = split_and_normalize(X_res, y_res)
        tuned_models_sorted = tune_and_train_models(x_train, y_train, x_test, y_test)
        voting_clf = build_voting({n: m for n, m, _ in tuned_models_sorted}, voting='hard')
        voting_clf.fit(x_train, y_train)
        self.best_model = tuned_models_sorted[0][1]

    def _final_predictions(self):
        X_res, y_res = undersample(self.X, self.y)
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
        )

        self.scaler = StandardScaler().fit(X_train_df)
        x_train_final = self.scaler.transform(X_train_df)
        x_test_final = self.scaler.transform(X_test_df)

        probs_fail = self.best_model.predict_proba(x_test_final)[:, 1]
        y_pred = self.best_model.predict(x_test_final)

        X_test_df = X_test_df.copy()
        X_test_df['prob_failure'] = probs_fail
        X_test_df['predicted_class'] = y_pred
        X_test_df['true_class'] = y_test.values

        save_predictions(X_test_df)
