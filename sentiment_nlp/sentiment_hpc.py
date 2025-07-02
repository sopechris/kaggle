import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from mpi4py import MPI
from sklearn.model_selection import cross_val_score
import optuna

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main():
    # Only rank 0 loads data and broadcasts to others
    if rank == 0:
        df = pd.read_csv('sentiment_data.csv')
        df.dropna(inplace=True)
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        X = df['Comment']
        y = df['Sentiment']
    else:
        X = None
        y = None

    # Broadcast data to all ranks
    X = comm.bcast(X, root=0)
    y = comm.bcast(y, root=0)

    # Split data (all ranks do the same split for simplicity)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Vectorize text
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Optuna hyperparameter optimization
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=1),
            'max_depth': trial.suggest_int('max_depth', 3, 20, step=1),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 3.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'use_label_encoder': False,
            'eval_metric': 'mlogloss',
            'random_state': 42
        }
        model = XGBClassifier(**params)
        score = cross_val_score(model, X_train_vectorized, y_train, cv=5, scoring='f1_macro', n_jobs=1).mean()
        return score

    # Use a shared SQLite DB file (must be on a shared filesystem)
    storage_url = 'sqlite:///optuna_study.db'

    sampler = optuna.samplers.TPESampler(n_startup_trials=300)

    if rank == 0:
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='xgb_sentiment_v2',  # <-- use this name
            storage=storage_url,
            load_if_exists=True
        )
    else:
        study = optuna.load_study(
            study_name='xgb_sentiment_v2',  # <-- use the same name here!
            storage=storage_url,
            sampler=sampler
        )

    study.optimize(objective, n_trials=1500)

    if rank == 0:
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Train final model and evaluate
        final_xgb = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            **trial.params
        )
        final_xgb.fit(X_train_vectorized, y_train)
        y_pred = final_xgb.predict(X_test_vectorized)
        print("Test Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Classical Grid Search Version (for reference) ---
# from sklearn.model_selection import GridSearchCV
# param_grid = {...}
# grid = GridSearchCV(XGBClassifier(...), param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
# grid.fit(X_train_vectorized, y_train)
# print("Best params:", grid.best_params_)
# print("Best score:", grid.best_score_)

if __name__ == "__main__":
    main()