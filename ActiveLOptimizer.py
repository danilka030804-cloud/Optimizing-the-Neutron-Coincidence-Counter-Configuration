# ================== IMPORTS ==================

import optuna
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans

from AdaptivePrecisionOptimizer import AdaptivePrecisionOptimizer
import settings
par = settings.set_par()

# ======================================================
#                    MODEL BUILDERS
# ======================================================

def build_bayesian_ridge(params):
    return Pipeline([
        ('poly', PolynomialFeatures(2)),
        ('scaler', StandardScaler()),
        ('reg', BayesianRidge(**params))
    ])


def build_catboost(params):
    return CatBoostRegressor(
        **params,
        verbose=False,
        thread_count=1
    )


def build_forest(params):
    return RandomForestRegressor(
        **params,
        n_jobs=1,
        random_state=42
    )


# ======================================================
#                  HYPERPARAM SEARCH
# ======================================================

def search_bayesian_ridge(trial):
    return {
        'max_iter': trial.suggest_int('max_iter', 100, 500),
        'tol': trial.suggest_float('tol', 1e-4, 1e-2),
        'alpha_1': trial.suggest_float('alpha_1', 1e-7, 1e-5),
        'alpha_2': trial.suggest_float('alpha_2', 1e-7, 1e-5),
        'lambda_1': trial.suggest_float('lambda_1', 1e-7, 1e-5),
        'lambda_2': trial.suggest_float('lambda_2', 1e-7, 1e-5),
    }


def search_catboost(trial):
    return {
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 7),
        'iterations': trial.suggest_int('iterations', 500, 1500),
    }


def search_forest(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 5),
    }


# ======================================================
#                   MODEL REGISTRY
# ======================================================

MODEL_REGISTRY = {
    'bayesian_ridge': {
        'builder': build_bayesian_ridge,
        'search_space': search_bayesian_ridge,
        'fit_params': lambda w: {'reg__sample_weight': w},
    },
    'catboost': {
        'builder': build_catboost,
        'search_space': search_catboost,
        'fit_params': lambda w: {'sample_weight': w},
    },
    'forest': {
        'builder': build_forest,
        'search_space': search_forest,
        'fit_params': lambda w: {'sample_weight': w},
    }
}


# ======================================================
#              SURROGATE MODEL TUNER
# ======================================================

class SurrogateModelTuner:
    """
    Подбор гиперпараметров суррогатной модели (Optuna + CV)
    """

    def __init__(self, df: pd.DataFrame, model_name: str):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'")

        df = df[df['f'].notna() & (df['f'] < 1)]

        self.X = df.iloc[:, :10]
        self.y = df['f']
        self.w = 1.0 / df['delt_f']

        self.cv = KFold(n_splits=5, shuffle=True, random_state=42)
        self.spec = MODEL_REGISTRY[model_name]


    def _objective(self, trial):
        params = self.spec['search_space'](trial)
        model = self.spec['builder'](params)

        score = cross_val_score(
            model,
            self.X,
            self.y,
            cv=self.cv,
            scoring='neg_mean_squared_error',
            n_jobs=1,
            params=self.spec['fit_params'](self.w)
        ).mean()

        return score


    def find(self, n_trials=500):
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=n_trials)
        return study.best_params


# ======================================================
#             PARAMETER SPACE FOR POINT SEARCH
# ======================================================

def point_search_space(trial):
    s_Cd = trial.suggest_float('s_Cd', 0.05, 2)
    s_poly = trial.suggest_float('s_poly', 138, 150)
    s_h_prisma_s = trial.suggest_float('s_h_prisma_s', 50, 150)
    s_h_prisma_h = trial.suggest_float('s_h_prisma_h', -30, 50)
    pos_src = trial.suggest_float('pos_src', max(s_h_prisma_h, -10), 70)
    s_h_top_a = trial.suggest_float('s_h_top_a', 15, 25)
    s_h_top_h = trial.suggest_float('s_h_top_h', 105, 122)
    counters_n = trial.suggest_int('counters_n', 1, 8)
    counters_l = trial.suggest_float('counters_l', 0, 1)
    counters_h = trial.suggest_float('counters_h', 134.2, min(145, s_poly))

    return np.array([
        s_Cd, s_poly, s_h_prisma_s, s_h_prisma_h,
        pos_src, s_h_top_a, s_h_top_h,
        counters_n, counters_l, counters_h
    ]).reshape(1, -1)


# ======================================================
#              BEST POINT SEARCH (SURROGATE)
# ======================================================

class BestPointFinder:
    """
    Поиск оптимальных точек параметров
    по обученному суррогату
    """

    def __init__(self, df, model_name, best_params):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'")

        df = df[df['f'].notna() & (df['f'] < 1)]

        X = df.iloc[:, :10]
        y = df['f']
        w = 1.0 / df['delt_f']

        spec = MODEL_REGISTRY[model_name]
        self.model = spec['builder'](best_params)
        self.model.fit(X, y, **spec['fit_params'](w))


    def _objective(self, trial):
        x = point_search_space(trial)
        f = float(self.model.predict(x))

        # штраф за невалидную область
        if f < 0:
            f *= -1000

        return f


    def find(self, n_trials=1000):
        study = optuna.create_study(direction='minimize')
        study.optimize(self._objective, n_trials=n_trials)
        return study.trials_dataframe(), study.best_params


# ======================================================
#                    EXAMPLE USAGE
# ======================================================

if __name__ == "__main__":
    df = pd.read_csv("latina_cube.csv")

    model_name = 'forest'   # 'bayesian_ridge' | 'catboost' | 'forest'

    tuner = SurrogateModelTuner(df, model_name)
    best_hyperparams = tuner.find(n_trials=500)

    finder = BestPointFinder(df, model_name, best_hyperparams)
    trials_df, best_point = finder.find(n_trials=1000)

    print("Best hyperparams:", best_hyperparams)
    print("Best point:", best_point)


# ======================================================
#                  CLUSTER BOUNDS
# ======================================================

def cluster_bounds(
    data: pd.DataFrame,
    q_low: float = 0.1,
    q_high: float = 0.9,
    prefix: str = "params_"
) -> dict:
    """
    Определяет границы параметров по квантилям внутри кластера
    """
    bounds = {}

    for col in data.columns:
        if col.startswith(prefix):
            name = col.replace(prefix, "")
            bounds[name] = (
                data[col].quantile(q_low),
                data[col].quantile(q_high)
            )

    return bounds


# ======================================================
#                  OPTIMIZE CLUSTER
# ======================================================

def optimize_clusters(
    trials_df: pd.DataFrame,
    n_clusters: int = 3,
    opt_trials: int = 50,
    neutrons: float = 1e6
):
    """
    Кластеризация суррогатных оптимумов и локальный BO в каждом кластере
    """
    param_cols = [c for c in trials_df.columns if c.startswith("params_")]
    X = trials_df[param_cols]

    labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)
    cluster_files = []

    for i in range(n_clusters):
        cluster_data = trials_df[labels == i]
        bounds = cluster_bounds(cluster_data)

        optimizer = AdaptivePrecisionOptimizer(
            initial_bounds=bounds,
            stages=[(opt_trials, neutrons)],
            sampler='TPE',
            name=f"cluster_{i+1}.csv"
        )

        optimizer.optimize()
        cluster_files.append(f"cluster_{i+1}.csv")

    return cluster_files


# ======================================================
#                  BEST CLUSTER
# ======================================================

def select_best_cluster(cluster_files: list) -> int:
    """
    Выбор кластера с наилучшим средним значением f
    """
    means = []

    for fname in cluster_files:
        df = pd.read_csv(fname)
        mean_f = df.loc[df['f'] > 0, 'f'].mean()
        means.append(mean_f)

    return int(np.nanargmax(means)) + 1


# ======================================================
#              REFINE BEST CLUSTER
# ======================================================

def refine_winner_cluster(
    cluster_id: int,
    df_source: pd.DataFrame,
    model_name: str,
    n_iter: int = 6,
    neutrons: float = 5e7
):
    """
    Итеративное дообогащение данных в лучшем кластере
    """
    filename = f"cluster_{cluster_id}.csv"

    for _ in range(n_iter):
        data = pd.read_csv(filename)

        # 1. переобучение суррогата
        best_params = SurrogateModelTuner(df_source, model_name).find()

        # 2. поиск новой точки
        _, params = BestPointFinder(
            df_source, model_name, best_params
        ).find()

        # 3. физический расчёт
        par.main(params)
        settings.set_neutrons(neutrons)

        a, b, da, db = settings.run_black_box(params['s_h_top_h'])
        f = np.sqrt(a + b) / np.sqrt(3900) / (a - b)
        df_val = (
            f / (a**2 - b**2)
            * np.sqrt((3*a + b)**2 * (a * da)**2 + (3*b + a)**2 * (b * db)**2)
            / 2
        )

        row = {
            **params,
            "a": a, "b": b,
            "delt_a": da, "delt_b": db,
            "f": f, "delt_f": df_val,
            "neutrons": neutrons,
            "stage_index": 1,
            "trial_number": len(data)
        }

        data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)
        data.to_csv(filename, index=False)


# ======================================================
#                   ORCHESTRATION
# ======================================================

if __name__ == "__main__":
    df = pd.read_csv("latina_cube.csv")

    best_surrogate_params = SurrogateModelTuner(df, 'catboost').find()
    trials_df, _ = BestPointFinder(df, 'catboost', best_surrogate_params).find()
    cluster_files = optimize_clusters(trials_df)
    winner = select_best_cluster(cluster_files)
    refine_winner_cluster(winner, df, model_name='catboost')

