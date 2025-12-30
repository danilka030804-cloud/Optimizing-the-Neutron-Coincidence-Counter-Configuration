import optuna
import numpy as np
import pandas as pd

from optuna.samplers import TPESampler, GPSampler

import settings
from settings import set_par


# ============================================================
#  Global physics handler
# ============================================================

par = set_par()


# ============================================================
#  Adaptive Precision Optimizer
# ============================================================

class AdaptivePrecisionOptimizer:
    """
    Multi-stage Bayesian optimisation with adaptive precision
    and dynamic shrinking of the parameter space.
    """

    def __init__(
        self,
        initial_bounds: dict,
        stages: list,
        sampler: str,
        name: str
    ):
        self.bounds = initial_bounds.copy()
        self.stages = stages
        self.sampler_name = sampler
        self.name = name

        self.current_stage = 0
        self.study = None
        self.completed_trials = []
        self.results = []

    # --------------------------------------------------------
    #  Sampler factory
    # --------------------------------------------------------

    @staticmethod
    def _create_sampler(name: str):
        if name == 'TPE':
            return TPESampler()
        if name == 'GP':
            return GPSampler(n_startup_trials=0)
        raise ValueError(f"Unknown sampler: {name}")

    # --------------------------------------------------------
    #  Physics evaluation
    # --------------------------------------------------------

    @staticmethod
    def _eval_physics(neutrons: float, h: float):
        settings.set_neutrons(neutrons)

        a, b, da, db = settings.run_black_box(h)

        f = np.sqrt(a + b) / np.sqrt(3900) / (a - b)
        df = (
            f / (a**2 - b**2)
            * np.sqrt((3*a + b)**2 * (a * da)**2 + (3*b + a)**2 * (b * db)**2)
            / 2
        )

        return a, b, da, db, f, df

    # --------------------------------------------------------
    #  Objective function
    # --------------------------------------------------------

    def _objective(self, trial: optuna.trial.Trial):

        n_trials, neutrons = self.stages[self.current_stage]

        trial.set_user_attr("stage_index", self.current_stage)
        trial.set_user_attr("neutrons", neutrons)

        params = {}

        for name, (low, high) in self.bounds.items():

            if name == 'counters_n':
                params[name] = trial.suggest_int(name, int(low), int(high))

            elif name == 'pos_src':
                val = trial.suggest_float(name, low, high)
                params[name] = max(val, params.get('s_h_prisma_h', low))

            elif name == 's_poly':
                val = trial.suggest_float(name, low, high)
                params[name] = max(val, self.bounds['counters_h'][1] + 2)

            elif name == 'counters_h':
                val = trial.suggest_float(name, low, high)
                params[name] = min(val, params.get('s_poly', high) - 1.5)

            else:
                params[name] = trial.suggest_float(name, low, high)

        # physics call
        par.main(params=params)
        a, b, da, db, f, df = self._eval_physics(neutrons, params['s_h_top_h'])

        # store metadata
        for key, value in {
            "a": a, "b": b,
            "delt_a": da, "delt_b": db,
            "f": f, "delt_f": df
        }.items():
            trial.set_user_attr(key, value)

        # log to pandas
        self.results.append({
            "trial_number": trial.number,
            "stage_index": self.current_stage,
            "neutrons": neutrons,
            **params,
            "a": a, "b": b,
            "delt_a": da, "delt_b": db,
            "f": f, "delt_f": df,
            "objective": f
        })

        return -1000 * f if f < 0 else f

    # --------------------------------------------------------
    #  Space refinement
    # --------------------------------------------------------

    def _refine_search_space(self):

        stage_trials = [
            t for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.user_attrs.get("stage_index") == self.current_stage
        ]

        self.completed_trials.append(stage_trials)
        stage_trials.sort(key=lambda t: t.value)

        best_trials = stage_trials[:max(5, len(stage_trials) // 10)]
        new_bounds = {}

        for name, (low, high) in self.bounds.items():

            values = [t.params[name] for t in best_trials if name in t.params]

            if len(values) < 2:
                new_bounds[name] = (low, high)
                continue

            mean = np.mean(values)
            width = 0.25 * (high - low)

            new_bounds[name] = (
                max(low, mean - width),
                min(high, mean + width)
            )

        self.bounds = new_bounds

    # --------------------------------------------------------
    #  Optimization loop
    # --------------------------------------------------------

    def optimize(self):

        for stage_idx, (n_trials, neutrons) in enumerate(self.stages):
            self.current_stage = stage_idx

            print(
                f"\nStage {stage_idx + 1}: "
                f"{n_trials} trials | neutrons = {neutrons:.1e}"
            )

            self.study = optuna.create_study(
                sampler=self._create_sampler(self.sampler_name),
                direction='minimize'
            )

            self.study.optimize(self._objective, n_trials=n_trials)
            self._refine_search_space()

            best = min(self.completed_trials[stage_idx], key=lambda t: t.value)
            delta = best.user_attrs.get("delt_f", np.nan)

            print(f"Current best = {best.value:.6e} ± {delta:.6e}")

        pd.DataFrame(self.results).to_csv(self.name, index=False)
        return self.study.best_params


# ============================================================
#  Configuration & run
# ============================================================

INITIAL_BOUNDS = {
    's_Cd': (0.05, 2),
    's_poly': (138, 150),
    's_h_prisma_s': (50, 150),
    's_h_prisma_h': (-30, 50),
    'pos_src': (-10, 70),
    's_h_top_a': (15, 25),
    's_h_top_h': (105, 122),
    'counters_n': (1, 8),
    'counters_l': (0, 1),
    'counters_h': (134.2, 145)
}

STAGES = [
    (500, 1e5),
    (100, 1e6),
    (25, 1e7),
    (10, 5e7)
]


if __name__ == "__main__":

    optimizer = AdaptivePrecisionOptimizer(
        initial_bounds=INITIAL_BOUNDS,
        stages=STAGES,
        sampler='GP',
        name="optuna_trials_log_BO1.csv"
    )

    best_params = optimizer.optimize()
    print("\nBest parameters:\n", best_params)
