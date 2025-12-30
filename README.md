# Optimizing-the-Neutron-Coincidence-Counter-Configuration

## Project Description
This project is dedicated to optimizing the configuration of a neutron coincidence counter modeled in the **Serpent 2** transport code. The problem is solved as a "black-box" optimization using methods:
- Bayesian Optimization (BO)
- Tree-structured Parzen Estimator (TPE)
- Active learning with surrogate models

Goal: Minimize the relative counting error with limited computational budget and stochastic nature of Monte Carlo simulation.

---

## Project Goals
- Develop a modular framework for optimizing neutron-physics model parameters
- Compare the effectiveness of BO, TPE, and hybrid methods
- Implement adaptive multi-stage optimization with calculation precision control
- Automate the neutron detector design process

---

## Optimization Methods

### Bayesian Optimization (BO)
- Uses Gaussian Process (GP) for modeling the objective function
- Acquisition function: Upper Confidence Bound (UCB)
- Supports adaptive control of calculation precision

### Tree-structured Parzen Estimator (TPE)
- Based on density estimation using Parzen kernels
- Effective for categorical and mixed parameter spaces

### Hybrid Method (Active Learning)
- Combines surrogate models (Random Forest, CatBoost) with TPE
- Includes parameter space clustering and local optimization
- Reduces the number of expensive queries to Serpent

---

## Experimental Results
Comparative analysis showed:
- **Hybrid method** demonstrates the best balance between convergence speed and solution quality
- **BO with GP** provides high stability but requires more computational resources
- **TPE** is effective in early search stages

---

## Result Visualization
Jupyter notebooks in the `notebooks/` folder are used for result analysis, including:
- Method convergence graphs
- Error heatmaps
- Boxplot parameter distributions

---

## Authors
- **Nikiforov D.O.** — development, experiments, analysis
- **Pugachev P.A.** — scientific supervision

---

## References
- Serpent 2: [https://serpent.vtt.fi/](https://serpent.vtt.fi/)
- Optuna: [https://optuna.org/](https://optuna.org/)
- BO methods paper: Gardner et al., 2023

---

**Keywords:** neutron physics, black-box optimization, Bayesian optimization, Serpent 2, Monte Carlo, active learning, surrogate models
