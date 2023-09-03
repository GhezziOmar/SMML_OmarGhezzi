![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyvhr)
[![GitHub license](https://img.shields.io/github/license/phuselab/pyVHR)](https://github.com/phuselab/pyVHR/blob/master/LICENSE)

# Ridge regression/kernel ridge regression

This repository stores the code, Jupyter notebook, and PDF reports associated with the experimental project on (Kernel) Ridge Regression, part of the "Statistical Methods for Machine Learning" course examination for the academic year 2022/2023.

## Code

The code features two 'main' scripts: 
- **main.py** : for the continuous features task.
- **main2.py** : including also binary, ordinal and categorial features.

The other scripts consits of the main classes and the usefull functions:
- **RidgeRegression.py**: implements the closed-form solution for ridge regression.
- **expectedRiskEstimate.py**: implements K-fold nested cross-validation and bestCV estimates (the latter being a less computationally demanding but statistically less rigorous way to do hyperparamethers tuning jointly with cross-validation expected risk estimate).
- **TargetEncoding.py**: implements one possible way to encode categorical features (replacing each value, realizing the i-th feature, with the average of the corresponding target values in the training set).
- **KernelRidgeRegression.py**: implements the closed-form solution for kernel ridge regression.
- **DCKRR.py**: implements the divide-and-conquer heuristc for kernel ridge regression (simplest way to try to use all the available data to run kernel ridge regression without incurring in time or space issues).
- **functions.py**: usefull functions, principaly to plot and save results.

You can run the 'main' scripts from the command line as:
```bash
python3 main.py
```
```bash
python3 main2.py
```
Alternatively, you can view the pre-run Jupyter notebook file, which already displays key results and plots. Additionally, the 'Results' directory includes .txt and .csv files featuring dataframes that present risk estimates and the optimal weights of predictors.
## Rreports

The directory 'Reports' contains two PDF files:
- RidgeRegression: report containing the critical analysis of the results regarding executing ridge regression to predict the tracks’ popularity on continuous features only.
- KernelRidgeRegression: this report offers an analysis of the outcomes when utilizing Kernel Ridge Regression (with a Gaussian kernel) to predict music track popularity on Spotify.

The choice to create two distinct PDF documents, along with the comprehensive coverage that surpasses the usual 15-16 page limit, arises from our aim to thoroughly explore and synthesize the theoretical foundation associated with the "statistical learning framework" and "RHKS theory". These subjects, while covered during lectures, were also enriched with insights from external sources. Our main objective is to provide a detailed illustration of how we bridged theoretical concepts to practical application, rather than merely focusing on coding and presenting outcomes. 
For a more concise yet comprehensive read that requires less time, you can focus on the **methods** and **results** sections in the PDF reports, skipping the **theoretical background** section.

 
