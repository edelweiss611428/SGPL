# Sparse-Group Pliable Lasso

`SGPL` is an R package implementing the **Sparse-Group Pliable Lasso**, a regularised varying-coefficient regression framework for high-dimensional data containing grouped predictors and modifying variables.

---

## What is this method about?

Modern data often display **effect modification**, where the effect of a main predictor $X$ on a response $Y$ depends on background characteristics or environmental modifiers $Z$. 

While existing methods like the **Pliable Lasso (PL)** capture varying coefficients and the **Group Pliable Lasso (GPL / svReg)** incorporate group-level structure, they suffer from two major limitations:
1. **Lack of within-group main-effect sparsity:** In GPL, individual predictors within an active group cannot be zeroed out.
2. **Missing predictor-level hierarchy within groups:** Active interaction groups in GPL do not guarantee that individual predictors within that group are non-zero when an interaction exists.

### The SGPL Solution

The **Sparse-Group Pliable Lasso (SGPL)** introduces a unified, two-level hierarchical regularisation mechanism that simultaneously promotes:
- **Group-level selection:** Evaluates entire groups of main predictors $X$ and modifiers $Z$.
- **Within-group predictor sparsity:** Selects individual predictors within active groups via an element-wise $\ell_1$ penalty on main effects $\boldsymbol{\beta}$.
- **Predictor-level hierarchical coupling:** Enforces per-predictor hierarchy ($\boldsymbol{\theta}_j \neq \mathbf{0} \implies \beta_j \neq 0$) so interactions only enter when their corresponding main effect is active.
- **Within-block interaction sparsity:** Allows fine-grained selection of individual modifier interactions via an element-wise $\ell_1$ penalty on interactions $\mathbf{\Theta}$.

---

## Supported models 

- **Linear / Gaussian Regression** (Squared-error loss)
- **Logistic Regression** (Binary response, log-loss)
- **Cox Proportional Hazards Model** (Survival analysis, negative partial log-likelihood) - To-be-implemented

---
## Installation

You can install the development version of `SGPL` directly from GitHub:

```r
install.packages("devtools")
devtools::install_github("edelweiss611428/SGPL")
```

Then load the package

```r
library(SGPL)
```



