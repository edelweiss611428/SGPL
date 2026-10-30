#' Sparse Group Pliable Lasso
#'
#' Fit a sparse group pliable lasso (SGPL) model using block-wise
#' coordinate descent with exact block proximal updates (Algorithm 1 of the
#' SGPL paper) for Gaussian, logistic (binomial) or Cox proportional hazards
#' models.
#'
#' Each predictor group is updated by proximal-gradient steps. The proximal
#' operator of the full block penalty is computed exactly by a Dykstra-like
#' proximal algorithm, and the step size is chosen by backtracking with a
#' sufficient-decrease test; a step that fails the test is never accepted.
#' Groups that satisfy the safe screening rule are set to zero without inner
#' iterations. For \code{family = "gaussian"}, the unpenalized coefficients
#' \code{(beta0, theta0)} are profiled out and recovered by least squares at
#' the end. For \code{"binomial"} and \code{"cox"}, they form an unpenalized
#' block that is updated once per outer iteration. For \code{"cox"}, the
#' intercept \code{beta0} is absorbed into the baseline hazard and is returned
#' as 0, while \code{theta0} is estimated.
#'
#' The penalty is
#' \deqn{\lambda_1\Big\{\sum_j \|(\beta_j,\theta_j)\|_2
#'   + \sum_l \sqrt{p_l}\,\|(\beta_l,\Theta_l)\|_2
#'   + \sum_l \sum_g w_{lg} \|\theta_{lg}\|_2\Big\}
#'   + \lambda_2\Big\{\|\beta\|_1 + \sum_j \|\theta_j\|_1\Big\},}
#' with \eqn{w_{lg} = \sqrt{p_l}\sqrt{p_g}/\sqrt{1+K}}, where \eqn{p_l} and
#' \eqn{p_g} are the sizes of predictor group \eqn{l} and modifier group \eqn{g}.
#'
#' @param X Numeric predictor matrix of dimension \eqn{n \times p}. No default.
#' @param Z Numeric modifying matrix of dimension \eqn{n \times K}. No default.
#' @param y Numeric response vector of length \eqn{n}. For \code{family = "binomial"},
#' this should be a vector containing 0s and 1s. For \code{family = "cox"},
#' this is the vector of observed (positive) times. No default.
#' @param family Model specification; \code{"gaussian"}, \code{"binomial"} or \code{"cox"}. Default: \code{"gaussian"}.
#' @param lambda Regularisation parameter. Default: \code{0.1}.
#' @param alpha Mixing parameter between group and lasso penalties.
#' Must lie in \code{[0, 1]}; \eqn{\lambda_1 = \lambda(1-\alpha)} and
#' \eqn{\lambda_2 = \lambda\alpha}. The convergence guarantee of the paper
#' (Proposition 3) requires \eqn{\alpha < 1}. Default: \code{0.5}.
#' @param groups_x Optional predictor group labels (positive integers, not
#' necessarily consecutive). Default: \code{NULL} (each predictor in its own group).
#' @param groups_z Optional modifier group labels (positive integers, not
#' necessarily consecutive). Default: \code{NULL} (each modifier in its own group).
#' @param max_iter_out Maximum outer iterations. Default: \code{200}.
#' @param max_iter_in Maximum inner (proximal-gradient) iterations per block visit. Default: \code{50}.
#' @param tol_out Outer convergence tolerance. Default: \code{1e-6}.
#' @param tol_in Inner convergence tolerance. Default: \code{1e-6}.
#' @param t_init Optional initial step size. Default: \code{NULL} (uses \eqn{1/L_l}, where
#' \eqn{L_l} is the Lipschitz constant of the block gradient).
#' @param bt_factor Backtracking shrinkage factor. Default: \code{0.5}.
#' @param bt_max Maximum backtracking iterations. If no trial step passes the
#' sufficient-decrease test, the step is rejected. Default: \code{50}.
#' @param use_screen Logical; use the safe KKT screening rule. Default: \code{TRUE}.
#' @param verbose Logical; print optimisation progress. Default: \code{TRUE}.
#' @param beta0_init Numeric scalar. Optional warm-start main intercept
#' (binomial only; ignored for gaussian and cox). Default: \code{NULL} (log-odds of the mean of \code{y}).
#' @param theta0_init Numeric vector of length \eqn{K}. Optional warm-start
#' modifier intercept vector (binomial and cox; ignored for gaussian). Default: \code{NULL} (zeros).
#' @param beta_init Numeric vector. Optional warm-start main effects. Default: \code{NULL} (zeros).
#' @param Theta_init Numeric matrix. Optional warm-start interaction effects. Default: \code{NULL} (zeros).
#' @param cached Logical; whether or not to use the cached implementation
#' (faster). \code{FALSE} uses an uncached reference implementation of the
#' same algorithm. Default: \code{TRUE}.
#' @param status Event indicator (1 = event, 0 = censored); required for
#' \code{family = "cox"} and ignored otherwise. Default: \code{NULL}.
#' @return A list containing estimated coefficients and optimization
#' diagnostics.
#'
#' @examples
#' set.seed(1)
#' n = 50; p = 4; K = 2
#' X = matrix(rnorm(n * p), n, p)
#' Z = matrix(rnorm(n * K), n, K)
#' y = X[, 1] * (1 + Z[, 1]) + rnorm(n)
#' fit = SGPL_fit(X, Z, y, lambda = 0.05, groups_x = c(1, 1, 2, 2), verbose = FALSE)
#' fit$beta
#' @export
SGPL_fit = function(X, Z, y,
                    family = c("gaussian", "binomial", "cox"),
                    lambda = 0.1, alpha = 0.5,
                    groups_x = NULL, groups_z = NULL,
                    max_iter_out = 200, max_iter_in = 50,
                    tol_out = 1e-6, tol_in = 1e-6,
                    t_init = NULL, bt_factor = 0.5, bt_max = 50,
                    use_screen = TRUE, verbose = TRUE,
                    beta0_init = NULL, theta0_init = NULL,
                    beta_init = NULL, Theta_init = NULL, cached = TRUE,
                    status = NULL) {

  # family validation
  family = match.arg(family)

  # data matrix validation
  if (!is.matrix(X) || !is.numeric(X))
    stop("X must be a numeric matrix.")

  if (!is.matrix(Z) || !is.numeric(Z))
    stop("Z must be a numeric matrix.")

  if (!is.numeric(y))
    stop("y must be numeric.")

  n = nrow(X)
  p = ncol(X)
  K = ncol(Z)

  if (nrow(Z) != n)
    stop("X and Z must have same number of rows.")

  if (length(y) != n)
    stop("length(y) must equal nrow(X).")

  if (!all(is.finite(X)) || !all(is.finite(Z)) || !all(is.finite(y)))
    stop("X, Z and y must not contain missing or infinite values.")

  # tuning parameter validation
  if (!is.numeric(lambda) || length(lambda) != 1 || is.na(lambda) || lambda < 0)
    stop("lambda must be a single non-negative number.")

  if (!is.numeric(alpha) || length(alpha) != 1 || is.na(alpha) || alpha < 0 || alpha > 1)
    stop("alpha must lie in [0, 1].")

  # extra response check for binomial model
  if (family == "binomial") {
    if (!all(y %in% c(0, 1))) {
      stop("For binomial family, y must only contain 0s and 1s.")
    }
  }

  # extra response checks for cox model
  if (family == "cox") {
    if (is.null(status))
      stop("For cox family, status must be supplied.")
    if (length(status) != n)
      stop("length(status) must equal nrow(X).")
    if (!all(status %in% c(0, 1)))
      stop("status must only contain 0s and 1s.")
    if (sum(status) == 0)
      stop("At least one event (status = 1) is required.")
    if (any(y <= 0))
      stop("For cox family, y must contain positive times.")
    status = as.numeric(status)
  } else {
    status = NULL
  }

  # group validation
  if (is.null(groups_x))
    groups_x = seq_len(p)

  if (is.null(groups_z))
    groups_z = seq_len(K)

  # labels must be positive integers but need not be consecutive (1, 2, 4 is allowed)
  if (!is.numeric(groups_x) || length(groups_x) != p || anyNA(groups_x) ||
      any(groups_x < 1) || any(groups_x != round(groups_x)))
    stop("groups_x must be a vector of ncol(X) positive integers.")

  if (!is.numeric(groups_z) || length(groups_z) != K || anyNA(groups_z) ||
      any(groups_z < 1) || any(groups_z != round(groups_z)))
    stop("groups_z must be a vector of ncol(Z) positive integers.")

  groups_x = as.integer(groups_x)
  groups_z = as.integer(groups_z)

  if(!cached){
    # route to the appropriate rcpp solver
    fit = sgpl_fit_cpp(
      X = X, Z = Z, y = as.numeric(y),
      lambda = lambda, alpha = alpha,
      groups_x_r = groups_x, groups_z_r = groups_z,
      max_iter_out = max_iter_out,
      max_iter_in = max_iter_in,
      tol_out = tol_out,
      tol_in = tol_in,
      t_init_r = t_init,
      bt_factor = bt_factor,
      bt_max = bt_max,
      use_screen = use_screen,
      verbose = verbose,
      beta_init_r = beta_init,
      Theta_init_r = Theta_init,
      beta0_init_r = beta0_init,
      theta0_init_r = theta0_init,
      family = family,
      status_r = status
    )
  } else{
    # route to the appropriate rcpp solver
    fit = fast_sgpl_fit_cpp6(
      X = X, Z = Z, y = as.numeric(y),
      lambda = lambda, alpha = alpha,
      groups_x_r = groups_x, groups_z_r = groups_z,
      max_iter_out = max_iter_out,
      max_iter_in = max_iter_in,
      tol_out = tol_out,
      tol_in = tol_in,
      t_init_r = t_init,
      bt_factor = bt_factor,
      bt_max = bt_max,
      use_screen = use_screen,
      verbose = verbose,
      beta_init_r = beta_init,
      Theta_init_r = Theta_init,
      beta0_init_r = beta0_init,
      theta0_init_r = theta0_init,
      family = family,
      status_r = status
    )

  }

  fit$beta = as.numeric(fit$beta)
  fit$theta0 = as.numeric(fit$theta0)
  fit$obj_path = as.numeric(fit$obj_path)
  names(fit$beta) = colnames(X)
  names(fit$theta0) = colnames(Z)
  rownames(fit$Theta) = colnames(Z)
  colnames(fit$Theta) = colnames(X)

  # preserve the structural family metadata in the returned object
  fit$family = family
  class(fit) = "sgpl"

  fit
}
#' Predict from Sparse Group Pliable Lasso
#'
#' Generate predictions (linear predictor) from an SGPL model. For
#' \code{family = "binomial"} this is the log-odds; for \code{family = "cox"}
#' it is the log relative hazard (use \code{beta0 = 0}).
#'
#' @param X Predictor matrix. No default.
#' @param Z Modifying matrix. No default.
#' @param beta Main effect coefficients. No default.
#' @param Theta Interaction coefficient matrix. No default.
#' @param beta0 Numeric scalar; overall intercept. Default: \code{0}.
#' @param theta0 Numeric vector of length \eqn{K}; modifier intercept vector. Default: \code{NULL} (zeros).
#' @return Numeric prediction vector.
#'
#' @examples
#' set.seed(1)
#' n = 50; p = 4; K = 2
#' X = matrix(rnorm(n * p), n, p)
#' Z = matrix(rnorm(n * K), n, K)
#' y = X[, 1] * (1 + Z[, 1]) + rnorm(n)
#' fit = SGPL_fit(X, Z, y, lambda = 0.05, groups_x = c(1, 1, 2, 2), verbose = FALSE)
#' head(SGPL_predict(X, Z, fit$beta, fit$Theta, fit$beta0, fit$theta0))
#' @export
SGPL_predict = function(X, Z, beta, Theta, beta0 = 0, theta0 = NULL) {

  n = nrow(X)
  p = ncol(X)
  out = numeric(n)


  K = ncol(Z)
  if (is.null(theta0)) {
    theta0 = numeric(K)
  }

  # dimension validation
  if (nrow(Z) != n || length(beta) != p || !identical(dim(Theta), c(K, p)) || length(theta0) != K)
    stop("Dimensions of X, Z, beta, Theta and theta0 do not match.")

  eta = as.numeric(beta0 + Z %*% theta0 + rowSums(X * (rep(1, n) %*% t(beta) + Z %*% Theta)))

  setNames(as.numeric(eta), rownames(X))
}

#' Compute Maximum Lambda for SGPL
#'
#' Compute a value of lambda at which the zero solution (all penalized
#' coefficients equal to zero) is optimal. The unpenalized coefficients are
#' first estimated from the null model (least squares, logistic regression or
#' Cox regression on \code{Z}). Let \eqn{g_j = (x_j^\top r_0,
#' \{Z^\top(x_j \odot r_0)\}^\top)^\top / n} be the null score of predictor
#' \eqn{j}, where \eqn{r_0} is the null residual (Gaussian), \eqn{y - \hat\mu_0}
#' (binomial) or the null martingale residual (Cox).
#'
#' For \eqn{\alpha < 1}:
#' \itemize{
#'   \item \code{"gaussian"} uses the conservative starting value of Section 4.3
#'   of the SGPL paper, \eqn{\max_j \|g_j\|_2 / (1-\alpha)};
#'   \item \code{"binomial"} and \code{"cox"} use the rule of Supplementary
#'   Section B, the smallest \eqn{\lambda} with
#'   \eqn{\|S_{\alpha\lambda}(g_j)\|_2 \le (1-\alpha)\lambda} for all \eqn{j},
#'   found by bisection. It equals the Gaussian value when \eqn{\alpha = 0}
#'   and is smaller otherwise.
#' }
#' Both values are sufficient for the zero solution, but need not be the
#' smallest such value. For \eqn{\alpha = 1} (lasso on the augmented design)
#' the value is the largest absolute null score.
#'
#' @param X Predictor matrix. No default.
#' @param Z Modifier matrix. No default.
#' @param y Response vector (observed times for \code{family = "cox"}). No default.
#' @param groups_x Predictor groups. No default.
#' @param groups_z Modifier groups. No default.
#' @param alpha Mixing parameter. Default: \code{0.5}.
#' @param family Family; \code{"gaussian"}, \code{"binomial"} or \code{"cox"}. Default: \code{"gaussian"}.
#' @param status Event indicator; required for \code{family = "cox"}. Default: \code{NULL}.
#' @return Numeric lambda maximum.
#' @examples
#' set.seed(1)
#' n = 50; p = 4; K = 2
#' X = matrix(rnorm(n * p), n, p)
#' Z = matrix(rnorm(n * K), n, K)
#' y = X[, 1] * (1 + Z[, 1]) + rnorm(n)
#' compute_lambda_max(X, Z, y, groups_x = c(1, 1, 2, 2), groups_z = c(1, 2))
#' @export
#'

compute_lambda_max <- function(X, Z, y, groups_x, groups_z, alpha = 0.5,
                               family = c("gaussian", "binomial", "cox"),
                               status = NULL) {
  # family validation
  family <- match.arg(family)

  # data matrix validation
  if (!is.matrix(X) || !is.numeric(X) || !is.matrix(Z) || !is.numeric(Z) || !is.numeric(y))
    stop("X and Z must be numeric matrices and y must be numeric.")

  if (!all(is.finite(X)) || !all(is.finite(Z)) || !all(is.finite(y)))
    stop("X, Z and y must not contain missing or infinite values.")

  # tuning parameter validation
  if (!is.numeric(alpha) || length(alpha) != 1 || is.na(alpha) || alpha < 0 || alpha > 1)
    stop("alpha must lie in [0, 1].")

  # response validation
  if (family == "binomial" && !all(y %in% c(0, 1)))
    stop("For binomial family, y must only contain 0s and 1s.")

  if (family == "cox" && is.null(status))
    stop("For cox family, status must be supplied.")

  compute_lambda_max_cpp(X, Z, as.numeric(y), alpha, family,
                         if (family == "cox") as.numeric(status) else NULL)
}

#' Cross-Validation for SGPL
#'
#' Perform K-fold cross-validation for sparse group pliable lasso
#' models using warm starts along the lambda path. Supports Gaussian,
#' binomial and Cox families. The CV error is the mean squared error
#' (gaussian), the binomial deviance (binomial) or the cross-validated
#' partial-likelihood deviance of Verweij and van Houwelingen (1993) (cox).
#'
#' @param X Predictor matrix. No default.
#' @param Z Modifier matrix. No default.
#' @param y Response vector (observed times for \code{family = "cox"}). No default.
#' @param family Model specification; \code{"gaussian"}, \code{"binomial"} or \code{"cox"}. Default: \code{"gaussian"}.
#' @param lambda_seq Optional lambda sequence. Default: \code{NULL} (25 log-spaced values
#' from \code{compute_lambda_max()} down to 0.005 times that value).
#' @param alpha Mixing parameter. Default: \code{0.5}.
#' @param groups_x Predictor groups. Default: \code{NULL} (each predictor in its own group).
#' @param groups_z Modifier groups. Default: \code{NULL} (each modifier in its own group).
#' @param nfolds Number of folds. Default: \code{5}.
#' @param max_iter_out Maximum outer iterations. Default: \code{200}.
#' @param max_iter_in Maximum inner iterations. Default: \code{50}.
#' @param tol_out Outer convergence tolerance. Default: \code{1e-5}.
#' @param tol_in Inner convergence tolerance. Default: \code{1e-5}.
#' @param cached Logical; whether or not to use the cached implementation (faster). Default: \code{TRUE}.
#' @param warmup Logical; whether or not to use warm starts along the lambda path. Default: \code{TRUE}.
#' @param status Event indicator; required for \code{family = "cox"}. Default: \code{NULL}.
#' @return Cross-validation results and selected lambdas.
#'
#' @examples
#' set.seed(1)
#' n = 50; p = 4; K = 2
#' X = matrix(rnorm(n * p), n, p)
#' Z = matrix(rnorm(n * K), n, K)
#' y = X[, 1] * (1 + Z[, 1]) + rnorm(n)
#' cv = SGPL_CV(X, Z, y, groups_x = c(1, 1, 2, 2), nfolds = 3)
#' cv$lambda_min
#' @export
SGPL_CV = function(X, Z, y,
                   family = c("gaussian", "binomial", "cox"),
                   lambda_seq = NULL, alpha = 0.5,
                   groups_x = NULL, groups_z = NULL,
                   nfolds = 5, max_iter_out = 200, max_iter_in = 50,
                   tol_out = 1e-5, tol_in = 1e-5, cached = TRUE, warmup = TRUE,
                   status = NULL) {

  # family validation
  family = match.arg(family)

  # data matrix validation
  if (!is.matrix(X) || !is.matrix(Z))
    stop("X and Z must be matrices.")

  n = nrow(X)

  # response validation
  if (family == "cox" && is.null(status))
    stop("For cox family, status must be supplied.")

  if (is.null(lambda_seq)) {
    lam_max = compute_lambda_max(X, Z, y, groups_x, groups_z, alpha, family, status)
    # Section 4.3: 25 log-spaced values from lambda_max to 0.005 * lambda_max
    lambda_seq = exp(seq(log(lam_max), log(lam_max * 0.005), length.out = 25))
  }

  # lambda_seq and nfolds validation
  if (!is.numeric(lambda_seq) || length(lambda_seq) < 1 || anyNA(lambda_seq) || any(lambda_seq < 0))
    stop("lambda_seq must be a vector of non-negative numbers.")

  if (!is.numeric(nfolds) || length(nfolds) != 1 || is.na(nfolds) || nfolds < 2 || nfolds > n || nfolds != round(nfolds))
    stop("nfolds must be an integer between 2 and nrow(X).")

  # The one-standard-error rule below assumes a decreasing sequence
  lambda_seq = sort(lambda_seq, decreasing = TRUE)

  nL = length(lambda_seq)

  # Stratify folds (binomial: by class; cox: by event status)
  strat = if (family == "binomial") y else if (family == "cox") status else NULL
  if (!is.null(strat)) {
    fids = integer(n)
    idx0 = which(strat == 0)
    idx1 = which(strat == 1)
    fids[idx0] = sample(rep(seq_len(nfolds), length.out = length(idx0)))
    fids[idx1] = sample(rep(seq_len(nfolds), length.out = length(idx1)))
  } else {
    fids = sample(rep(seq_len(nfolds), length.out = n))
  }

  cv_err = matrix(NA_real_, nfolds, nL)

  for (f in seq_len(nfolds)) {
    cat(sprintf("CV fold %d / %d\n", f, nfolds))

    tr = which(fids != f)
    te = which(fids == f)

    warm_beta0 = NULL
    warm_theta0 = NULL
    warm_beta = NULL
    warm_Theta = NULL

    for (li in seq_len(nL)) {
      fit = SGPL_fit(
        X = X[tr, , drop = FALSE],
        Z = Z[tr, , drop = FALSE],
        y = y[tr],
        family = family,
        lambda = lambda_seq[li],
        alpha = alpha,
        groups_x = groups_x,
        groups_z = groups_z,
        max_iter_out = max_iter_out,
        max_iter_in = max_iter_in,
        tol_out = tol_out,
        tol_in = tol_in,
        beta_init = warm_beta,
        Theta_init = warm_Theta,
        beta0_init = warm_beta0,
        theta0_init = warm_theta0,
        verbose = FALSE, cached = cached,
        status = if (family == "cox") status[tr] else NULL
      )

      if (family == "gaussian") {
        eta = SGPL_predict(X[te, , drop = FALSE], Z[te, , drop = FALSE], fit$beta, fit$Theta,
                           beta0 = fit$beta0, theta0 = fit$theta0)
        # MSE
        cv_err[f, li] = mean((y[te] - eta)^2)
      } else if (family == "binomial") {
        eta = SGPL_predict(X[te, , drop = FALSE], Z[te, , drop = FALSE], fit$beta, fit$Theta,
                           beta0 = fit$beta0, theta0 = fit$theta0)
        # Binomial Deviance loss: -2 * log-likelihood / n_test
        # Written to avoid log(0) and keep calculation numerically stable
        logloss = numeric(length(te))
        pos_idx = eta > 0

        logloss[pos_idx]  = (1 - y[te][pos_idx]) * eta[pos_idx] + log1p(exp(-eta[pos_idx]))
        logloss[!pos_idx] = -y[te][!pos_idx] * eta[!pos_idx] + log1p(exp(eta[!pos_idx]))

        cv_err[f, li] = 2 * mean(logloss)
      } else {
        # Cross-validated partial likelihood (Verweij and van Houwelingen, 1993):
        # deviance = 2 * { nlpl_full(eta_all) - nlpl_train(eta_train) },
        # where nlpl is the negative log partial likelihood.
        eta_all = SGPL_predict(X, Z, fit$beta, fit$Theta, beta0 = 0, theta0 = fit$theta0)
        nlpl_full = cox_loss_cpp(eta_all, as.numeric(y), as.numeric(status))
        nlpl_tr = cox_loss_cpp(eta_all[tr], as.numeric(y[tr]), as.numeric(status[tr]))
        cv_err[f, li] = 2 * (nlpl_full - nlpl_tr)
      }

      if(warmup){
        warm_beta0 = fit$beta0
        warm_theta0 = fit$theta0
        warm_beta = fit$beta
        warm_Theta = fit$Theta
      } else{
        warm_beta0 = NULL
        warm_theta0 = NULL
        warm_beta = NULL
        warm_Theta = NULL
      }
    }
  }

  cv_mean = colMeans(cv_err)
  cv_se = apply(cv_err, 2, sd) / sqrt(nfolds)

  best_idx = which.min(cv_mean)

  # Find 1-standard-error rule lambda (largest lambda within one SE of the minimum)
  idx_1se = which(cv_mean <= cv_mean[best_idx] + cv_se[best_idx])[1]

  out = list(
    lambda_seq = lambda_seq,
    cv_mean = cv_mean,
    cv_se = cv_se,
    lambda_min = lambda_seq[best_idx],
    lambda_1se = lambda_seq[idx_1se],
    best_idx = best_idx,
    idx_1se = idx_1se,
    cv_error = cv_err,
    foldid = fids,
    family = family
  )

  class(out) = "sgpl_cv"
  out
}
