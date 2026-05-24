#' Sparse Group Pliable Lasso
#'
#' Fit a sparse group pliable lasso (SGPL) model using block-wise
#' coordinate-descent with nested proximal updates
#'
#' @param X Numeric predictor matrix of dimension \eqn{n \times p}.
#' @param Z Numeric modifying matrix of dimension \eqn{n \times K}.
#' @param y Numeric response vector of length \eqn{n}.
#' @param lambda Regularisation parameter.
#' @param alpha Mixing parameter between group and lasso penalties.
#' Must lie in \code{[0, 1]}.
#' @param groups_x Optional predictor group labels.
#' Default assigns each predictor to its own group.
#' @param groups_z Optional modifier group labels.
#' Default assigns each modifier to its own group.
#' @param max_iter_out Maximum outer iterations.
#' @param max_iter_in Maximum inner iterations.
#' @param tol_out Outer convergence tolerance.
#' @param tol_in Inner convergence tolerance.
#' @param t_init Optional initial step size.
#' @param bt_factor Backtracking shrinkage factor.
#' @param bt_max Maximum backtracking iterations.
#' @param use_screen Logical; use KKT screening rules.
#' @param verbose Logical; print optimisation progress.
#' @param beta_init Numeric vector. Optional warm-start main effects.
#' @param Theta_init Numeric matrix. Optional warm-start interaction effects.
#' @return A list containing estimated coefficients and optimization
#' diagnostics.
#'
#' @export
SGPL_fit = function(X, Z, y,
                    lambda = 0.1, alpha = 0.5,
                    groups_x = NULL, groups_z = NULL,
                    max_iter_out = 200, max_iter_in = 50,
                    tol_out = 1e-6, tol_in = 1e-6,
                    t_init = NULL,bt_factor = 0.5, bt_max = 50,
                    use_screen = TRUE, verbose = TRUE,
                    beta_init = NULL, Theta_init = NULL) {

  #data matrix validation
  if (!is.matrix(X))
    stop("X must be a matrix.")

  if (!is.matrix(Z))
    stop("Z must be a matrix.")

  if (!is.numeric(y))
    stop("y must be numeric.")

  n = nrow(X)
  p = ncol(X)
  K = ncol(Z)

  if (nrow(Z) != n)
    stop("X and Z must have same number of rows.")

  if (length(y) != n)
    stop("length(y) must equal nrow(X).")

  # group validation

  if (is.null(groups_x))
    groups_x = seq_len(p)

  if (is.null(groups_z))
    groups_z = seq_len(K)

  groups_x = as.integer(groups_x)
  groups_z = as.integer(groups_z)

  fit = sgpl_fit_rcpp(
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
    Theta_init_r = Theta_init
  )

  class(fit) = "sgpl"

  fit
}

#' Predict from Sparse Group Pliable Lasso
#'
#' Generate predictions from an SGPL model.
#'
#' @param X Predictor matrix.
#' @param Z Modifying matrix.
#' @param beta Main effect coefficients.
#' @param Theta Interaction coefficient matrix.
#'
#' @return Numeric prediction vector.
#'
#' @export
SGPL_predict = function(X, Z, beta, Theta) {
  n = nrow(X)
  p = ncol(X)

  out = numeric(n)

  for (j in seq_len(p)) {
    eta = beta[j] + Z %*% Theta[, j]

    out = out + X[, j] * eta
  }

  as.numeric(out)
}

#' Compute Maximum Lambda for SGPL
#'
#' Compute the smallest value of lambda such that all coefficients
#' are zero.
#'
#' @param X Predictor matrix.
#' @param Z Modifier matrix.
#' @param y Response vector.
#' @param groups_x Predictor groups.
#' @param groups_z Modifier groups.
#' @param alpha Mixing parameter.
#'
#' @return Numeric lambda maximum.
#' @importFrom  stats sd uniroot
#' @export
compute_lambda_max = function(X, Z, y,
                              groups_x = NULL, groups_z = NULL,
                              alpha = 0.5) {
  if (!is.matrix(X))
    stop("X must be a matrix.")

  if (!is.matrix(Z))
    stop("Z must be a matrix.")

  if (!is.numeric(y))
    stop("y must be numeric.")

  n = nrow(X)
  p = ncol(X)
  K = ncol(Z)

  if (length(y) != n)
    stop("length(y) must equal nrow(X).")

  if (nrow(Z) != n)
    stop("nrow(Z) must equal nrow(X).")

  if (is.null(groups_x))
    groups_x = seq_len(p)

  if (is.null(groups_z))
    groups_z = seq_len(K)

  groups_x = as.integer(groups_x)
  groups_z = as.integer(groups_z)

  L = max(groups_x)

  pl = tabulate(groups_x)

  lam_candidates = numeric(L)

  for (l in seq_len(L)) {

    idx_l = which(groups_x == l)
    p_l = pl[l]
    X_l = X[, idx_l, drop = FALSE]
    g_l = as.numeric(crossprod(X_l, y)) / n
    H_l = crossprod(Z, X_l * matrix(y, n, length(idx_l))) / n
    v = c(g_l, as.vector(H_l))

    kkt_fun = function(lambda) {

      lam1 = lambda * (1 - alpha)
      lam2 = lambda * alpha

      sv = sign(v) *pmax(abs(v) - lam2, 0)

      sqrt(sum(sv^2)) - sqrt(p_l) * lam1
    }

    upper =
      max(abs(v)) /
      max(alpha, 1e-8)

    while (kkt_fun(upper) > 0)
      upper = upper * 2

    lam_candidates[l] = uniroot(kkt_fun,
                                lower = 0,
                                upper = upper,
                                tol = 1e-10)$root
  }

  max(lam_candidates)
}

#' Cross-Validation for SGPL
#'
#' Perform K-fold cross-validation for sparse group pliable lasso
#' models using warm starts along the lambda path.
#'
#' @param X Predictor matrix.
#' @param Z Modifier matrix.
#' @param y Response vector.
#' @param lambda_seq Optional lambda sequence.
#' @param alpha Mixing parameter.
#' @param groups_x Predictor groups.
#' @param groups_z Modifier groups.
#' @param nfolds Number of folds.
#' @param max_iter_out Maximum outer iterations.
#' @param max_iter_in Maximum inner iterations.
#' @param tol_out Outer convergence tolerance.
#' @param tol_in Inner convergence tolerance.
#'
#' @return Cross-validation results and selected lambdas.
#'
#' @export
SGPL_CV = function(X,Z, y,
                   lambda_seq = NULL, alpha = 0.5,
                   groups_x = NULL, groups_z = NULL,
                   nfolds = 5, max_iter_out = 200, max_iter_in = 50,
                   tol_out = 1e-5, tol_in = 1e-5) {
  n = nrow(X)

  if (is.null(lambda_seq)) {
    lam_max = compute_lambda_max(X, Z, y, groups_x, groups_z, alpha)

    lambda_seq = exp(seq(log(lam_max), log(lam_max * 0.005), length.out = 30))
  }

  nL = length(lambda_seq)

  fids = sample(rep(seq_len(nfolds), length.out = n))

  cv_err = matrix(NA_real_, nfolds, nL)

  for (f in seq_len(nfolds)) {
    cat(sprintf("CV fold %d / %d\n", f, nfolds))

    tr = which(fids != f)

    te = which(fids == f)

    warm_beta = NULL

    warm_Theta = NULL

    for (li in seq_len(nL)) {
      fit = SGPL_fit(
        X[tr, , drop = FALSE],
        Z[tr, , drop = FALSE],
        y[tr],
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
        verbose = FALSE
      )

      yp = SGPL_predict(X[te, , drop = FALSE], Z[te, , drop = FALSE], fit$beta, fit$Theta)

      cv_err[f, li] =
        mean((y[te] - yp)^2)

      warm_beta = fit$beta

      warm_Theta = fit$Theta
    }
  }

  cv_mean = colMeans(cv_err)

  cv_se =
    apply(cv_err, 2, sd) /
    sqrt(nfolds)

  best_idx = which.min(cv_mean)

  idx_1se = which(cv_mean <=
                    cv_mean[best_idx] +
                    cv_se[best_idx])[1]

  out = list(
    lambda_seq = lambda_seq,
    cv_mean = cv_mean,
    cv_se = cv_se,
    lambda_min = lambda_seq[best_idx],
    lambda_1se = lambda_seq[idx_1se],
    best_idx = best_idx,
    idx_1se = idx_1se,
    cv_error = cv_err
  )

  class(out) = "sgpl_cv"

  out
}
