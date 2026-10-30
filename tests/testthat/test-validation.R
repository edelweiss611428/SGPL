#Input validation

# ========================================================
#                   Simulated datasets
# ========================================================

set.seed(12345)
n = 40; p = 6; K = 3
X = matrix(rnorm(n * p), n)
Z = matrix(rnorm(n * K), n)
y = rnorm(n)
yb = rbinom(n, 1, 0.5)
yc = rexp(n)
st = rbinom(n, 1, 0.7); st[1] = 1

fit = function(...) SGPL_fit(verbose = FALSE, max_iter_out = 5, ...)


test_that("SGPL_fit accepts valid inputs", {

  expect_no_error(fit(X, Z, y))
  expect_no_error(fit(X, Z, yb, family = "binomial"))
  expect_no_error(fit(X, Z, yc, family = "cox", status = st))
  expect_no_error(fit(X, Z, y, alpha = 0))
  expect_no_error(fit(X, Z, y, alpha = 1))
  expect_no_error(fit(X, Z, y, lambda = 0))

})

test_that("SGPL_fit validates data", {

  expect_error(fit(as.data.frame(X), Z, y), regexp = "X must be a numeric matrix")
  expect_error(fit(X, as.data.frame(Z), y), regexp = "Z must be a numeric matrix")
  expect_error(fit(matrix("a", n, p), Z, y), regexp = "X must be a numeric matrix")
  expect_error(fit(X, Z, as.character(y)), regexp = "y must be numeric")
  expect_error(fit(X, Z[-1, ], y), regexp = "same number of rows")
  expect_error(fit(X, Z, y[-1]), regexp = "length\\(y\\) must equal nrow\\(X\\)")

  X_na = X; X_na[1, 1] = NA
  Z_inf = Z; Z_inf[1, 1] = Inf
  y_na = y; y_na[1] = NA
  expect_error(fit(X_na, Z, y), regexp = "missing or infinite")
  expect_error(fit(X, Z_inf, y), regexp = "missing or infinite")
  expect_error(fit(X, Z, y_na), regexp = "missing or infinite")

})

test_that("SGPL_fit validates tuning parameters", {

  expect_error(fit(X, Z, y, lambda = -1), regexp = "lambda must be")
  expect_error(fit(X, Z, y, lambda = NA), regexp = "lambda must be")
  expect_error(fit(X, Z, y, lambda = c(0.1, 0.2)), regexp = "lambda must be")
  expect_error(fit(X, Z, y, lambda = "0.1"), regexp = "lambda must be")

  expect_error(fit(X, Z, y, alpha = -0.1), regexp = "alpha must lie in \\[0, 1\\]")
  expect_error(fit(X, Z, y, alpha = 1.1), regexp = "alpha must lie in \\[0, 1\\]")
  expect_error(fit(X, Z, y, alpha = NA), regexp = "alpha must lie in \\[0, 1\\]")
  expect_error(fit(X, Z, y, alpha = c(0.2, 0.5)), regexp = "alpha must lie in \\[0, 1\\]")

  expect_error(fit(X, Z, y, family = "poisson"))

})

test_that("SGPL_fit validates the response", {

  expect_error(fit(X, Z, y, family = "binomial"), regexp = "only contain 0s and 1s")

  expect_error(fit(X, Z, yc, family = "cox"), regexp = "status must be supplied")
  expect_error(fit(X, Z, yc, family = "cox", status = st[-1]), regexp = "length\\(status\\)")
  expect_error(fit(X, Z, yc, family = "cox", status = st + 1), regexp = "only contain 0s and 1s")
  expect_error(fit(X, Z, yc, family = "cox", status = numeric(n)), regexp = "At least one event")
  expect_error(fit(X, Z, -yc, family = "cox", status = st), regexp = "positive times")

})

test_that("SGPL_fit validates groups; labels need not be consecutive", {

  gx = c(1, 1, 2, 2, 3, 3); gz = c(1, 1, 2)
  f1 = fit(X, Z, y, groups_x = gx, groups_z = gz)
  f2 = fit(X, Z, y, groups_x = c(1, 1, 2, 2, 5, 5), groups_z = c(1, 1, 4))
  expect_equal(f1$beta, f2$beta)
  expect_equal(f1$Theta, f2$Theta)

  expect_error(fit(X, Z, y, groups_x = gx[-1]), regexp = "groups_x must be")
  expect_error(fit(X, Z, y, groups_x = c(0, gx[-1])), regexp = "groups_x must be")
  expect_error(fit(X, Z, y, groups_x = c(-1, gx[-1])), regexp = "groups_x must be")
  expect_error(fit(X, Z, y, groups_x = c(1.5, gx[-1])), regexp = "groups_x must be")
  expect_error(fit(X, Z, y, groups_x = c(NA, gx[-1])), regexp = "groups_x must be")
  expect_error(fit(X, Z, y, groups_x = letters[gx]), regexp = "groups_x must be")

  expect_error(fit(X, Z, y, groups_z = gz[-1]), regexp = "groups_z must be")
  expect_error(fit(X, Z, y, groups_z = c(0, 1, 1)), regexp = "groups_z must be")
  expect_error(fit(X, Z, y, groups_z = c(1, 1, 2.5)), regexp = "groups_z must be")
  expect_error(fit(X, Z, y, groups_z = c("a", "a", "b")), regexp = "groups_z must be")

})

test_that("SGPL_fit validates warm starts", {

  expect_error(fit(X, Z, y, beta_init = numeric(p - 1)), regexp = "beta_init has incorrect length")
  expect_error(fit(X, Z, y, Theta_init = matrix(0, K, p - 1)), regexp = "Theta_init has incorrect dimensions")

})

test_that("SGPL_fit inherits names from X and Z", {

  Xn = X; colnames(Xn) = paste0("x", 1:p)
  Zn = Z; colnames(Zn) = paste0("z", 1:K)
  f = fit(Xn, Zn, y)
  expect_equal(names(f$beta), colnames(Xn))
  expect_equal(names(f$theta0), colnames(Zn))
  expect_equal(dimnames(f$Theta), list(colnames(Zn), colnames(Xn)))

  f0 = fit(X, Z, y)
  expect_null(names(f0$beta))
  expect_null(dimnames(f0$Theta))

})

test_that("SGPL_predict validates dimensions", {

  f = fit(X, Z, y)
  expect_no_error(SGPL_predict(X, Z, f$beta, f$Theta, f$beta0, f$theta0))
  expect_error(SGPL_predict(X, Z[-1, ], f$beta, f$Theta), regexp = "Dimensions")
  expect_error(SGPL_predict(X, Z, f$beta[-1], f$Theta), regexp = "Dimensions")
  expect_error(SGPL_predict(X, Z, f$beta, f$Theta[-1, ]), regexp = "Dimensions")
  expect_error(SGPL_predict(X, Z, f$beta, f$Theta, theta0 = numeric(K + 1)), regexp = "Dimensions")

  Xn = X; rownames(Xn) = paste0("obs", 1:n)
  expect_equal(names(SGPL_predict(Xn, Z, f$beta, f$Theta)), rownames(Xn))

})

test_that("compute_lambda_max validates inputs", {

  expect_no_error(compute_lambda_max(X, Z, y, NULL, NULL))
  expect_no_error(compute_lambda_max(X, Z, yb, NULL, NULL, family = "binomial"))
  expect_no_error(compute_lambda_max(X, Z, yc, NULL, NULL, family = "cox", status = st))

  X_na = X; X_na[1, 1] = NA
  expect_error(compute_lambda_max(as.data.frame(X), Z, y, NULL, NULL), regexp = "numeric matrices")
  expect_error(compute_lambda_max(X_na, Z, y, NULL, NULL), regexp = "missing or infinite")
  expect_error(compute_lambda_max(X, Z[-1, ], y, NULL, NULL), regexp = "same number of rows")
  expect_error(compute_lambda_max(X, Z, y, NULL, NULL, alpha = 2), regexp = "alpha must lie")
  expect_error(compute_lambda_max(X, Z, y, NULL, NULL, family = "binomial"), regexp = "only contain 0s and 1s")
  expect_error(compute_lambda_max(X, Z, yc, NULL, NULL, family = "cox"), regexp = "status must be supplied")

})

test_that("compute_lambda_max gives the zero solution", {

  for (fam in c("gaussian", "binomial", "cox")) {
    yy = switch(fam, gaussian = y, binomial = yb, cox = yc)
    ss = if (fam == "cox") st else NULL
    lmax = compute_lambda_max(X, Z, yy, NULL, NULL, family = fam, status = ss)
    f = SGPL_fit(X, Z, yy, family = fam, lambda = lmax, verbose = FALSE, status = ss)
    expect_true(all(f$beta == 0) && all(f$Theta == 0))
  }

})

test_that("SGPL_CV validates inputs", {

  expect_error(SGPL_CV(as.data.frame(X), Z, y), regexp = "X and Z must be matrices")
  expect_error(SGPL_CV(X, Z, yc, family = "cox"), regexp = "status must be supplied")
  expect_error(SGPL_CV(X, Z, y, lambda_seq = c(0.1, -0.1)), regexp = "lambda_seq must be")
  expect_error(SGPL_CV(X, Z, y, lambda_seq = c(0.1, NA)), regexp = "lambda_seq must be")
  expect_error(SGPL_CV(X, Z, y, lambda_seq = 0.1, nfolds = 1), regexp = "nfolds must be")
  expect_error(SGPL_CV(X, Z, y, lambda_seq = 0.1, nfolds = n + 1), regexp = "nfolds must be")
  expect_error(SGPL_CV(X, Z, y, lambda_seq = 0.1, nfolds = 2.5), regexp = "nfolds must be")

  expect_no_error(capture.output(SGPL_CV(X, Z, y, lambda_seq = c(0.1, 0.05), nfolds = 3, max_iter_out = 5)))

})
