// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include "sgpl_helpers.h"

using namespace Rcpp;

// lambda_max for SGPL (see compute_lambda_max in R/sgpl.R).
// [[Rcpp::export]]
double compute_lambda_max_cpp(const arma::mat& X, const arma::mat& Z,
                              const arma::vec& y, double alpha,
                              std::string family,
                              Rcpp::Nullable<arma::vec> status_r = R_NilValue) {
  const int n = X.n_rows;
  const int K = Z.n_cols;
  if ((int)Z.n_rows != n || (int)y.n_elem != n)
    Rcpp::stop("X, Z and y must have the same number of rows");

  // Null residual r0
  arma::vec r;
  if (family == "gaussian") {
    arma::mat Q = arma::orth(arma::join_rows(arma::ones<arma::vec>(n), Z));
    r = y - Q * (Q.t() * y);
  } else if (family == "binomial") {
    double p_bar = std::min(std::max(arma::mean(y), 1e-5), 1.0 - 1e-5);
    double beta0 = std::log(p_bar / (1.0 - p_bar));
    arma::vec theta0(K, arma::fill::zeros);
    arma::vec zero_off(n, arma::fill::zeros);
    update_intercepts_logit(Z, y, zero_off, beta0, theta0, 100, 1e-10);
    r = y - 1.0 / (1.0 + arma::exp(-(beta0 + Z * theta0)));
  } else if (family == "cox") {
    if (status_r.isNull()) Rcpp::stop("status must be supplied for family = 'cox'");
    arma::vec status = Rcpp::as<arma::vec>(status_r);
    if ((int)status.n_elem != n) Rcpp::stop("length(status) must equal nrow(X)");
    CoxData C = make_cox_data(y, status);
    arma::vec theta0(K, arma::fill::zeros);
    arma::vec zero_off(n, arma::fill::zeros);
    if (K > 0) update_intercepts_cox(C, Z, zero_off, theta0, 100, 1e-10);
    r = -(double)n * compute_cox_psi(C, Z * theta0);
  } else {
    Rcpp::stop("family must be 'gaussian', 'binomial' or 'cox'");
  }

  // Null scores g_j = (x_j' r, Z'(x_j * r)) / n, columns of a (1 + K) x p matrix
  arma::mat scores = arma::join_cols(r.t() * X, Z.t() * (X.each_col() % r)) / n;
  scores.replace(arma::datum::nan, 0.0);

  if (alpha >= 1.0) return arma::abs(scores).max();

  const double lam_simple = std::sqrt(arma::sum(arma::square(scores), 0).max()) / (1.0 - alpha);
  if (family == "gaussian" || alpha == 0.0 || lam_simple <= 0.0) return lam_simple;

  // Bisection for the root of h(lambda) on [0, lam_simple]
  arma::mat abs_scores = arma::abs(scores);
  auto h = [&](double lam) {
    arma::mat s = arma::clamp(abs_scores - alpha * lam, 0.0, arma::datum::inf);
    return std::sqrt(arma::sum(arma::square(s), 0).max()) - (1.0 - alpha) * lam;
  };
  double lo = 0.0, hi = lam_simple;
  for (int it = 0; it < 200; ++it) {
    double mid = 0.5 * (lo + hi);
    if (h(mid) <= 0.0) hi = mid; else lo = mid;
    if (hi - lo <= 1e-12 * hi) break;
  }
  return hi;
}
