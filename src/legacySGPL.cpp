// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include "sgpl_helpers.h"

using namespace Rcpp;
using namespace arma;

// Uncached reference implementation of Algorithm 1 (exact block proximal
// updates). It follows the same steps as fast_sgpl_fit_cpp6 but recomputes the
// loss from the full coefficient vector at every trial step, so it can be used
// to check the cached solver. It is slower and intended for testing.

// Loss of the SGPL model for a given (beta0, theta0, beta, Theta).
// gaussian: squared-error loss; binomial: logistic loss; cox: Breslow negative
// log partial likelihood divided by n (beta0 is ignored for cox).
inline double sgpl_loss_full(
    const arma::mat& X, const arma::mat& Z, const arma::vec& y,
    const arma::vec& beta, const arma::mat& Theta,
    double beta0, const arma::vec& theta0,
    const std::string& family, const CoxData* coxp
) {
  if (family == "gaussian") {
    return compute_gaussian_loss(y, predict_sgpl_cpp(X, Z, beta, Theta, beta0, theta0));
  } else if (family == "binomial") {
    return compute_logistic_loss(y, predict_sgpl_cpp(X, Z, beta, Theta, beta0, theta0));
  }
  return compute_cox_loss(*coxp, predict_sgpl_cpp(X, Z, beta, Theta, 0.0, theta0));
}

// [[Rcpp::export]]
double objective_sgpl_cpp(
    const arma::mat& X, const arma::mat& Z, const arma::vec& y,
    const arma::vec& beta, const arma::mat& Theta,
    double beta0, const arma::vec& theta0,
    double lambda, double alpha,
    const arma::uvec& groups_x, const arma::uvec& groups_z,
    std::string family,
    Rcpp::Nullable<arma::vec> status_r = R_NilValue
) {
  int K = Z.n_cols;
  CoxData cox;
  if (family == "cox") {
    if (status_r.isNull()) Rcpp::stop("status must be supplied for family = 'cox'");
    cox = make_cox_data(y, Rcpp::as<arma::vec>(status_r));
  } else if (family != "gaussian" && family != "binomial") {
    Rcpp::stop("family must be 'gaussian', 'binomial' or 'cox'");
  }
  double loss = sgpl_loss_full(X, Z, y, beta, Theta, beta0, theta0, family,
                               family == "cox" ? &cox : nullptr);
  double penalty = compute_sgpl_penalties(beta, Theta, lambda, alpha, groups_x, groups_z, K);
  return loss + penalty;
}

// Null Cox fit on Z (no penalized terms): returns theta0, the null martingale
// residuals r0 = -n * psi (used for lambda_max), and the loss.
// [[Rcpp::export]]
Rcpp::List cox_null_fit_cpp(const arma::mat& Z, const arma::vec& time,
                            const arma::vec& status) {
  int n = Z.n_rows;
  if ((int)time.n_elem != n || (int)status.n_elem != n)
    Rcpp::stop("time and status must have length nrow(Z)");
  CoxData C = make_cox_data(time, status);
  arma::vec theta0(Z.n_cols, arma::fill::zeros);
  arma::vec zero_off(n, arma::fill::zeros);
  if (Z.n_cols > 0) update_intercepts_cox(C, Z, zero_off, theta0, 100, 1e-10);
  arma::vec eta = Z * theta0;
  arma::vec r0 = -(double)n * compute_cox_psi(C, eta);
  return Rcpp::List::create(Named("theta0") = theta0,
                            Named("r0") = r0,
                            Named("loss") = compute_cox_loss(C, eta));
}

// Negative log partial likelihood (Breslow), NOT divided by n.
// Used for the cross-validated partial likelihood in SGPL_CV.
// [[Rcpp::export]]
double cox_loss_cpp(const arma::vec& eta, const arma::vec& time,
                    const arma::vec& status) {
  if (time.n_elem != eta.n_elem || status.n_elem != eta.n_elem)
    Rcpp::stop("eta, time and status must have the same length");
  CoxData C = make_cox_data(time, status);
  return compute_cox_loss(C, eta) * eta.n_elem;
}


// [[Rcpp::export]]
Rcpp::List sgpl_fit_cpp(
    const arma::mat& X, const arma::mat& Z, const arma::vec& y,
    double lambda = 0.1, double alpha = 0.5,
    Rcpp::Nullable<arma::uvec> groups_x_r = R_NilValue,
    Rcpp::Nullable<arma::uvec> groups_z_r = R_NilValue,
    int max_iter_out = 200, int max_iter_in = 50,
    double tol_out = 1e-6, double tol_in = 1e-6,
    Rcpp::Nullable<double> t_init_r = R_NilValue,
    double bt_factor = 0.5, int bt_max = 50,
    bool use_screen = true, bool verbose = true,
    Rcpp::Nullable<arma::vec> beta_init_r = R_NilValue,
    Rcpp::Nullable<arma::mat> Theta_init_r = R_NilValue,
    Rcpp::Nullable<double> beta0_init_r = R_NilValue,
    Rcpp::Nullable<arma::vec> theta0_init_r = R_NilValue,
    std::string family = "gaussian",
    Rcpp::Nullable<arma::vec> status_r = R_NilValue,
    double prox_tol = 1e-10, int prox_maxit = 10000
){
  int n = X.n_rows;
  int p = X.n_cols;
  int K = Z.n_cols;

  // Dimension checks
  if ((int)Z.n_rows != n) Rcpp::stop("X and Z must have same number of rows");
  if ((int)y.n_elem != n) Rcpp::stop("length(y) must equal nrow(X)");
  if (family != "gaussian" && family != "binomial" && family != "cox")
    Rcpp::stop("family must be 'gaussian', 'binomial' or 'cox'");
  if (alpha < 0.0 || alpha > 1.0) Rcpp::stop("alpha must lie in [0, 1]");
  const bool is_gauss = (family == "gaussian");

  CoxData cox;
  if (family == "cox") {
    if (status_r.isNull()) Rcpp::stop("status must be supplied for family = 'cox'");
    arma::vec status = Rcpp::as<arma::vec>(status_r);
    if ((int)status.n_elem != n) Rcpp::stop("length(status) must equal nrow(X)");
    cox = make_cox_data(y, status);
  }
  const CoxData* coxp = (family == "cox") ? &cox : nullptr;

  // Groups (if null, each feature forms a group)
  arma::uvec groups_x;
  arma::uvec groups_z;
  if (groups_x_r.isNull()) {
    groups_x = arma::linspace<arma::uvec>(1, p, p);
  } else {
    groups_x = Rcpp::as<arma::uvec>(groups_x_r);
    if ((int)groups_x.n_elem != p) Rcpp::stop("groups_x has incorrect length");
  }
  if (groups_z_r.isNull()) {
    groups_z = arma::linspace<arma::uvec>(1, K, K);
  } else {
    groups_z = Rcpp::as<arma::uvec>(groups_z_r);
    if ((int)groups_z.n_elem != K) Rcpp::stop("groups_z has incorrect length");
  }

  // Initialise penalised params
  arma::vec beta;
  if (beta_init_r.isNull()) {
    beta.zeros(p);
  } else {
    beta = Rcpp::as<arma::vec>(beta_init_r);
    if ((int)beta.n_elem != p) Rcpp::stop("beta_init has incorrect length");
  }
  arma::mat Theta;
  if (Theta_init_r.isNull()) {
    Theta.zeros(K, p);
  } else {
    Theta = Rcpp::as<arma::mat>(Theta_init_r);
    if ((int)Theta.n_rows != K || (int)Theta.n_cols != p) stop("Theta_init has incorrect dimensions");
  }

  // Initialise intercepts (gaussian: profiled out, initial values ignored)
  double beta0 = 0.0;
  arma::vec theta0(K, arma::fill::zeros);
  if (family == "binomial") {
    if (beta0_init_r.isNotNull()) {
      beta0 = Rcpp::as<double>(beta0_init_r);
    } else {
      double p_bar = std::min(std::max(arma::mean(y), 1e-5), 1.0 - 1e-5);
      beta0 = std::log(p_bar / (1.0 - p_bar));
    }
  }
  if (family != "gaussian" && theta0_init_r.isNotNull()) {
    theta0 = Rcpp::as<arma::vec>(theta0_init_r);
    if ((int)theta0.n_elem != K) stop("theta0_init has incorrect length");
  }

  // Helpers
  int L = groups_x.max();
  int G = groups_z.max();
  std::vector<arma::uvec> group_z_indices(G);
  for (int g = 1; g <= G; ++g) group_z_indices[g - 1] = arma::find(groups_z == (arma::uword)g);

  double lam1 = lambda * (1.0 - alpha);
  double lam2 = lambda * alpha;
  bool screen_ok = use_screen && lam1 > 0.0;

  arma::mat Q;
  if (is_gauss) Q = arma::orth(arma::join_rows(arma::ones<arma::vec>(n), Z));

  // Loss with the unpenalized coefficients fixed (binomial / cox) or
  // profiled out (gaussian)
  auto loss_of = [&](const arma::vec& b, const arma::mat& T) {
    arma::vec eta_pen = predict_sgpl_cpp(X, Z, b, T, 0.0, arma::vec());
    if (is_gauss) {
      arma::vec rr = project_out(Q, y - eta_pen);
      return 0.5 * arma::dot(rr, rr) / n;
    }
    arma::vec eta = eta_pen + Z * theta0;
    if (family == "binomial") eta += beta0;
    return glm_loss(family, y, coxp, eta);
  };
  // psi = d loss / d eta at (b, T)
  auto psi_of = [&](const arma::vec& b, const arma::mat& T) {
    arma::vec eta_pen = predict_sgpl_cpp(X, Z, b, T, 0.0, arma::vec());
    if (is_gauss) return arma::vec(-project_out(Q, y - eta_pen) / n);
    arma::vec eta = eta_pen + Z * theta0;
    if (family == "binomial") eta += beta0;
    return glm_psi(family, y, coxp, eta);
  };

  // Block-specific step sizes
  arma::vec Lip(L, arma::fill::zeros);
  for (int l = 1; l <= L; l++) {
    arma::uvec idx = arma::find(groups_x == (arma::uword)l);
    if (idx.n_elem == 0) continue;
    Lip(l - 1) = family_curvature(family) *
      block_lipschitz(X.cols(idx), Z, Q, is_gauss, idx.n_elem, K);
  }

  std::vector<double> obj_path;
  bool converged = false, monotone = true;
  int n_iter = max_iter_out, bt_failures = 0, prox_cap_hits = 0;
  double obj_prev = loss_of(beta, Theta) +
    compute_sgpl_penalties(beta, Theta, lambda, alpha, groups_x, groups_z, K);

  // Outer loop (block coordinate descent)
  for (int iter_out = 0; iter_out < max_iter_out; iter_out++) {

    arma::vec beta_old = beta;
    arma::mat Theta_old = Theta;
    double beta0_old = beta0;
    arma::vec theta0_old = theta0;

    // Unpenalized block (binomial / cox)
    if (!is_gauss) {
      arma::vec eta_pen = predict_sgpl_cpp(X, Z, beta, Theta, 0.0, arma::vec());
      if (family == "binomial") update_intercepts_logit(Z, y, eta_pen, beta0, theta0);
      else update_intercepts_cox(cox, Z, eta_pen, theta0);
    }

    for (int l = 1; l <= L; l++) {
      arma::uvec idx_l = arma::find(groups_x == (arma::uword)l);
      int p_l = idx_l.n_elem;
      if (p_l == 0) continue;
      arma::mat X_l = X.cols(idx_l);
      SGPLBlock B = make_sgpl_block(p_l, K, group_z_indices);

      // KKT screening (Corollary 1(ii)), evaluated with block l set to zero
      if (screen_ok) {
        arma::vec b0 = beta;
        arma::mat T0 = Theta;
        b0.elem(idx_l).zeros();
        T0.cols(idx_l).zeros();
        arma::vec c_l = -block_crossprod(X_l, Z, psi_of(b0, T0));
        if (arma::norm(soft_thresh_vec(c_l, lam2), 2) <= std::sqrt((double)p_l) * lam1) {
          beta = b0;
          Theta = T0;
          continue;   // skip inner loop
        }
      }

      // Inner loop
      arma::vec v = pack_block(beta.elem(idx_l), Theta.cols(idx_l));
      for (int iter_in = 0; iter_in < max_iter_in; iter_in++) {
        double f_cur = loss_of(beta, Theta);
        arma::vec grad = block_crossprod(X_l, Z, psi_of(beta, Theta));

        double t_l = t_init_r.isNotNull() ? Rcpp::as<double>(t_init_r) : 1.0 / Lip(l - 1);
        bool accepted = false;
        arma::vec cand, d;
        arma::vec beta_cand = beta;
        arma::mat Theta_cand = Theta;

        // Backtracking with the quadratic upper-bound test
        for (int bt = 0; bt < bt_max; bt++) {
          int pit = 0;
          cand = prox_sgpl_block(v - t_l * grad, B, t_l * lam1, t_l * lam2, prox_tol, prox_maxit, pit);
          if (pit >= prox_maxit) ++prox_cap_hits;
          d = cand - v;
          arma::vec bl;
          arma::mat Tl;
          unpack_block(cand, p_l, K, bl, Tl);
          beta_cand.elem(idx_l) = bl;
          Theta_cand.cols(idx_l) = Tl;
          double f_c = loss_of(beta_cand, Theta_cand);
          double bound = f_cur + arma::dot(grad, d) + 0.5 * arma::dot(d, d) / t_l;
          if (f_c <= bound + 1e-12 * std::max(1.0, std::abs(f_cur))) {
            accepted = true;
            break;
          }
          t_l *= bt_factor;
        }

        if (!accepted) {     // never accept a step that fails the test
          ++bt_failures;
          break;
        }
        beta = beta_cand;
        Theta = Theta_cand;
        double delta_in = arma::norm(d, "inf");
        v = cand;
        if (delta_in < tol_in) break;
      }
    }

    // Check outer convergence
    double obj = loss_of(beta, Theta) +
      compute_sgpl_penalties(beta, Theta, lambda, alpha, groups_x, groups_z, K);
    if (obj > obj_prev + 1e-10 * std::max(1.0, std::abs(obj_prev))) monotone = false;
    obj_prev = obj;
    obj_path.push_back(obj);

    double delta_out = std::max(
      arma::max(arma::abs(beta - beta_old)),
      arma::max(arma::vectorise(arma::abs(Theta - Theta_old)))
    );
    if (!is_gauss) {
      delta_out = std::max(delta_out, std::abs(beta0 - beta0_old));
      if (K > 0) delta_out = std::max(delta_out, arma::max(arma::abs(theta0 - theta0_old)));
    }

    if (verbose && ((iter_out + 1) % 10 == 0 || iter_out == 0)) {
      Rcout << "Outer " << (iter_out + 1) << " obj=" << obj << " delta=" << delta_out << "\n";
    }

    if (delta_out < tol_out) {   // Algorithm 1, line 26
      converged = true;
      n_iter = iter_out + 1;
      break;
    }
  }

  // Recover the profiled coefficients (gaussian)
  if (is_gauss) update_intercepts_gaussian(X, Z, y, beta, Theta, beta0, theta0);

  return Rcpp::List::create(
    Named("beta") = beta,
    Named("Theta") = Theta,
    Named("beta0") = beta0,
    Named("theta0") = theta0,
    Named("obj_path") = obj_path,
    Named("converged") = converged,
    Named("n_iter") = n_iter,
    Named("lambda") = lambda,
    Named("alpha") = alpha,
    Named("groups_x") = groups_x,
    Named("groups_z") = groups_z,
    Named("monotone") = monotone,
    Named("bt_failures") = bt_failures,
    Named("prox_cap_hits") = prox_cap_hits
  );
}
