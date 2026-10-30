#include <RcppArmadillo.h>
#include "sgpl_helpers.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// Cached implementation of Algorithm 1 (exact block proximal updates).
//
// Changes from the previous version (four-step shrinkage):
//   * the block proximal operator is computed exactly (Dykstra-like algorithm);
//   * backtracking uses the quadratic upper-bound (sufficient-decrease) test,
//     and a step that fails the test is NEVER accepted (the cache is always
//     consistent with the coefficients);
//   * step sizes start from 1 / L_l with a block-specific Lipschitz constant;
//   * safe screening (Corollary 1(ii)) is used for all three families;
//   * gaussian: (beta0, theta0) are profiled out and recovered at the end;
//     binomial / cox: the unpenalized block is updated once per outer iteration
//     (cox: theta0 only; beta0 is absorbed into the baseline hazard);
//   * family = "cox" added (y = observed times, status_r = event indicators).

// [[Rcpp::export]]
Rcpp::List fast_sgpl_fit_cpp6(
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
) {
  const int n = X.n_rows;
  const int p = X.n_cols;
  const int K = Z.n_cols;

  // --- Checks ---
  if ((int)Z.n_rows != n) Rcpp::stop("X and Z must have same number of rows");
  if ((int)y.n_elem != n) Rcpp::stop("length(y) must equal nrow(X)");
  if (family != "gaussian" && family != "binomial" && family != "cox")
    Rcpp::stop("family must be 'gaussian', 'binomial' or 'cox'");
  if (alpha < 0.0 || alpha > 1.0) Rcpp::stop("alpha must lie in [0, 1]");
  if (lambda < 0.0) Rcpp::stop("lambda must be non-negative");
  const bool is_gauss = (family == "gaussian");

  CoxData cox;
  if (family == "cox") {
    if (status_r.isNull()) Rcpp::stop("status must be supplied for family = 'cox'");
    arma::vec status = Rcpp::as<arma::vec>(status_r);
    if ((int)status.n_elem != n) Rcpp::stop("length(status) must equal nrow(X)");
    cox = make_cox_data(y, status);
  }
  const CoxData* coxp = (family == "cox") ? &cox : nullptr;

  // --- Groups ---
  arma::uvec groups_x = groups_x_r.isNull() ? arma::linspace<arma::uvec>(1, p, p) : Rcpp::as<arma::uvec>(groups_x_r);
  arma::uvec groups_z = groups_z_r.isNull() ? arma::linspace<arma::uvec>(1, K, K) : Rcpp::as<arma::uvec>(groups_z_r);
  if ((int)groups_x.n_elem != p) Rcpp::stop("groups_x has incorrect length");
  if ((int)groups_z.n_elem != K) Rcpp::stop("groups_z has incorrect length");

  const int L = groups_x.max();
  const int G = groups_z.max();

  std::vector<arma::uvec> group_x_indices(L);
  for (int l = 1; l <= L; ++l) group_x_indices[l - 1] = arma::find(groups_x == (arma::uword)l);
  std::vector<arma::uvec> group_z_indices(G);
  for (int g = 1; g <= G; ++g) group_z_indices[g - 1] = arma::find(groups_z == (arma::uword)g);

  // --- Parameters ---
  arma::vec beta = beta_init_r.isNull() ? arma::vec(p, arma::fill::zeros) : Rcpp::as<arma::vec>(beta_init_r);
  arma::mat Theta = Theta_init_r.isNull() ? arma::mat(K, p, arma::fill::zeros) : Rcpp::as<arma::mat>(Theta_init_r);
  if ((int)beta.n_elem != p) Rcpp::stop("beta_init has incorrect length");
  if ((int)Theta.n_rows != K || (int)Theta.n_cols != p) Rcpp::stop("Theta_init has incorrect dimensions");

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
    if ((int)theta0.n_elem != K) Rcpp::stop("theta0_init has incorrect length");
  }
  // gaussian: (beta0, theta0) are profiled out; beta0_init / theta0_init are ignored.

  const double lam1 = lambda * (1.0 - alpha);
  const double lam2 = lambda * alpha;
  const bool screen_ok = use_screen && lam1 > 0.0;

  // --- Profiling (gaussian) ---
  arma::mat Q;
  if (is_gauss) Q = arma::orth(arma::join_rows(arma::ones<arma::vec>(n), Z));

  // --- Block data ---
  std::vector<arma::mat> X_blocks(L);
  std::vector<SGPLBlock> blocks(L);
  arma::vec Lip(L, arma::fill::zeros);
  for (int l = 0; l < L; ++l) {
    const arma::uvec& idx = group_x_indices[l];
    int p_l = idx.n_elem;
    if (p_l == 0) continue;
    X_blocks[l] = X.cols(idx);
    blocks[l] = make_sgpl_block(p_l, K, group_z_indices);
    Lip(l) = family_curvature(family) * block_lipschitz(X_blocks[l], Z, Q, is_gauss, p_l, K);
  }
  const bool user_t = t_init_r.isNotNull();
  const double t_user = user_t ? Rcpp::as<double>(t_init_r) : 0.0;

  // --- Caches ---
  arma::mat ZTheta = Z * Theta;                         // n x p
  arma::vec eta_pen(n, arma::fill::zeros);              // X~ gamma
  {
    arma::mat M = ZTheta;
    M.each_row() += beta.t();
    eta_pen = arma::sum(X % M, 1);
  }
  arma::vec pen_vec(L, arma::fill::zeros);
  for (int l = 0; l < L; ++l) {
    const arma::uvec& idx = group_x_indices[l];
    if (idx.n_elem == 0) continue;
    pen_vec(l) = block_penalty_sgpl(pack_block(beta.elem(idx), Theta.cols(idx)), blocks[l], lam1, lam2);
  }

  arma::vec ytil, r, offset, eta;
  if (is_gauss) {
    ytil = project_out(Q, y);
    r = ytil - project_out(Q, eta_pen);                 // profiled residual
  } else {
    offset = beta0 + Z * theta0;
    eta = offset + eta_pen;
  }

  auto current_loss = [&]() {
    return is_gauss ? 0.5 * arma::dot(r, r) / n : glm_loss(family, y, coxp, eta);
  };

  std::vector<double> obj_path;
  bool converged = false, monotone = true;
  int n_iter = max_iter_out, bt_failures = 0, prox_cap_hits = 0, n_screened = 0;
  double obj_prev = current_loss() + arma::accu(pen_vec);

  // --- Outer loop ---
  for (int iter_out = 0; iter_out < max_iter_out; ++iter_out) {

    arma::vec beta_old = beta;
    arma::mat Theta_old = Theta;
    double beta0_old = beta0;
    arma::vec theta0_old = theta0;

    // Unpenalized block (binomial / cox), with the penalized fit held fixed
    if (family == "binomial") {
      update_intercepts_logit(Z, y, eta_pen, beta0, theta0);
    } else if (family == "cox") {
      update_intercepts_cox(cox, Z, eta_pen, theta0);
    }
    if (!is_gauss) {
      offset = beta0 + Z * theta0;
      eta = offset + eta_pen;
    }

    // Loop over predictor groups
    for (int l = 0; l < L; ++l) {
      const arma::uvec& idx_l = group_x_indices[l];
      const int p_l = idx_l.n_elem;
      if (p_l == 0) continue;
      const arma::mat& X_l = X_blocks[l];
      const SGPLBlock& B = blocks[l];

      arma::vec v = pack_block(beta.elem(idx_l), Theta.cols(idx_l));
      arma::vec u_old = X_l * beta.elem(idx_l) + arma::sum(X_l % ZTheta.cols(idx_l), 1);

      // Base quantities excluding block l
      arma::vec r_minus, eta_minus;
      if (is_gauss) r_minus = r + project_out(Q, u_old);
      else eta_minus = eta - u_old;

      // Safe screening (Corollary 1(ii))
      if (screen_ok) {
        arma::vec c_l = is_gauss
          ? arma::vec(block_crossprod(X_l, Z, r_minus) / n)
          : arma::vec(-block_crossprod(X_l, Z, glm_psi(family, y, coxp, eta_minus)));
        if (arma::norm(soft_thresh_vec(c_l, lam2), 2) <= std::sqrt((double)p_l) * lam1) {
          beta.elem(idx_l).zeros();
          Theta.cols(idx_l).zeros();
          ZTheta.cols(idx_l).zeros();
          eta_pen -= u_old;
          if (is_gauss) r = r_minus; else eta = eta_minus;
          pen_vec(l) = 0.0;
          ++n_screened;
          continue;
        }
      }

      // Proximal-gradient steps on block l
      arma::vec u = u_old;
      arma::vec r_cur, eta_cur;
      double f_cur;
      if (is_gauss) {
        r_cur = r;
        f_cur = 0.5 * arma::dot(r_cur, r_cur) / n;
      } else {
        eta_cur = eta;
        f_cur = glm_loss(family, y, coxp, eta_cur);
      }

      for (int iter_in = 0; iter_in < max_iter_in; ++iter_in) {
        arma::vec psi = is_gauss ? arma::vec(-r_cur / n) : glm_psi(family, y, coxp, eta_cur);
        arma::vec grad = block_crossprod(X_l, Z, psi);

        double t_l = user_t ? t_user : 1.0 / Lip(l);
        bool accepted = false;
        arma::vec cand, d, u_c, r_c, eta_c;
        double f_c = f_cur;

        for (int bt = 0; bt < bt_max; ++bt) {
          int pit = 0;
          cand = prox_sgpl_block(v - t_l * grad, B, t_l * lam1, t_l * lam2, prox_tol, prox_maxit, pit);
          if (pit >= prox_maxit) ++prox_cap_hits;
          d = cand - v;
          u_c = block_fit(X_l, Z, cand, p_l, K);
          if (is_gauss) {
            r_c = r_minus - project_out(Q, u_c);
            f_c = 0.5 * arma::dot(r_c, r_c) / n;
          } else {
            eta_c = eta_minus + u_c;
            f_c = glm_loss(family, y, coxp, eta_c);
          }
          double bound = f_cur + arma::dot(grad, d) + 0.5 * arma::dot(d, d) / t_l;
          if (f_c <= bound + 1e-12 * std::max(1.0, std::abs(f_cur))) {
            accepted = true;
            break;
          }
          t_l *= bt_factor;
        }

        if (!accepted) {          // never accept a step that fails the test
          ++bt_failures;
          break;
        }

        double delta_in = arma::norm(d, "inf");
        v = cand;
        u = u_c;
        f_cur = f_c;
        if (is_gauss) r_cur = r_c; else eta_cur = eta_c;
        if (delta_in < tol_in) break;
      }

      // Commit block l (caches stay consistent with the coefficients)
      arma::vec beta_l;
      arma::mat Theta_l;
      unpack_block(v, p_l, K, beta_l, Theta_l);
      beta.elem(idx_l) = beta_l;
      Theta.cols(idx_l) = Theta_l;
      ZTheta.cols(idx_l) = Z * Theta_l;
      eta_pen += u - u_old;
      if (is_gauss) r = r_cur; else eta = eta_cur;
      pen_vec(l) = block_penalty_sgpl(v, B, lam1, lam2);
    }

    // Outer convergence
    double obj = current_loss() + arma::accu(pen_vec);
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
    Named("prox_cap_hits") = prox_cap_hits,
    Named("n_screened") = n_screened
  );
}
