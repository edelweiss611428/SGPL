#include <RcppArmadillo.h>
#include "sgpl_helpers.h"

using namespace Rcpp;
using namespace arma;

// --- Fast Loss Functions via eta ---

inline double compute_gaussian_loss_eta(const arma::vec& y, const arma::vec& eta) {
  arma::vec r = y - eta;
  return 0.5 * arma::dot(r, r) / y.n_elem;
}

inline double compute_logistic_loss_eta(const arma::vec& y, const arma::vec& eta) {
  double logloss = 0.0;
  int n = y.n_elem;
  for (int i = 0; i < n; ++i) {
    if (eta(i) > 0.0) {
      logloss += (1.0 - y(i)) * eta(i) + std::log1p(std::exp(-eta(i)));
    } else {
      logloss += -y(i) * eta(i) + std::log1p(std::exp(eta(i)));
    }
  }
  return logloss / n;
}

// --- Fast Penalty Function (Flat Arguments) ---

inline double compute_sgpl_penalties_fast(
    const arma::vec& beta,
    const arma::mat& Theta,
    double lambda,
    double alpha,
    unsigned int L,
    unsigned int G,
    const arma::vec& pl,
    const arma::vec& pg,
    const std::vector<arma::uvec>& group_x_indices,
    const std::vector<arma::uvec>& group_z_indices,
    double w_denom
) {
  double lam1 = lambda * (1.0 - alpha);
  double lam2 = lambda * alpha;

  arma::mat Theta2 = arma::square(Theta);
  arma::vec beta2 = arma::square(beta);

  // 1. Per-variable predictor penalty
  arma::vec col_sums_Theta2 = arma::sum(Theta2, 0).t();
  double per_pred = lam1 * arma::accu(arma::sqrt(beta2 + col_sums_Theta2));

  // 2. Joint predictor group penalty
  double joint_grp = 0.0;
  for (unsigned int l = 0; l < L; ++l) {
    const arma::uvec& idx = group_x_indices[l];
    if (idx.n_elem > 0) {
      double svec = arma::accu(beta2.elem(idx)) + arma::accu(col_sums_Theta2.elem(idx));
      joint_grp += std::sqrt(pl(l)) * std::sqrt(svec);
    }
  }
  joint_grp *= lam1;

  // 3. Modifier group penalty
  double z_grp = 0.0;
  for (unsigned int l = 0; l < L; ++l) {
    const arma::uvec& idx_l = group_x_indices[l];
    if (idx_l.n_elem == 0) continue;

    for (unsigned int g = 0; g < G; ++g) {
      const arma::uvec& idx_g = group_z_indices[g];
      if (idx_g.n_elem == 0) continue;

      double s = arma::accu(Theta2.submat(idx_g, idx_l));
      z_grp += (std::sqrt(pg(g)) / w_denom) * std::sqrt(s);
    }
  }
  z_grp *= lam1;

  // 4. L1 Lasso Penalties
  double l1_b = lam2 * arma::accu(arma::abs(beta));
  double l1_t = lam2 * arma::accu(arma::abs(Theta));

  return per_pred + joint_grp + z_grp + l1_b + l1_t;
}

// --- Fast Objective Function ---

inline double objective_sgpl_cpp_fast(
    const arma::vec& y,
    const arma::vec& eta,
    const arma::vec& beta,
    const arma::mat& Theta,
    double lambda,
    double alpha,
    unsigned int L,
    unsigned int G,
    const arma::vec& pl,
    const arma::vec& pg,
    const std::vector<arma::uvec>& group_x_indices,
    const std::vector<arma::uvec>& group_z_indices,
    double w_denom,
    const std::string& family
) {
  double loss = 0.0;
  if (family == "gaussian") {
    loss = compute_gaussian_loss_eta(y, eta);
  } else if (family == "binomial") {
    loss = compute_logistic_loss_eta(y, eta);
  }

  double penalty = compute_sgpl_penalties_fast(
    beta, Theta, lambda, alpha, L, G, pl, pg,
    group_x_indices, group_z_indices, w_denom
  );
  return loss + penalty;
}

// --- Helper for Block Linear Predictor ---

inline arma::vec compute_block_eta(
    const arma::mat& X_l,
    const arma::mat& Z,
    const arma::vec& beta_l,
    const arma::mat& Theta_l
) {
  int n = X_l.n_rows;
  int p_l = X_l.n_cols;
  arma::vec eta_l(n, arma::fill::zeros);

  for (int jj = 0; jj < p_l; ++jj) {
    eta_l += X_l.col(jj) % (beta_l(jj) + Z * Theta_l.col(jj));
  }
  return eta_l;
}

// --- Main Fitting Function ---

// [[Rcpp::export]]
Rcpp::List fast_sgpl_fit_cpp(
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
    std::string family = "gaussian"
) {
  int n = X.n_rows;
  int p = X.n_cols;
  int K = Z.n_cols;

  // --- Dimension Checks ---
  if ((int)Z.n_rows != n) Rcpp::stop("X and Z must have same number of rows");
  if ((int)y.n_elem != n) Rcpp::stop("length(y) must equal nrow(X)");

  // --- Groups Initialization ---
  arma::uvec groups_x = groups_x_r.isNull() ? arma::linspace<arma::uvec>(1, p, p) : Rcpp::as<arma::uvec>(groups_x_r);
  arma::uvec groups_z = groups_z_r.isNull() ? arma::linspace<arma::uvec>(1, K, K) : Rcpp::as<arma::uvec>(groups_z_r);

  if ((int)groups_x.n_elem != p) Rcpp::stop("groups_x has incorrect length");
  if ((int)groups_z.n_elem != K) Rcpp::stop("groups_z has incorrect length");

  // --- Flat Precomputations ---
  unsigned int L = groups_x.max();
  unsigned int G = groups_z.max();
  double w_denom = std::sqrt(1.0 + K);

  arma::vec pl(L, arma::fill::zeros);
  arma::vec pg(G, arma::fill::zeros);

  std::vector<arma::uvec> group_x_indices(L);
  for (unsigned int l = 1; l <= L; ++l) {
    group_x_indices[l - 1] = arma::find(groups_x == l);
    pl(l - 1) = group_x_indices[l - 1].n_elem;
  }

  std::vector<arma::uvec> group_z_indices(G);
  for (unsigned int g = 1; g <= G; ++g) {
    group_z_indices[g - 1] = arma::find(groups_z == g);
    pg(g - 1) = group_z_indices[g - 1].n_elem;
  }

  // --- Parameter Initializations ---
  arma::vec beta = beta_init_r.isNull() ? arma::vec(p, arma::fill::zeros) : Rcpp::as<arma::vec>(beta_init_r);
  arma::mat Theta = Theta_init_r.isNull() ? arma::mat(K, p, arma::fill::zeros) : Rcpp::as<arma::mat>(Theta_init_r);

  double beta0 = 0.0;
  arma::vec theta0(K, arma::fill::zeros);

  if (beta0_init_r.isNotNull()) {
    beta0 = Rcpp::as<double>(beta0_init_r);
  } else {
    if (family == "binomial") {
      double p_bar = std::min(std::max(arma::mean(y), 1e-5), 1.0 - 1e-5);
      beta0 = std::log(p_bar / (1.0 - p_bar));
    } else if (family == "gaussian") {
      beta0 = arma::mean(y);
    }
  }

  if (theta0_init_r.isNotNull()) {
    theta0 = Rcpp::as<arma::vec>(theta0_init_r);
    if ((int)theta0.n_elem != K) Rcpp::stop("theta0_init has incorrect length");
  }

  double lam1 = lambda * (1.0 - alpha);
  double lam2 = lambda * alpha;
  double t_init = compute_t_init(X, Z, t_init_r, family);

  // --- Global Linear Predictor Initialization ---
  arma::vec eta = predict_sgpl_cpp(X, Z, beta, Theta, beta0, theta0);

  arma::vec obj_path(max_iter_out);
  bool converged = false;
  int n_iter = max_iter_out;

  // --- Outer Loop ---
  for (int iter_out = 0; iter_out < max_iter_out; ++iter_out) {

    arma::vec beta_old = beta;
    arma::mat Theta_old = Theta;
    double beta0_old = beta0;
    arma::vec theta0_old = theta0;

    // Update Intercepts
    if (iter_out > 0 || (beta0_init_r.isNull() && theta0_init_r.isNull())) {
      eta -= (beta0 + Z * theta0);

      if (family == "gaussian") {
        update_intercepts_gaussian(X, Z, y, beta, Theta, beta0, theta0);
      } else if (family == "binomial") {
        arma::mat M = Z * Theta;
        M.each_row() += beta.t();
        arma::vec eta_pen = arma::sum(X % M, 1);
        update_intercepts_logit(Z, y, eta_pen, beta0, theta0);
      }

      eta += (beta0 + Z * theta0);
    }

    // Loop over predictor groups
    for (unsigned int l = 0; l < L; ++l) {
      const arma::uvec& idx_l = group_x_indices[l];
      int p_l = idx_l.n_elem;
      if (p_l == 0) continue;

      arma::mat X_l = X.cols(idx_l);

      arma::vec beta_l_tilde = beta.elem(idx_l);
      arma::mat Theta_l_tilde = Theta.cols(idx_l);

      // Subtract current block contribution from global eta
      arma::vec eta_l_old = compute_block_eta(X_l, Z, beta_l_tilde, Theta_l_tilde);
      arma::vec eta_neg_l = eta - eta_l_old;

      arma::vec r_neg_l = y - eta_neg_l;

      // KKT Screening (Gaussian)
      if (use_screen && family == "gaussian") {
        arma::vec g_l = (X_l.t() * r_neg_l) / n;
        arma::mat H_l(K, p_l);
        for (int jj = 0; jj < p_l; ++jj) {
          H_l.col(jj) = (Z.t() * (X_l.col(jj) % r_neg_l)) / n;
        }

        arma::vec vv = arma::join_cols(g_l, arma::vectorise(H_l));
        vv = soft_thresh_vec(vv, lam2);

        if (arma::norm(vv, 2) <= std::sqrt((double)p_l) * lam1) {
          beta.elem(idx_l).zeros();
          Theta.cols(idx_l).zeros();
          eta = eta_neg_l;
          continue;
        }
      }

      double t_l = t_init;

      // Inner Loop
      for (int iter_in = 0; iter_in < max_iter_in; ++iter_in) {
        arma::vec beta_l_old_in = beta_l_tilde;
        arma::mat Theta_l_old_in = Theta_l_tilde;

        arma::vec eta_l_current = eta_neg_l + compute_block_eta(X_l, Z, beta_l_tilde, Theta_l_tilde);

        arma::vec gb = gradient_smooth_loss_beta(X_l, y, eta_l_current, n, family);
        arma::mat gT = gradient_smooth_loss_theta(X_l, Z, y, eta_l_current, p_l, K, n, family);

        arma::vec beta_l_new;
        arma::mat Theta_l_new;

        // Precompute base current objective for backtracking
        arma::vec beta_curr = beta;
        arma::mat Theta_curr = Theta;
        beta_curr.elem(idx_l) = beta_l_tilde;
        Theta_curr.cols(idx_l) = Theta_l_tilde;

        double obj_curr = objective_sgpl_cpp_fast(
          y, eta_l_current, beta_curr, Theta_curr, lambda, alpha,
          L, G, pl, pg, group_x_indices, group_z_indices, w_denom, family
        );

        // Backtracking line search
        for (int bt = 0; bt < bt_max; ++bt) {
          arma::vec beta_try = soft_thresh_vec(beta_l_tilde - t_l * gb, t_l * lam2);
          arma::mat Theta_try = soft_thresh_mat(Theta_l_tilde - t_l * gT, t_l * lam2);

          // Predictor block shrinkage
          for (int jj = 0; jj < p_l; ++jj) {
            arma::vec vv = arma::join_cols(arma::vec({beta_try(jj)}), Theta_try.col(jj));
            vv = block_soft(vv, t_l * lam1);
            beta_try(jj) = vv(0);
            Theta_try.col(jj) = vv.subvec(1, K);
          }

          // Modifier group shrinkage
          for (unsigned int g = 0; g < G; ++g) {
            const arma::uvec& idxg = group_z_indices[g];
            int ng = idxg.n_elem;
            if (ng == 0) continue;

            arma::vec vv = arma::vectorise(Theta_try.rows(idxg));
            double w = std::sqrt(pg(g)) / w_denom;
            vv = block_soft(vv, t_l * lam1 * w);

            Theta_try.rows(idxg) = arma::reshape(vv, ng, p_l);
          }

          // Full group shrink
          arma::vec vv = arma::join_cols(beta_try, arma::vectorise(Theta_try));
          vv = block_soft(vv, t_l * lam1 * std::sqrt((double)p_l));

          beta_try = vv.head(p_l);
          Theta_try = arma::reshape(vv.tail(K * p_l), K, p_l);

          beta_l_new = beta_try;
          Theta_l_new = Theta_try;

          // Candidate evaluation
          arma::vec beta_cand = beta;
          arma::mat Theta_cand = Theta;
          beta_cand.elem(idx_l) = beta_l_new;
          Theta_cand.cols(idx_l) = Theta_l_new;

          arma::vec eta_cand = eta_neg_l + compute_block_eta(X_l, Z, beta_l_new, Theta_l_new);

          double obj_cand = objective_sgpl_cpp_fast(
            y, eta_cand, beta_cand, Theta_cand, lambda, alpha,
            L, G, pl, pg, group_x_indices, group_z_indices, w_denom, family
          );

          if (obj_cand <= obj_curr + 1e-12) break;
          t_l *= bt_factor;
        }

        beta_l_tilde = beta_l_new;
        Theta_l_tilde = Theta_l_new;

        double delta_in = std::max(
          arma::max(arma::abs(beta_l_tilde - beta_l_old_in)),
          arma::max(arma::vectorise(arma::abs(Theta_l_tilde - Theta_l_old_in)))
        );

        if (delta_in < tol_in) break;
      }

      // Accept block update
      beta.elem(idx_l) = beta_l_tilde;
      Theta.cols(idx_l) = Theta_l_tilde;

      // Sync global linear predictor
      eta = eta_neg_l + compute_block_eta(X_l, Z, beta_l_tilde, Theta_l_tilde);
    }

    // Check outer convergence
    obj_path(iter_out) = objective_sgpl_cpp_fast(
      y, eta, beta, Theta, lambda, alpha,
      L, G, pl, pg, group_x_indices, group_z_indices, w_denom, family
    );

    double delta_out = std::max({
      arma::max(arma::abs(beta - beta_old)),
      arma::max(arma::vectorise(arma::abs(Theta - Theta_old))),
      std::abs(beta0 - beta0_old),
      arma::max(arma::abs(theta0 - theta0_old))
    });

    if (verbose && ((iter_out + 1) % 10 == 0 || iter_out == 0)) {
      Rcout << "Outer " << (iter_out + 1) << " obj=" << obj_path(iter_out) << " delta=" << delta_out << "\n";
    }

    if (delta_out < tol_out && iter_out > 0) {
      converged = true;
      n_iter = iter_out + 1;
      break;
    }
  }

  return Rcpp::List::create(
    Named("beta") = beta,
    Named("Theta") = Theta,
    Named("beta0") = beta0,
    Named("theta0") = theta0,
    Named("obj_path") = obj_path.head(n_iter),
    Named("converged") = converged,
    Named("n_iter") = n_iter,
    Named("lambda") = lambda,
    Named("alpha") = alpha,
    Named("groups_x") = groups_x,
    Named("groups_z") = groups_z
  );
}
