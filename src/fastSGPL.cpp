#include <RcppArmadillo.h>
#include "sgpl_helpers.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// Block modifier penalty

inline double compute_block_modifier_penalty(
    const arma::mat& Theta_l,
    int G,
    const arma::vec& pg,
    const std::vector<arma::uvec>& group_z_indices,
    double w_denom
) {
  double z_grp_l = 0.0;
  for (int g = 0; g < G; ++g) {
    const arma::uvec& idx_g = group_z_indices[g];
    if (idx_g.n_elem == 0) continue;

    double s = arma::accu(arma::square(Theta_l.rows(idx_g)));
    z_grp_l += (std::sqrt(pg(g)) / w_denom) * std::sqrt(s);
  }
  return z_grp_l;
}

// Block predictive contribution

inline arma::vec compute_block_eta_fast(
    const arma::mat& X_l,
    const arma::vec& beta_l,
    const arma::mat& ZTheta_l
) {
  // Uses precomputed ZTheta_l = Z * Theta_l
  arma::vec eta_l = X_l * beta_l;
  eta_l += arma::sum(X_l % ZTheta_l, 1);
  return eta_l;
}


// KKT screening (for Gaussian only)


inline bool screen_kkt_gaussian(
    std::size_t l, bool use_screen, const std::string& family,
    const arma::mat& X_l, const arma::vec& r_neg_l, const arma::mat& Z,
    double n, std::size_t p_l,
    double lam1, double lam2,
    const arma::uvec& idx_l, const arma::vec& beta_l_tilde, const arma::mat& Theta_l_tilde,
    arma::vec& beta, arma::mat& Theta, arma::vec& beta2, arma::vec& col_sums_Theta2, arma::mat& ZTheta,
    arma::vec& eta, const arma::vec& eta_neg_l,
    arma::vec& joint_grp_vec, arma::vec& mod_grp_vec,
    double& total_per_pred, double& total_joint_grp, double& total_mod_grp,
    double& total_l1_b, double& total_l1_t, double& total_penalty
) {
  if (!use_screen || family != "gaussian") {
    return false;
  }

  arma::vec g_l = (X_l.t() * r_neg_l) / n;
  arma::mat H_l = (Z.t() * (X_l.each_col() % r_neg_l)) / n;

  arma::vec vv = arma::join_cols(g_l, arma::vectorise(H_l));
  vv = soft_thresh_vec(vv, lam2);

  if (arma::norm(vv, 2) <= std::sqrt(static_cast<double>(p_l)) * lam1) {

    // Compute penalty change from zeroing out block l
    double per_pred_old_l = lam1 * arma::accu(arma::sqrt(beta2.elem(idx_l) + col_sums_Theta2.elem(idx_l)));
    double l1_b_old_l = lam2 * arma::accu(arma::abs(beta_l_tilde));
    double l1_t_old_l = lam2 * arma::accu(arma::abs(Theta_l_tilde));

    total_per_pred  -= per_pred_old_l;
    total_joint_grp -= joint_grp_vec(l);
    joint_grp_vec(l) = 0.0;

    total_mod_grp   -= mod_grp_vec(l);
    mod_grp_vec(l)   = 0.0;

    total_l1_b -= l1_b_old_l;
    total_l1_t -= l1_t_old_l;

    total_penalty = total_per_pred + total_joint_grp + total_mod_grp + total_l1_b + total_l1_t;

    beta.elem(idx_l).zeros();
    Theta.cols(idx_l).zeros();
    beta2.elem(idx_l).zeros();
    col_sums_Theta2.elem(idx_l).zeros();
    ZTheta.cols(idx_l).zeros();

    eta = eta_neg_l;
    return true;
  }
  return false;
}

// Optimisation algorithm

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

  int L = groups_x.max();
  int G = groups_z.max();
  double w_denom = std::sqrt(1.0 + K);

  arma::vec pl(L, arma::fill::zeros);
  arma::vec pg(G, arma::fill::zeros);

  std::vector<arma::uvec> group_x_indices(L);
  for (int l = 1; l <= L; ++l) {
    group_x_indices[l - 1] = arma::find(groups_x == l);
    pl(l - 1) = group_x_indices[l - 1].n_elem;
  }

  std::vector<arma::uvec> group_z_indices(G);
  for (int g = 1; g <= G; ++g) {
    group_z_indices[g - 1] = arma::find(groups_z == g);
    pg(g - 1) = group_z_indices[g - 1].n_elem;
  }

  // Param initialisation
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
    if ((int)theta0.n_elem != K) Rcpp::stop("theta0_init has incorrect length");
    theta0 = Rcpp::as<arma::vec>(theta0_init_r);
  }



  double lam1 = lambda * (1.0 - alpha);
  double lam2 = lambda * alpha;
  double t_init = compute_t_init(X, Z, t_init_r, family);


  // Cache matrix for Z * Theta
  arma::mat ZTheta = Z * Theta;

  // CURRENT VALUES

  // Cached beta2 and col_sums_Theta2
  arma::vec beta2 = arma::square(beta);
  arma::vec col_sums_Theta2 = arma::sum(arma::square(Theta), 0).t();

  // Track individual component sums for fast scalar updates
  double total_per_pred = lam1 * arma::accu(arma::sqrt(beta2 + col_sums_Theta2));

  arma::vec joint_grp_vec(L, arma::fill::zeros);
  for (int l = 0; l < L; ++l) {
    const arma::uvec& idx = group_x_indices[l];
    if (idx.n_elem > 0) {
      double svec = arma::accu(beta2.elem(idx)) + arma::accu(col_sums_Theta2.elem(idx));
      joint_grp_vec(l) = lam1 * std::sqrt(pl(l)) * std::sqrt(svec);
    }
  }
  double total_joint_grp = arma::accu(joint_grp_vec);

  arma::vec mod_grp_vec(L, arma::fill::zeros);
  for (int l = 0; l < L; ++l) {
    const arma::uvec& idx_l = group_x_indices[l];
    if (idx_l.n_elem == 0) continue;
    mod_grp_vec(l) = lam1 * compute_block_modifier_penalty(Theta.cols(idx_l), G, pg, group_z_indices, w_denom);
  }

  double total_mod_grp = arma::accu(mod_grp_vec);
  double total_l1_b = lam2 * arma::accu(arma::abs(beta));
  double total_l1_t = lam2 * arma::accu(arma::abs(Theta));

  // Base total penalty
  double total_penalty = total_per_pred + total_joint_grp + total_mod_grp + total_l1_b + total_l1_t;

  // Global predictor
  arma::vec eta = predict_sgpl_cpp(X, Z, beta, Theta, beta0, theta0);

  arma::vec obj_path(max_iter_out);
  bool converged = false;
  int n_iter = max_iter_out;

  // Outer loop
  for (int iter_out = 0; iter_out < max_iter_out; ++iter_out) {

    arma::vec beta_old = beta;
    arma::mat Theta_old = Theta;
    double beta0_old = beta0;
    arma::vec theta0_old = theta0;

    // Intercept updates
    if (iter_out > 0 || (beta0_init_r.isNull() && theta0_init_r.isNull())) {
      eta -= (beta0 + Z * theta0);
      if (family == "gaussian") {
        update_intercepts_gaussian(X, Z, y, beta, Theta, beta0, theta0);
      } else if (family == "binomial") {
        arma::mat M = Z * Theta;
        M.each_row() += beta.t();
        arma::vec eta_pen = arma::sum(X % M, 1);
        update_intercepts_logit(Z, y, eta_pen, beta0, theta0); //future version allow users to input tolerance for irls solver
      }
      eta += (beta0 + Z * theta0);
    }

    // Loop over predictor groups
    for (int l = 0; l < L; ++l) {


      const arma::uvec& idx_l = group_x_indices[l];
      int p_l = idx_l.n_elem;
      if (p_l == 0) continue;

      arma::mat X_l = X.cols(idx_l);
      arma::vec beta_l_tilde = beta.elem(idx_l);
      arma::mat Theta_l_tilde = Theta.cols(idx_l);
      arma::mat ZTheta_l = ZTheta.cols(idx_l); // Fetch cached ZTheta columns

      arma::vec eta_l_old = compute_block_eta_fast(X_l, beta_l_tilde, ZTheta_l);
      arma::vec eta_neg_l = eta - eta_l_old;
      arma::vec r_neg_l = y - eta_neg_l;
      arma::vec eta_l_current = eta;

      // can modify objects if kkt conditions are met
      bool skipped = screen_kkt_gaussian(
        l, use_screen, family, X_l, r_neg_l, Z, n, p_l, lam1, lam2,
        idx_l, beta_l_tilde, Theta_l_tilde, beta, Theta, beta2,
        col_sums_Theta2, ZTheta, eta_l_current, eta_neg_l, joint_grp_vec,
        mod_grp_vec, total_per_pred, total_joint_grp, total_mod_grp,
        total_l1_b, total_l1_t, total_penalty
      );

      if(skipped){
        continue;
      }

      double t_l = t_init;

      // Inner loop
      for (int iter_in = 0; iter_in < max_iter_in; ++iter_in) {

        arma::vec beta_l_old_in = beta_l_tilde;
        arma::mat Theta_l_old_in = Theta_l_tilde;
        // arma::vec eta_l_current = eta_neg_l + compute_block_eta_fast(X_l, beta_l_tilde, ZTheta_l);

        arma::vec gb = gradient_smooth_loss_beta(X_l, y, eta_l_current, n, family);
        arma::mat gT = gradient_smooth_loss_theta(X_l, Z, y, eta_l_current, p_l, K, n, family);

        arma::vec beta_l_new(p_l);
        arma::mat Theta_l_new(K, p_l);

        // Cache current old penalty metrics specific to block l (this is a bit inefficient - computed twice)
        // Future versions could improve this step and the part where we compute eta_l_current (empirical
        // evidence shows ~ 10% gain in computational efficiency).

        double per_pred_old_l = lam1 * arma::accu(arma::sqrt(beta2.elem(idx_l) + col_sums_Theta2.elem(idx_l)));
        double joint_grp_old_l = joint_grp_vec(l);
        double mod_grp_old_l = mod_grp_vec(l);
        double l1_b_old_l = lam2 * arma::accu(arma::abs(beta_l_tilde));
        double l1_t_old_l = lam2 * arma::accu(arma::abs(Theta_l_tilde));

        double loss_curr = (family == "gaussian") ? compute_gaussian_loss(y, eta_l_current)
          : compute_logistic_loss(y, eta_l_current);
        double obj_curr = loss_curr + total_penalty;

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
          for (int g = 0; g < G; ++g) {
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

          beta_l_new = vv.head(p_l);
          Theta_l_new = arma::reshape(vv.tail(K * p_l), K, p_l);

          // Candidate ZTheta matrix computed ONCE per step
          arma::mat ZTheta_l_cand = Z * Theta_l_new;

          // Compute new block local penalties in O(K * p_l)
          arma::vec beta2_l_cand = arma::square(beta_l_new);
          arma::vec col_sums_Theta2_l_cand = arma::sum(arma::square(Theta_l_new), 0).t();

          double per_pred_cand_l = lam1 * arma::accu(arma::sqrt(beta2_l_cand + col_sums_Theta2_l_cand));
          double joint_grp_cand_l = lam1 * std::sqrt(pl(l)) * std::sqrt(arma::accu(beta2_l_cand) + arma::accu(col_sums_Theta2_l_cand));
          double mod_grp_cand_l = lam1 * compute_block_modifier_penalty(Theta_l_new, G, pg, group_z_indices, w_denom);
          double l1_b_cand_l = lam2 * arma::accu(arma::abs(beta_l_new));
          double l1_t_cand_l = lam2 * arma::accu(arma::abs(Theta_l_new));

          // Instant O(1) candidate penalty check via delta subtraction/addition
          double total_penalty_cand = total_penalty
          - per_pred_old_l + per_pred_cand_l
          - joint_grp_old_l + joint_grp_cand_l
          - mod_grp_old_l + mod_grp_cand_l
          - l1_b_old_l + l1_b_cand_l
          - l1_t_old_l + l1_t_cand_l;

          arma::vec eta_cand = eta_neg_l + compute_block_eta_fast(X_l, beta_l_new, ZTheta_l_cand);
          double loss_cand = (family == "gaussian") ? compute_gaussian_loss(y, eta_cand)
            : compute_logistic_loss(y, eta_cand);

          double obj_cand = loss_cand + total_penalty_cand;

          if (obj_cand <= obj_curr + 1e-12) {

            // Commit delta changes permanently to global tracked metrics
            total_per_pred += (per_pred_cand_l - per_pred_old_l);
            joint_grp_vec(l) = joint_grp_cand_l;
            total_joint_grp += (joint_grp_cand_l - joint_grp_old_l);
            mod_grp_vec(l) = mod_grp_cand_l;
            total_mod_grp += (mod_grp_cand_l - mod_grp_old_l);
            total_l1_b += (l1_b_cand_l - l1_b_old_l);
            total_l1_t += (l1_t_cand_l - l1_t_old_l);

            total_penalty = total_penalty_cand;

            beta2.elem(idx_l) = beta2_l_cand;
            col_sums_Theta2.elem(idx_l) = col_sums_Theta2_l_cand;

            // Commit candidate ZTheta slice to global cache
            ZTheta_l = ZTheta_l_cand;
            ZTheta.cols(idx_l) = ZTheta_l_cand;
            eta_l_current = eta_cand;
            break;
          }
          t_l *= bt_factor;
        }

        // This is a potential source of error. The algorithm still accepts new beta, Theta even if
        // backtracking fails however no caching is updated (due to scope difference this requires some
        // rewriting of the code. However, this is unlikely to happen given that backtracking will eventually
        // leads to a small enough t). Still, this still requires fixing in future versions.

        beta_l_tilde = beta_l_new;
        Theta_l_tilde = Theta_l_new;

        double delta_in = std::max(
          arma::max(arma::abs(beta_l_tilde - beta_l_old_in)),
          arma::max(arma::vectorise(arma::abs(Theta_l_tilde - Theta_l_old_in)))
        );

        if (delta_in < tol_in) break;
      }

      // Accept block parameter update
      beta.elem(idx_l) = beta_l_tilde;
      Theta.cols(idx_l) = Theta_l_tilde;

      // Sync global linear predictor using cached ZTheta_l
      eta = eta_neg_l + compute_block_eta_fast(X_l, beta_l_tilde, ZTheta_l);
    }

    // Check outer convergence
    double current_loss = (family == "gaussian") ? compute_gaussian_loss(y, eta)
      : compute_logistic_loss(y, eta);
    obj_path(iter_out) = current_loss + total_penalty;

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
