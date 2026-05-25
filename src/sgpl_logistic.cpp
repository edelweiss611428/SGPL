// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;


// HELPER FUNCTIONS

// soft threshold (arma::vec and arma::mat)
inline arma::vec soft_thresh_vec(const arma::vec& x, double t) {
  arma::vec out(x.n_elem);
  for (unsigned int i = 0; i < x.n_elem; i++) {
    double xi = x(i);
    if (xi > t) {
      out(i) = xi - t;
    } else if (xi < -t) {
      out(i) = xi + t;
    } else {
      out(i) = 0.0;
    }
  }
  return out;
}

inline arma::mat soft_thresh_mat(const arma::mat& x, double t) {
  arma::mat out(x.n_rows, x.n_cols);
  for (unsigned int i = 0; i < x.n_rows; i++) {
    for (unsigned int j = 0; j < x.n_cols; j++) {
      double xi = x(i, j);
      if (xi > t) {
        out(i, j) = xi - t;
      } else if (xi < -t) {
        out(i, j) = xi + t;
      } else {
        out(i, j) = 0.0;
      }
    }
  }
  return out;
}

// block_soft
inline arma::vec block_soft(const arma::vec& v, double t) {
  double nv = 0.0;
  for (unsigned int i = 0; i < v.n_elem; i++) {
    nv += v(i) * v(i);
  }
  nv = std::sqrt(nv);
  arma::vec out(v.n_elem);

  if (!std::isfinite(nv) || nv <= t) {
    out.zeros();
    return out;
  }

  double mult = 1.0 - t / nv;
  for (unsigned int i = 0; i < v.n_elem; i++) {
    out(i) = v(i) * mult;
  }
  return out;
}

// PREDICTION

inline arma::vec predict_sgpl_cpp(
    const arma::mat& X, // n x p
    const arma::mat& Z, // n x K
    const arma::vec& beta, // p x 1
    const arma::mat& Theta // K x p
) {

  arma::mat M = Z * Theta;              // n x p

  // add beta to each row of M
  M.each_row() += beta.t();

  // elementwise product + row sums
  return arma::sum(X % M, 1);
}


// OBJECTIVE (Logistic Loss + Penalties)
double objective_sgpl_logistic_cpp(
    const arma::mat& X,
    const arma::mat& Z,
    const arma::vec& y,
    const arma::vec& beta,
    const arma::mat& Theta,
    double lambda,
    double alpha,
    const arma::uvec& groups_x,
    const arma::uvec& groups_z
) {
  int n = X.n_rows;
  int p = X.n_cols;
  int K = Z.n_cols;

  unsigned int L = groups_x.max();
  unsigned int G = groups_z.max();

  arma::vec pl(L, fill::zeros);
  arma::vec pg(G, fill::zeros);

  for (unsigned int i = 0; i < groups_x.n_elem; i++) {
    pl(groups_x(i) - 1) += 1.0;
  }
  for (unsigned int i = 0; i < groups_z.n_elem; i++) {
    pg(groups_z(i) - 1) += 1.0;
  }

  double lam1 = lambda * (1.0 - alpha);
  double lam2 = lambda * alpha;

  // Logistic loss calculation
  arma::vec eta = predict_sgpl_cpp(X, Z, beta, Theta);
  double logloss = 0.0;
  for (int i = 0; i < n; i++) {
    // Numerically stable evaluation of logistic loss
    if (eta(i) > 0) {
      logloss += (1.0 - y(i)) * eta(i) + std::log1p(std::exp(-eta(i)));
    } else {
      logloss += -y(i) * eta(i) + std::log1p(std::exp(eta(i)));
    }
  }
  logloss /= n;

  arma::mat Theta2 = arma::square(Theta);

  // Penalties
  double per_pred = lam1 * arma::accu(arma::sqrt(arma::square(beta) + arma::sum(Theta2, 0).t()));
  double joint_grp = 0.0;

  for (unsigned int l = 1; l <= L; l++) {
    arma::uvec idx = arma::find(groups_x == l);
    if (idx.n_elem > 0) {
      arma::vec svec = arma::square(beta.elem(idx)) + arma::sum(Theta2.cols(idx), 0).t();
      joint_grp += std::sqrt(pl(l - 1)) * std::sqrt(arma::accu(svec));
    }
  }
  joint_grp *= lam1;

  double z_grp = 0.0;
  double w_denom = std::sqrt(1.0 + K);

  for (unsigned int l = 1; l <= L; l++) {
    arma::uvec idx_l = arma::find(groups_x == l);
    if (idx_l.n_elem == 0) continue;

    for (unsigned int g = 1; g <= G; g++) {
      arma::uvec idx_g = arma::find(groups_z == g);
      if (idx_g.n_elem == 0) continue;
      arma::mat sub = Theta2.submat(idx_g, idx_l);
      z_grp += std::sqrt(pg(g - 1)) / w_denom * std::sqrt(arma::accu(sub));
    }
  }
  z_grp *= lam1;

  double l1_b = lam2 * arma::accu(arma::abs(beta));
  double l1_t = lam2 * arma::accu(arma::abs(Theta));

  return logloss + per_pred + joint_grp + z_grp + l1_b + l1_t;
}

// sgpl_logistic_fit_rcpp

// [[Rcpp::export]]
Rcpp::List sgpl_logistic_fit_rcpp(
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
    Rcpp::Nullable<arma::mat> Theta_init_r = R_NilValue
) {
  int n = X.n_rows;
  int p = X.n_cols;
  int K = Z.n_cols;

  if ((int)Z.n_rows != n) stop("X and Z must have same number of rows");
  if ((int)y.n_elem != n) stop("length(y) must equal nrow(X)");

  arma::uvec groups_x;
  arma::uvec groups_z;

  if (groups_x_r.isNull()) {
    groups_x.set_size(p);
    for (int j = 0; j < p; j++) groups_x(j) = j + 1;
  } else {
    groups_x = Rcpp::as<arma::uvec>(groups_x_r);
    if ((int)groups_x.n_elem != p) stop("groups_x has incorrect length");
  }

  if (groups_z_r.isNull()) {
    groups_z.set_size(K);
    for (int k = 0; k < K; k++) groups_z(k) = k + 1;
  } else {
    groups_z = Rcpp::as<arma::uvec>(groups_z_r);
    if ((int)groups_z.n_elem != K) stop("groups_z has incorrect length");
  }

  arma::vec beta;
  if (beta_init_r.isNull()) {
    beta.zeros(p);
  } else {
    beta = Rcpp::as<arma::vec>(beta_init_r);
    if ((int)beta.n_elem != p) stop("beta_init has incorrect length");
  }

  arma::mat Theta;
  if (Theta_init_r.isNull()) {
    Theta.zeros(K, p);
  } else {
    Theta = Rcpp::as<arma::mat>(Theta_init_r);
    if ((int)Theta.n_rows != K || (int)Theta.n_cols != p) stop("Theta_init has incorrect dimensions");
  }

  unsigned int L = groups_x.max();
  unsigned int G = groups_z.max();

  arma::vec pl(L, fill::zeros);
  arma::vec pg(G, fill::zeros);

  for (unsigned int i = 0; i < groups_x.n_elem; i++) pl(groups_x(i) - 1) += 1.0;
  for (unsigned int i = 0; i < groups_z.n_elem; i++) pg(groups_z(i) - 1) += 1.0;

  double lam1 = lambda * (1.0 - alpha);
  double lam2 = lambda * alpha;

  double t_init;
  if (t_init_r.isNull()) {
    arma::rowvec x_norm2 = arma::sum(arma::square(X), 0);
    arma::rowvec z_norm2 = arma::sum(arma::square(Z), 0);
    double max_xnorm2 = x_norm2.max();
    double max_znorm2 = z_norm2.max();
    double Lf = (max_xnorm2 * (1.0 + max_znorm2)) / X.n_rows;
    t_init = 4.0 / Lf; // Adjusted upper bound for logistic loss curvature (Hessian <= 1/4)
  } else {
    t_init = Rcpp::as<double>(t_init_r);
  }

  arma::vec obj_path(max_iter_out);
  bool converged = false;
  int n_iter = max_iter_out;

  // Outer loop
  for (int iter_out = 0; iter_out < max_iter_out; iter_out++) {
    arma::vec beta_old = beta;
    arma::mat Theta_old = Theta;

    for (unsigned int l = 1; l <= L; l++) {
      std::vector<unsigned int> idx_l_vec;
      for (int j = 0; j < p; j++) {
        if (groups_x(j) == l) idx_l_vec.push_back(j);
      }

      int p_l = idx_l_vec.size();
      if (p_l == 0) continue;
      arma::mat X_l(n, p_l);
      for (int jj = 0; jj < p_l; jj++) {
        int j = idx_l_vec[jj];
        X_l.col(jj) = X.col(j);
      }

      // Compute linear predictor excluding block l
      arma::vec eta_neg_l(n, fill::zeros);
      for (unsigned int lp = 1; lp <= L; lp++) {
        if (lp == l) continue;
        for (int j = 0; j < p; j++) {
          if (groups_x(j) == lp) {
            arma::vec eta_j = beta(j) + Z * Theta.col(j);
            eta_neg_l += X.col(j) % eta_j;
          }
        }
      }

      // KKT screening under logistic generalized residual: r = y - prob
      if (use_screen) {
        arma::vec prob_neg_l = 1.0 / (1.0 + arma::exp(-eta_neg_l));
        arma::vec r_neg_l = y - prob_neg_l;

        arma::vec g_l = (X_l.t() * r_neg_l) / n;
        arma::mat H_l(K, p_l);
        for (int jj = 0; jj < p_l; jj++) {
          H_l.col(jj) = (Z.t() * (X_l.col(jj) % r_neg_l)) / n;
        }

        arma::vec vv(g_l.n_elem + H_l.n_elem);
        for (unsigned int i = 0; i < g_l.n_elem; i++) vv(i) = g_l(i);
        unsigned int ctr = g_l.n_elem;
        for (int j = 0; j < H_l.n_cols; j++) {
          for (unsigned int k = 0; k < H_l.n_rows; k++) {
            vv(ctr) = H_l(k, j);
            ctr++;
          }
        }

        vv = soft_thresh_vec(vv, lam2);
        double nv = std::sqrt(arma::accu(arma::square(vv)));

        if (nv <= std::sqrt((double)p_l) * lam1) {
          for (int jj = 0; jj < p_l; jj++) {
            int j = idx_l_vec[jj];
            beta(j) = 0.0;
            Theta.col(j).zeros();
          }
          continue;
        }
      }

      // Init block
      arma::vec beta_l_tilde(p_l);
      arma::mat Theta_l_tilde(K, p_l);
      for (int jj = 0; jj < p_l; jj++) {
        int j = idx_l_vec[jj];
        beta_l_tilde(jj) = beta(j);
        Theta_l_tilde.col(jj) = Theta.col(j);
      }

      double t_l = t_init;

      // Inner loop
      for (int iter_in = 0; iter_in < max_iter_in; iter_in++) {
        arma::vec beta_l_old = beta_l_tilde;
        arma::mat Theta_l_old = Theta_l_tilde;

        // Construct full block eta and generalized residuals
        arma::vec eta_l = eta_neg_l;
        for (int jj = 0; jj < p_l; jj++) {
          arma::vec eta_j = beta_l_tilde(jj) + Z * Theta_l_tilde.col(jj);
          eta_l += X_l.col(jj) % eta_j;
        }
        arma::vec prob_l = 1.0 / (1.0 + arma::exp(-eta_l));
        arma::vec r_l = y - prob_l; // Generalized working residual

        // Logistic Gradients
        arma::vec gb = - (X_l.t() * r_l) / n;
        arma::mat gT(K, p_l);
        for (int jj = 0; jj < p_l; jj++) {
          gT.col(jj) = - (Z.t() * (X_l.col(jj) % r_l)) / n;
        }

        bool bt_ok = false;
        arma::vec beta_l_new;
        arma::mat Theta_l_new;

        // Backtracking Proximal Gradient Step
        for (int bt = 0; bt < bt_max; bt++) {
          arma::vec beta_try = beta_l_tilde - t_l * gb;
          arma::mat Theta_try = Theta_l_tilde - t_l * gT;

          beta_try = soft_thresh_vec(beta_try, t_l * lam2);
          Theta_try = soft_thresh_mat(Theta_try, t_l * lam2);

          // Predictor block shrinkage
          for (int jj = 0; jj < p_l; jj++) {
            arma::vec vv(1 + K);
            vv(0) = beta_try(jj);
            for (int k = 0; k < K; k++) vv(k + 1) = Theta_try(k, jj);

            vv = block_soft(vv, t_l * lam1);
            beta_try(jj) = vv(0);
            for (int k = 0; k < K; k++) Theta_try(k, jj) = vv(k + 1);
          }

          // Modifier group shrinkage
          double w_denom = std::sqrt(1.0 + K);
          for (unsigned int g = 1; g <= G; g++) {
            std::vector<unsigned int> idxg;
            for (int k = 0; k < K; k++) {
              if (groups_z(k) == g) idxg.push_back(k);
            }
            int ng = idxg.size();
            if (ng == 0) continue;

            arma::vec vv(ng * p_l);
            int ctr = 0;
            for (int jj = 0; jj < p_l; jj++) {
              for (int kk = 0; kk < ng; kk++) {
                vv(ctr) = Theta_try(idxg[kk], jj);
                ctr++;
              }
            }

            double w = std::sqrt(pg(g - 1)) / w_denom;
            vv = block_soft(vv, t_l * lam1 * w);
            ctr = 0;
            for (int jj = 0; jj < p_l; jj++) {
              for (int kk = 0; kk < ng; kk++) {
                Theta_try(idxg[kk], jj) = vv(ctr);
                ctr++;
              }
            }
          }

          // Full group shrinkage
          arma::vec vv(p_l + K * p_l);
          for (int jj = 0; jj < p_l; jj++) vv(jj) = beta_try(jj);
          int ctr = p_l;
          for (int jj = 0; jj < p_l; jj++) {
            for (int k = 0; k < K; k++) {
              vv(ctr) = Theta_try(k, jj);
              ctr++;
            }
          }

          vv = block_soft(vv, t_l * lam1 * std::sqrt((double)p_l));
          for (int jj = 0; jj < p_l; jj++) beta_try(jj) = vv(jj);
          ctr = p_l;
          for (int jj = 0; jj < p_l; jj++) {
            for (int k = 0; k < K; k++) {
              Theta_try(k, jj) = vv(ctr);
              ctr++;
            }
          }

          beta_l_new = beta_try;
          Theta_l_new = Theta_try;

          // Backtracking Objective evaluation
          arma::vec beta_cand = beta;
          arma::mat Theta_cand = Theta;
          for (int jj = 0; jj < p_l; jj++) {
            int j = idx_l_vec[jj];
            beta_cand(j) = beta_l_new(jj);
            Theta_cand.col(j) = Theta_l_new.col(jj);
          }

          double obj_cand = objective_sgpl_logistic_cpp(X, Z, y, beta_cand, Theta_cand, lambda, alpha, groups_x, groups_z);

          arma::vec beta_curr = beta;
          arma::mat Theta_curr = Theta;
          for (int jj = 0; jj < p_l; jj++) {
            int j = idx_l_vec[jj];
            beta_curr(j) = beta_l_tilde(jj);
            Theta_curr.col(j) = Theta_l_tilde.col(jj);
          }

          double obj_curr = objective_sgpl_logistic_cpp(X, Z, y, beta_curr, Theta_curr, lambda, alpha, groups_x, groups_z);

          if (obj_cand <= obj_curr + 1e-12) {
            bt_ok = true;
            break;
          }
          t_l *= bt_factor;
        }

        beta_l_tilde = beta_l_new;
        Theta_l_tilde = Theta_l_new;

        double delta_in = 0.0;
        for (int jj = 0; jj < p_l; jj++) {
          double d = std::abs(beta_l_tilde(jj) - beta_l_old(jj));
          if (d > delta_in) delta_in = d;
        }
        for (int jj = 0; jj < p_l; jj++) {
          for (int k = 0; k < K; k++) {
            double d = std::abs(Theta_l_tilde(k, jj) - Theta_l_old(k, jj));
            if (d > delta_in) delta_in = d;
          }
        }

        if (delta_in < tol_in) break;
      }

      // Commit changes to the main structures
      for (int jj = 0; jj < p_l; jj++) {
        int j = idx_l_vec[jj];
        beta(j) = beta_l_tilde(jj);
        Theta.col(j) = Theta_l_tilde.col(jj);
      }
    }

    // Check outer convergence
    obj_path(iter_out) = objective_sgpl_logistic_cpp(X, Z, y, beta, Theta, lambda, alpha, groups_x, groups_z);

    double delta_out = 0.0;
    for (int j = 0; j < p; j++) {
      double d = std::abs(beta(j) - beta_old(j));
      if (d > delta_out) delta_out = d;
    }
    for (int j = 0; j < p; j++) {
      for (int k = 0; k < K; k++) {
        double d = std::abs(Theta(k, j) - Theta_old(k, j));
        if (d > delta_out) delta_out = d;
      }
    }

    if (verbose && ((iter_out + 1) % 10 == 0 || iter_out == 0)) {
      Rcout << "Outer " << (iter_out + 1) << " obj=" << obj_path(iter_out) << " delta=" << delta_out << "\n";
    }

    if (delta_out < tol_out && iter_out > 0) {
      converged = true;
      n_iter = iter_out + 1;
      break;
    }
  }

  arma::vec final_obj(n_iter);
  for (int i = 0; i < n_iter; i++) final_obj(i) = obj_path(i);

  return Rcpp::List::create(
    Named("beta") = beta,
    Named("Theta") = Theta,
    Named("obj_path") = final_obj,
    Named("converged") = converged,
    Named("n_iter") = n_iter,
    Named("lambda") = lambda,
    Named("alpha") = alpha,
    Named("groups_x") = groups_x,
    Named("groups_z") = groups_z
  );
}
