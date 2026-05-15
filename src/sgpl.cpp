// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// helper functions

// soft threshold

arma::vec soft_thresh_vec(const arma::vec& x, double t) {

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

arma::mat soft_thresh_mat(const arma::mat& x, double t) {

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

// ================================================================
// block soft
// ================================================================

arma::vec block_soft(
    const arma::vec& v,
    double t
) {

  double nv = 0.0;

  for (unsigned int i = 0; i < v.n_elem; i++) {
    nv += v(i) * v(i);
  }

  nv = std::sqrt(nv);

  arma::vec out(v.n_elem);

  if (!arma::is_finite(nv) || nv <= t) {

    out.zeros();

    return out;
  }

  double mult = 1.0 - t / nv;

  for (unsigned int i = 0; i < v.n_elem; i++) {
    out(i) = v(i) * mult;
  }

  return out;
}

// ================================================================
// prediction
// ================================================================

arma::vec predict_sgpl_cpp(
    const arma::mat& X,
    const arma::mat& Z,
    const arma::vec& beta,
    const arma::mat& Theta
) {

  int n = X.n_rows;
  int p = X.n_cols;

  arma::vec out(n);
  out.zeros();

  for (int i = 0; i < n; i++) {

    double val = 0.0;

    for (int j = 0; j < p; j++) {

      double eta = beta(j);

      for (unsigned int k = 0; k < Z.n_cols; k++) {
        eta += Z(i, k) * Theta(k, j);
      }

      val += X(i, j) * eta;
    }

    out(i) = val;
  }

  return out;
}

// ================================================================
// objective
// ================================================================

double objective_sgpl_cpp(
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

  arma::vec pl(L);
  arma::vec pg(G);

  pl.zeros();
  pg.zeros();

  for (unsigned int i = 0; i < groups_x.n_elem; i++) {
    pl(groups_x(i) - 1) += 1.0;
  }

  for (unsigned int i = 0; i < groups_z.n_elem; i++) {
    pg(groups_z(i) - 1) += 1.0;
  }

  double lam1 = lambda * (1.0 - alpha);
  double lam2 = lambda * alpha;

  arma::vec pred =
    predict_sgpl_cpp(X, Z, beta,Theta);

  double rss = 0.0;

  for (int i = 0; i < n; i++) {

    double d = y(i) - pred(i);

    rss += d * d;
  }

  rss /= (2.0 * n);

  double per_pred = 0.0;

  for (int j = 0; j < p; j++) {

    double s = beta(j) * beta(j);

    for (int k = 0; k < K; k++) {
      s += Theta(k, j) * Theta(k, j);
    }

    per_pred += std::sqrt(s);
  }

  per_pred *= lam1;

  double joint_grp = 0.0;

  for (unsigned int l = 1; l <= L; l++) {

    double s = 0.0;

    for (int j = 0; j < p; j++) {

      if (groups_x(j) == l) {

        s += beta(j) * beta(j);

        for (int k = 0; k < K; k++) {
          s += Theta(k, j) * Theta(k, j);
        }
      }
    }

    joint_grp +=
      std::sqrt(pl(l - 1)) *
      std::sqrt(s);
  }

  joint_grp *= lam1;

  double z_grp = 0.0;

  double w_denom = std::sqrt(1.0 + K);

  for (unsigned int l = 1; l <= L; l++) {

    for (unsigned int g = 1; g <= G; g++) {

      double s = 0.0;

      for (int j = 0; j < p; j++) {

        if (groups_x(j) == l) {

          for (int k = 0; k < K; k++) {

            if (groups_z(k) == g) {
              s += Theta(k, j) * Theta(k, j);
            }
          }
        }
      }

      z_grp +=
        std::sqrt(pg(g - 1)) /
          w_denom *
            std::sqrt(s);
    }
  }

  z_grp *= lam1;

  double l1_b = 0.0;
  double l1_t = 0.0;

  for (int j = 0; j < p; j++) {
    l1_b += std::abs(beta(j));
  }

  for (int k = 0; k < K; k++) {
    for (int j = 0; j < p; j++) {
      l1_t += std::abs(Theta(k, j));
    }
  }

  l1_b *= lam2;
  l1_t *= lam2;

  return rss +
    per_pred +
    joint_grp +
    z_grp +
    l1_b +
    l1_t;
}

// ================================================================
// main fit
// ================================================================

// [[Rcpp::export]]

Rcpp::List sgpl_fit_rcpp(
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

  // ============================================================
  // dimension checks
  // ============================================================

  if ((int)Z.n_rows != n) {
    stop("X and Z must have same number of rows");
  }

  if ((int)y.n_elem != n) {
    stop("length(y) must equal nrow(X)");
  }

  arma::uvec groups_x;
  arma::uvec groups_z;

  if (groups_x_r.isNull()) {

    groups_x.set_size(p);

    for (int j = 0; j < p; j++) {
      groups_x(j) = j + 1;
    }

  } else {

    groups_x =
      Rcpp::as<arma::uvec>(groups_x_r);

    if ((int)groups_x.n_elem != p) {
      stop("groups_x has incorrect length");
    }
  }

  if (groups_z_r.isNull()) {

    groups_z.set_size(K);

    for (int k = 0; k < K; k++) {
      groups_z(k) = k + 1;
    }

  } else {

    groups_z = Rcpp::as<arma::uvec>(groups_z_r);

    if ((int)groups_z.n_elem != K) {
      stop("groups_z has incorrect length");
    }
  }

  arma::vec beta;

  if (beta_init_r.isNull()) {

    beta.zeros(p);

  } else {

    beta =
      Rcpp::as<arma::vec>(beta_init_r);

    if ((int)beta.n_elem != p) {
      stop("beta_init has incorrect length");
    }
  }

  arma::mat Theta;

  if (Theta_init_r.isNull()) {

    Theta.zeros(K, p);

  } else {

    Theta =
      Rcpp::as<arma::mat>(Theta_init_r);

    if ((int)Theta.n_rows != K ||
        (int)Theta.n_cols != p) {
      stop("Theta_init has incorrect dimensions");
    }
  }

  unsigned int L = groups_x.max();
  unsigned int G = groups_z.max();

  arma::vec pl(L);
  arma::vec pg(G);

  pl.zeros();
  pg.zeros();

  for (unsigned int i = 0; i < groups_x.n_elem; i++) {
    pl(groups_x(i) - 1) += 1.0;
  }

  for (unsigned int i = 0; i < groups_z.n_elem; i++) {
    pg(groups_z(i) - 1) += 1.0;
  }

  double lam1 = lambda * (1.0 - alpha);
  double lam2 = lambda * alpha;

  double t_init;

  if (t_init_r.isNull()) {

    double max_xnorm2 = 0.0;
    double max_znorm2 = 0.0;

    for (int j = 0; j < p; j++) {

      double s = 0.0;

      for (int i = 0; i < n; i++) {
        s += X(i, j) * X(i, j);
      }

      if (s > max_xnorm2) {
        max_xnorm2 = s;
      }
    }

    for (int k = 0; k < K; k++) {

      double s = 0.0;

      for (int i = 0; i < n; i++) {
        s += Z(i, k) * Z(i, k);
      }

      if (s > max_znorm2) {
        max_znorm2 = s;
      }
    }

    double Lf =
      max_xnorm2 *
      (1.0 + max_znorm2) / n;

    t_init = 1.0 / Lf;

  } else {

    t_init =
      Rcpp::as<double>(t_init_r);
  }

  arma::vec obj_path(max_iter_out);

  bool converged = false;
  int n_iter = max_iter_out;

  // ============================================================
  // OUTER LOOP
  // ============================================================

  for (int iter_out = 0;
       iter_out < max_iter_out;
       iter_out++) {

    arma::vec beta_old = beta;
    arma::mat Theta_old = Theta;

    for (unsigned int l = 1;
         l <= L;
         l++) {

      // ========================================================
      // group indices
      // ========================================================

      std::vector<unsigned int> idx_l_vec;

      for (int j = 0; j < p; j++) {
        if (groups_x(j) == l) {
          idx_l_vec.push_back(j);
        }
      }

      int p_l = idx_l_vec.size();

      arma::mat X_l(n, p_l);

      for (int jj = 0; jj < p_l; jj++) {

        int j = idx_l_vec[jj];

        for (int i = 0; i < n; i++) {
          X_l(i, jj) = X(i, j);
        }
      }

      // ========================================================
      // partial residual
      // ========================================================

      arma::vec r_neg_l = y;

      for (unsigned int lp = 1;
           lp <= L;
           lp++) {

        if (lp == l) continue;

        for (int j = 0; j < p; j++) {

          if (groups_x(j) == lp) {

            for (int i = 0; i < n; i++) {

              double eta = beta(j);

              for (int k = 0; k < K; k++) {
                eta += Z(i, k) * Theta(k, j);
              }

              r_neg_l(i) -= X(i, j) * eta;
            }
          }
        }
      }

      // ========================================================
      // KKT screen
      // ========================================================

      if (use_screen) {

        arma::vec g_l(p_l);
        arma::mat H_l(K, p_l);

        for (int jj = 0; jj < p_l; jj++) {

          double s = 0.0;

          for (int i = 0; i < n; i++) {
            s += X_l(i, jj) * r_neg_l(i);
          }

          g_l(jj) = s / n;

          for (int k = 0; k < K; k++) {

            double sk = 0.0;

            for (int i = 0; i < n; i++) {
              sk +=
                Z(i, k) *
                X_l(i, jj) *
                r_neg_l(i);
            }

            H_l(k, jj) = sk / n;
          }
        }

        arma::vec vv(
            g_l.n_elem + H_l.n_elem
        );

        for (unsigned int i = 0;
             i < g_l.n_elem;
             i++) {
          vv(i) = g_l(i);
        }

        unsigned int ctr = g_l.n_elem;

        for (unsigned int j = 0;
             j < H_l.n_cols;
             j++) {

          for (unsigned int k = 0;
               k < H_l.n_rows;
               k++) {

            vv(ctr) = H_l(k, j);
            ctr++;
          }
        }

        vv = soft_thresh_vec(vv, lam2);

        double nv = 0.0;

        for (unsigned int i = 0;
             i < vv.n_elem;
             i++) {
          nv += vv(i) * vv(i);
        }

        nv = std::sqrt(nv);

        if (nv <= std::sqrt((double)p_l) * lam1) {

          for (int jj = 0; jj < p_l; jj++) {

            int j = idx_l_vec[jj];

            beta(j) = 0.0;

            for (int k = 0; k < K; k++) {
              Theta(k, j) = 0.0;
            }
          }

          continue;
        }
      }

      // ========================================================
      // init block
      // ========================================================

      arma::vec beta_l_tilde(p_l);
      arma::mat Theta_l_tilde(K, p_l);

      for (int jj = 0; jj < p_l; jj++) {

        int j = idx_l_vec[jj];

        beta_l_tilde(jj) = beta(j);

        for (int k = 0; k < K; k++) {
          Theta_l_tilde(k, jj) =
            Theta(k, j);
        }
      }

      double t_l = t_init;

      // inner loop

      for (int iter_in = 0;
           iter_in < max_iter_in;
           iter_in++) {

        arma::vec beta_l_old =
          beta_l_tilde;

        arma::mat Theta_l_old =
          Theta_l_tilde;

        // ======================================================
        // block residual
        // ======================================================

        arma::vec r_l = r_neg_l;

        for (int jj = 0; jj < p_l; jj++) {

          for (int i = 0; i < n; i++) {

            double eta =
              beta_l_tilde(jj);

            for (int k = 0; k < K; k++) {
              eta +=
                Z(i, k) *
                Theta_l_tilde(k, jj);
            }

            r_l(i) -=
              X_l(i, jj) * eta;
          }
        }

        // gradient

        arma::vec gb(p_l);
        arma::mat gT(K, p_l);

        for (int jj = 0; jj < p_l; jj++) {

          double s = 0.0;

          for (int i = 0; i < n; i++) {
            s += X_l(i, jj) * r_l(i);
          }

          gb(jj) = -s / n;

          for (int k = 0; k < K; k++) {

            double sk = 0.0;

            for (int i = 0; i < n; i++) {
              sk +=
                Z(i, k) *
                X_l(i, jj) *
                r_l(i);
            }

            gT(k, jj) = -sk / n;
          }
        }

        bool bt_ok = false;

        arma::vec beta_l_new;
        arma::mat Theta_l_new;

        // backtracking

        for (int bt = 0;
             bt < bt_max;
             bt++) {

          arma::vec beta_try =
            beta_l_tilde - t_l * gb;

          arma::mat Theta_try =
            Theta_l_tilde - t_l * gT;

          // ====================================================
          // soft threshold
          // ====================================================

          beta_try =
            soft_thresh_vec(
              beta_try,
              t_l * lam2
            );

          Theta_try =
            soft_thresh_mat(
              Theta_try,
              t_l * lam2
            );

          // predictor block shrinkage

          for (int jj = 0; jj < p_l; jj++) {

            arma::vec vv(1 + K);

            vv(0) = beta_try(jj);

            for (int k = 0; k < K; k++) {
              vv(k + 1) =Theta_try(k, jj);
            }

            vv = block_soft(vv, t_l * lam1);

            beta_try(jj) = vv(0);

            for (int k = 0; k < K; k++) {
              Theta_try(k, jj) = vv(k + 1);
            }
          }

          // modifier group shrinkage

          double w_denom = std::sqrt(1.0 + K);

          for (unsigned int g = 1; g <= G; g++) {

            std::vector<unsigned int> idxg;

            for (int k = 0; k < K; k++) {
              if (groups_z(k) == g) {
                idxg.push_back(k);
              }
            }

            int ng = idxg.size();

            arma::vec vv(ng * p_l);

            int ctr = 0;

            for (int jj = 0; jj < p_l; jj++) {

              for (int kk = 0; kk < ng; kk++) {

                vv(ctr) = Theta_try( idxg[kk], jj);
                ctr++;
              }
            }

            double w = std::sqrt(pg(g - 1)) / w_denom;

            vv = block_soft(vv, t_l * lam1 * w);

            ctr = 0;

            for (int jj = 0;jj < p_l;jj++) {

              for (int kk = 0; kk < ng; kk++) {

                Theta_try( idxg[kk], jj) = vv(ctr);

                ctr++;
              }
            }
          }

          // ====================================================
          // full group shrink
          // ====================================================

          arma::vec vv(
              p_l + K * p_l
          );

          for (int jj = 0; jj < p_l; jj++) {
            vv(jj) = beta_try(jj);
          }

          int ctr = p_l;

          for (int jj = 0; jj < p_l; jj++) {

            for (int k = 0; k < K; k++) {

              vv(ctr) = Theta_try(k, jj);

              ctr++;
            }
          }

          vv =
            block_soft(
              vv,
              t_l *
                lam1 *
                std::sqrt((double)p_l)
            );

          for (int jj = 0;
               jj < p_l;
               jj++) {
            beta_try(jj) = vv(jj);
          }

          ctr = p_l;

          for (int jj = 0;
               jj < p_l;
               jj++) {

            for (int k = 0;
                 k < K;
                 k++) {

              Theta_try(k, jj) =
                vv(ctr);

              ctr++;
            }
          }

          beta_l_new = beta_try;
          Theta_l_new = Theta_try;

          // ====================================================
          // candidate objective
          // ====================================================

          arma::vec beta_cand = beta;
          arma::mat Theta_cand = Theta;

          for (int jj = 0; jj < p_l; jj++) {

            int j = idx_l_vec[jj];

            beta_cand(j) = beta_l_new(jj);

            for (int k = 0; k < K; k++) {

              Theta_cand(k, j) = Theta_l_new(k, jj);
            }
          }

          double obj_cand =
            objective_sgpl_cpp(
              X, Z, y,
              beta_cand, Theta_cand,
              lambda, alpha,
              groups_x, groups_z
            );

          arma::vec beta_curr = beta;
          arma::mat Theta_curr = Theta;

          for (int jj = 0; jj < p_l; jj++) {

            int j = idx_l_vec[jj];

            beta_curr(j) = beta_l_tilde(jj);

            for (int k = 0; k < K; k++) {

              Theta_curr(k, j) = Theta_l_tilde(k, jj);
            }
          }

          double obj_curr =
            objective_sgpl_cpp(
              X, Z, y,
              beta_curr, Theta_curr,
              lambda, alpha,
              groups_x, groups_z
            );

          if (obj_cand <= obj_curr + 1e-12) {
            bt_ok = true;
            break;
          }

          t_l *= bt_factor;
        }

        beta_l_tilde = beta_l_new;

        Theta_l_tilde = Theta_l_new;

        double delta_in = 0.0;

        for (int jj = 0; jj < p_l;jj++) {

          double d =
            std::abs(
              beta_l_tilde(jj) -
                beta_l_old(jj)
            );

          if (d > delta_in) {
            delta_in = d;
          }
        }

        for (int jj = 0; jj < p_l; jj++) {

          for (int k = 0;k < K; k++) {

            double d = std::abs(Theta_l_tilde(k, jj) - Theta_l_old(k, jj));

            if (d > delta_in) {
              delta_in = d;
            }
          }
        }

        if (delta_in < tol_in) {
          break;
        }
      }

      // ========================================================
      // accept update
      // ========================================================

      for (int jj = 0; jj < p_l; jj++) {

        int j = idx_l_vec[jj];

        beta(j) = beta_l_tilde(jj);

        for (int k = 0; k < K; k++) {

          Theta(k, j) = Theta_l_tilde(k, jj);
        }
      }
    }

    // ==========================================================
    // convergence
    // ==========================================================

    obj_path(iter_out) = objective_sgpl_cpp(X, Z, y, beta, Theta, lambda, alpha, groups_x,
                         groups_z);

    double delta_out = 0.0;

    for (int j = 0; j < p; j++) {

      double d = std::abs(beta(j) - beta_old(j));

      if (d > delta_out) {
        delta_out = d;
      }
    }

    for (int j = 0; j < p; j++) {

      for (int k = 0; k < K; k++) {

        double d =
          std::abs(
            Theta(k, j) -
              Theta_old(k, j)
          );

        if (d > delta_out) {
          delta_out = d;
        }
      }
    }

    if (verbose &&
        ((iter_out + 1) % 10 == 0 ||
        iter_out == 0)) {

      Rcout
      << "Outer "
      << (iter_out + 1)
      << " obj="
      << obj_path(iter_out)
      << " delta="
      << delta_out
      << "\n";
    }

    if (delta_out < tol_out &&
        iter_out > 0) {

      converged = true;
      n_iter = iter_out + 1;

      break;
    }
  }

  arma::vec final_obj(n_iter);

  for (int i = 0;
       i < n_iter;
       i++) {
    final_obj(i) = obj_path(i);
  }

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



