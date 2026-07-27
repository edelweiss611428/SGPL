// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include "sgpl_helpers.h"

using namespace Rcpp;
using namespace arma;


// [[Rcpp::export]]
double objective_sgpl_cpp(
    const arma::mat& X, const arma::mat& Z, const arma::vec& y,
    const arma::vec& beta, const arma::mat& Theta,
    double lambda, double alpha,
    const arma::uvec& groups_x, const arma::uvec& groups_z,
    std::string family,
    Rcpp::Nullable<arma::vec> status = R_NilValue // Added parameter
) {
  int K = Z.n_cols;
  arma::vec pred = predict_sgpl_cpp(X, Z, beta, Theta);
  double loss;

  if(family == "gaussian"){
    loss = compute_gaussian_loss(y, pred);
  } else if (family == "binomial"){
    loss = compute_logistic_loss(y, pred);
  } else if (family == "cox") {
    if (status.isNull()) stop("status is required for family='cox'");
    loss = compute_cox_loss(y, as<arma::vec>(status), pred, X.n_rows);
  }

  double penalty = compute_sgpl_penalties(beta, Theta, lambda, alpha, groups_x, groups_z, K);
  return loss + penalty;
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
    std::string family = "gaussian",
    Rcpp::Nullable<arma::vec> status = R_NilValue // for Cox
){


  int n = X.n_rows;
  int p = X.n_cols;
  int K = Z.n_cols;

  // for Cox

  if (family == "Cox") {
    if (status.isNull()) {
      stop("status is required for family='Cox'");
    }

    arma::vec status_vec = Rcpp::as<arma::vec>(status);

    if ((int)status_vec.n_elem != n) {
      stop("length(status) must equal nrow(X)");
    }

    arma::uvec mask = (status_vec != 0) % (status_vec != 1);
    arma::uvec invalid = arma::find(mask == 1);
    if (invalid.n_elem > 0) {
      stop("status must be coded as 0/1");
    }
  }

  // Dimension checks

  // Data

  if ((int)Z.n_rows != n) {
    stop("X and Z must have same number of rows");
  }

  if ((int)y.n_elem != n) {
    stop("length(y) must equal nrow(X)");
  }

  // Groups
  arma::uvec groups_x;
  arma::uvec groups_z;

  // If group is null then each feature forms a group

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
    beta = Rcpp::as<arma::vec>(beta_init_r);
    if ((int)beta.n_elem != p) {
      stop("beta_init has incorrect length");
    }
  }

  arma::mat Theta;

  if (Theta_init_r.isNull()) {
    Theta.zeros(K, p);
  } else {
    Theta = Rcpp::as<arma::mat>(Theta_init_r);
    if ((int)Theta.n_rows != K || (int)Theta.n_cols != p) {
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

  double lam1 = lambda * (1.0 - alpha); // redundant - to be removed later
  double lam2 = lambda * alpha;
  double t_init = compute_t_init(X,Z,t_init_r,family);

  arma::vec obj_path(max_iter_out);
  bool converged = false;
  int n_iter = max_iter_out;

  // outer-loop (block coordinate descent)

  for (int iter_out = 0; iter_out < max_iter_out; iter_out++) {
    arma::vec beta_old = beta;
    arma::mat Theta_old = Theta;

    for (int l = 1; l <= L; l++) {
      // group indices
      std::vector<unsigned int> idx_l_vec;

      // future versions consider pre-computing group indexes
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

      // partial residual

      arma::vec r_neg_l = y;

      // future versions consider pre-computation
      for (int lp = 1; lp <= L; lp++) {
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

      // KKT screening (currently developed for gaussian only)

      if (use_screen && family == "gaussian") {

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
              sk += Z(i, k) * X_l(i, jj) * r_neg_l(i);
            }
            H_l(k, jj) = sk / n;
          }
        }

        arma::vec vv(g_l.n_elem + H_l.n_elem);

        for (int i = 0; i < g_l.n_elem;i++) {
          vv(i) = g_l(i);
        }

        unsigned int ctr = g_l.n_elem;

        for (int j = 0; j < H_l.n_cols;j++) {
          for (int k = 0;k < H_l.n_rows;k++) {
            vv(ctr) = H_l(k, j);
            ctr++;
          }
        }

        vv = soft_thresh_vec(vv, lam2);

        double nv = 0.0;

        for (int i = 0;i < vv.n_elem; i++) {
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

      // init block

      arma::vec beta_l_tilde(p_l);
      arma::mat Theta_l_tilde(K, p_l);

      for (int jj = 0; jj < p_l; jj++) {
        int j = idx_l_vec[jj];
        beta_l_tilde(jj) = beta(j);
        for (int k = 0; k < K; k++) {
          Theta_l_tilde(k, jj) = Theta(k, j);
        }
      }

      double t_l = t_init;

      // inner loop

      for (int iter_in = 0; iter_in < max_iter_in; iter_in++) {

        arma::vec beta_l_old = beta_l_tilde;
        arma::mat Theta_l_old = Theta_l_tilde;

        // block residual

        arma::vec eta_l = y - r_neg_l;

        for (int jj = 0; jj < p_l; jj++) {
          arma::vec eta_j = beta_l_tilde(jj) + Z * Theta_l_tilde.col(jj);
          eta_l += X_l.col(jj) % eta_j;
        }

        // Accurately compute gradients using native linear space eta_l vectors
        arma::vec gb = gradient_smooth_loss_beta(X_l, y, eta_l, n, family, status);
        arma::mat gT = gradient_smooth_loss_theta(X_l, Z, y, eta_l, p_l, K, n, family, status);


        bool bt_ok = false;

        arma::vec beta_l_new;
        arma::mat Theta_l_new;

        // backtracking

        for (int bt = 0; bt < bt_max; bt++) {

          arma::vec beta_try = beta_l_tilde - t_l * gb;
          arma::mat Theta_try = Theta_l_tilde - t_l * gT;

          // soft threshold

          beta_try = soft_thresh_vec(beta_try, t_l * lam2);
          Theta_try = soft_thresh_mat(Theta_try, t_l * lam2);

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

          // full group shrink

          arma::vec vv(p_l + K * p_l);

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

          vv = block_soft(vv, t_l * lam1 * std::sqrt((double)p_l));

          for (int jj = 0; jj < p_l; jj++) {
            beta_try(jj) = vv(jj);
          }

          ctr = p_l;

          for (int jj = 0; jj < p_l; jj++) {
            for (int k = 0; k < K; k++) {
              Theta_try(k, jj) = vv(ctr);
              ctr++;
            }
          }

          beta_l_new = beta_try;
          Theta_l_new = Theta_try;

          // candidate objective

          arma::vec beta_cand = beta;
          arma::mat Theta_cand = Theta;

          for (int jj = 0; jj < p_l; jj++) {
            int j = idx_l_vec[jj];
            beta_cand(j) = beta_l_new(jj);
            Theta_cand.col(j) = Theta_l_new.col(jj);
          }

          double obj_cand = objective_sgpl_cpp(X, Z, y, beta_cand, Theta_cand, lambda, alpha,
                                               groups_x, groups_z, family, status);

          arma::vec beta_curr = beta;
          arma::mat Theta_curr = Theta;

          for (int jj = 0; jj < p_l; jj++) {
            int j = idx_l_vec[jj];
            beta_curr(j) = beta_l_tilde(jj);
            Theta_curr.col(j) = Theta_l_tilde.col(jj);
          }

          double obj_curr = objective_sgpl_cpp(X, Z, y, beta_curr, Theta_curr, lambda, alpha,
                                               groups_x, groups_z, family, status);

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

          double d =std::abs(beta_l_tilde(jj) -beta_l_old(jj));
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

      // accept update

      for (int jj = 0; jj < p_l; jj++) {

        int j = idx_l_vec[jj];
        beta(j) = beta_l_tilde(jj);
        Theta.col(j) = Theta_l_tilde.col(jj);
      }
    }

    // Check outer convergence

    obj_path(iter_out) = objective_sgpl_cpp(X, Z, y, beta, Theta, lambda, alpha,
             groups_x, groups_z, family, status);
    double delta_out = 0.0;

    for (int j = 0; j < p; j++) {
      double d = std::abs(beta(j) - beta_old(j));
      if (d > delta_out) {
        delta_out = d;
      }
    }

    for (int j = 0; j < p; j++) {
      for (int k = 0; k < K; k++) {
        double d = std::abs(Theta(k, j) - Theta_old(k, j));

        if (d > delta_out) {
          delta_out = d;
        }
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

  for (int i = 0; i < n_iter; i++) {
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




