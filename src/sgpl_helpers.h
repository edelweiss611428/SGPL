#ifndef SGPL_HELPERS_H
#define SGPL_HELPERS_H

#include <RcppArmadillo.h>
#include <cmath>

// Soft threshold for vectors and matrices
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

// Linear prediction

inline arma::vec predict_sgpl_cpp(
    const arma::mat& X, // n x p
    const arma::mat& Z, // n x K
    const arma::vec& beta, // p x 1
    const arma::mat& Theta, // K x p
    double beta0 = 0.0,
    const arma::vec& theta0 = arma::vec() // not nullablle
) {

  int n = X.n_rows;
  //base intercept

  arma::vec eta = arma::vec(n, arma::fill::value(beta0));

  if (!theta0.is_empty() && theta0.n_elem > 0) {
    eta += Z * theta0;
  }

  // might not safe
  arma::mat M = Z * Theta;              // n x p
  M.each_row() += beta.t();
  eta += arma::sum(X % M, 1);

  return eta; //prediction
}


// Loss functions

// gaussian loss
inline double compute_gaussian_loss(const arma::vec& y, const arma::vec& pred) {
  arma::vec resid = y - pred;
  return arma::dot(resid, resid) / (2.0 * y.n_elem);
}

// logistic loss (eta is prediction in linear)
inline double compute_logistic_loss(const arma::vec& y, const arma::vec& eta) {
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

// Compute sgpl penalties
// per_pred + joint_grp + z_grp + l1_b + l1_t
inline double compute_sgpl_penalties(
    const arma::vec& beta, const arma::mat& Theta,
    double lambda, double alpha,
    const arma::uvec& groups_x, const arma::uvec& groups_z, int K
) {

  int L = groups_x.max();
  int G = groups_z.max();

  arma::vec pl(L);
  arma::vec pg(G);

  pl.zeros();
  pg.zeros();

  for (int i = 0; i < groups_x.n_elem; i++) {
    pl(groups_x(i) - 1) += 1.0;
  }

  for (int i = 0; i < groups_z.n_elem; i++) {
    pg(groups_z(i) - 1) += 1.0;
  }

  double lam1 = lambda * (1.0 - alpha);
  double lam2 = lambda * alpha;

  arma::mat Theta2 = arma::square(Theta);

  // per-variable prediction penalty
  double per_pred = lam1 * arma::accu(arma::sqrt(arma::square(beta) + arma::sum(Theta2, 0).t()));
  double joint_grp = 0.0;

  for (int l = 1; l <= L; l++) {

    // future versions consider precomputations
    arma::uvec idx = arma::find(groups_x == l);
    if (idx.n_elem > 0) {
      arma::vec svec = arma::square(beta.elem(idx)) + arma::sum(Theta2.cols(idx), 0).t();
      joint_grp += std::sqrt(pl(l - 1)) * std::sqrt(arma::accu(svec));
    }

  }

  joint_grp *= lam1;
  double z_grp = 0.0;
  double w_denom = std::sqrt(1.0 + K);

  for (int l = 1; l <= L; l++) {

    arma::uvec idx_l = arma::find(groups_x == l);
    if (idx_l.n_elem == 0) continue;

    for (int g = 1; g <= G; g++) {

      arma::uvec idx_g = arma::find(groups_z == g);
      if (idx_g.n_elem == 0) continue;
      arma::mat sub = Theta2.submat(idx_g, idx_l);
      double s = arma::accu(sub);
      z_grp += std::sqrt(pg(g - 1)) / w_denom * std::sqrt(s);

    }

  }

  z_grp *= lam1;
  double l1_b = lam2 * arma::accu(arma::abs(beta));
  double l1_t = lam2 * arma::accu(arma::abs(Theta));

  return per_pred + joint_grp + z_grp + l1_b + l1_t;
}


//' Compute initial step size (1 / Lipschitz constant) based on family arg
 inline double compute_t_init(
     const arma::mat& X,
     const arma::mat& Z,
     const Rcpp::Nullable<double>& t_init_r,
     const std::string& family
 ) {
   if (t_init_r.isNotNull()) {
     return Rcpp::as<double>(t_init_r);
   }

   // Column-wise squared norms of X
   arma::rowvec x_norm2 = arma::sum(arma::square(X), 0);
   double max_xnorm2 = x_norm2.max();

   // Spectral norm squared of modifier matrix Z
   double z_op2 = arma::norm(Z, 2);
   z_op2 *= z_op2;

   // Lipschitz constant base for the linear components
   double Lf = (max_xnorm2 * (1.0 + z_op2)) / X.n_rows;

   // Logistic loss Hessian is bounded by 1/4 of the Gaussian loss Hessian
   if (family == "logistic" || family == "binomial") {
     Lf *= 0.25;
   }

   return 1.0 / Lf;
 }


// gradient functions
// separate functions for beta and Theta for efficiency

inline arma::vec gradient_smooth_loss_beta(
    const arma::mat& X_l,
    const arma::vec& y,
    const arma::vec& eta_l,
    int n,
    const std::string& family
) {
  arma::vec r_l(n);

  if (family == "gaussian") {
    r_l = y - eta_l;
  } else if (family == "binomial") {
    arma::vec prob_l = 1.0 / (1.0 + arma::exp(-eta_l));
    r_l = y - prob_l;
  }

  return - (X_l.t() * r_l) / n;
}

// w.r.t. theta
inline arma::mat gradient_smooth_loss_theta(
    const arma::mat& X_l,
    const arma::mat& Z,
    const arma::vec& y,
    const arma::vec& eta_l, // full linear predictor for the block
    int p_l,
    int K,
    int n,
    const std::string& family
){
  arma::vec r_l(n);

  if (family == "gaussian") {
    r_l = y - eta_l;
  } else if (family == "binomial"){
    arma::vec prob_l = 1.0 / (1.0 + arma::exp(-eta_l));
    r_l = y - prob_l;
  }

  arma::mat gT(K, p_l);

   // Vectorised interaction modifier gradients: gT_j = - (Z^T * (X_l_j % r_l)) / n
   for (int jj = 0; jj < p_l; jj++) {
     gT.col(jj) = - (Z.t() * (X_l.col(jj) % r_l)) / n;
   }

   return gT;
 }


// intercept estimators (void functions - wont return anything but will directly modify beta0, theta0)

inline void update_intercepts_gaussian(
    const arma::mat& X, const arma::mat& Z, const arma::vec& y,
    const arma::vec& beta, const arma::mat& Theta,
    double& beta0, arma::vec& theta0
) {

  int n = X.n_rows;
  arma::mat M = Z * Theta;
  M.each_row() += beta.t();
  arma::vec fitted_pen = arma::sum(X % M, 1);
  arma::vec r0 = y - fitted_pen; //manual computation of fitted values without intercept contribution
  // could call predict_sgpl_cpp as well but a bit slower
  // int K = Z.n_cols;
  // arma::vec fitted_pen = predict_sgpl_cpp(X<Z,beta, Theta, 0.0, arma::vec theta0(K, arma::fill::zeros));

  arma::mat DesignMat = arma::join_rows(arma::ones<arma::vec>(n), Z);
  arma::vec coefs;
  bool success = arma::solve(coefs, DesignMat, r0); // solve OLS problem to coefs
  // might be problematic if DesignMat too large -> still solvable but might see warning (not errors and crashed the env)

  if (success) {
    beta0 = coefs(0); // OLS -> beta0 will be intecept of the aboved OLS problem
    theta0 = coefs.subvec(1, Z.n_cols);
    theta0.replace(arma::datum::nan, 0.0);
  }

  // this is not safe -> i might inject noise into this (ridge style)
}

inline void update_intercepts_logit(
    const arma::mat& Z, const arma::vec& y, const arma::vec& eta_pen,
    double& beta0, arma::vec& theta0, int max_irls = 25, double tol = 1e-8
) {
  int n = Z.n_rows;
  int K = Z.n_cols;
  arma::mat W_design = arma::join_rows(arma::ones<arma::vec>(n), Z);
  arma::vec coefs(1 + K, arma::fill::zeros); // prevent division by 0
  coefs(0) = beta0;
  if ((int)theta0.n_elem == K) coefs.subvec(1, K) = theta0;

  for (int iter = 0; iter < max_irls; iter++) {
    arma::vec eta = W_design * coefs + eta_pen;
    arma::vec p = 1.0 / (1.0 + arma::exp(-eta));
    p = arma::clamp(p, 1e-15, 1.0 - 1e-15);

    arma::vec w = p % (1.0 - p);
    arma::vec z_adj = (eta - eta_pen) + (y - p) / w;

    arma::mat WX = W_design;
    WX.each_col() %= arma::sqrt(w);
    arma::vec Wz = z_adj % arma::sqrt(w);

    arma::vec coefs_new;
    if (!arma::solve(coefs_new, WX, Wz)) break;

    if (arma::max(arma::abs(coefs_new - coefs)) < tol) {
      coefs = coefs_new;
      break;
    }
    coefs = coefs_new;
  }

  beta0 = coefs(0);
  theta0 = coefs.subvec(1, K);
  theta0.replace(arma::datum::nan, 0.0);
}

#endif // SGPL_HELPERS_H
