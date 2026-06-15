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


// Loss functions

// gaussian loss
inline double compute_gaussian_loss(const arma::vec& y, const arma::vec& pred) {
  arma::vec resid = y - pred;
  return arma::dot(resid, resid) / (2.0 * y.n_elem);
}


// logistic loss
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

// cox loss
inline double compute_cox_loss(
    const arma::vec& y_time,
    const arma::vec& status,
    const arma::vec& eta,
    int n
) {
  double loss = 0.0;
  arma::uvec events = arma::find(status == 1);

  if (events.n_elem == 0) return 0.0;

  for (unsigned int i = 0; i < events.n_elem; ++i) {
    int idx = events(i);
    arma::uvec risk = arma::find(y_time >= y_time(idx));
    arma::vec eta_risk = eta.elem(risk);

    double m = eta_risk.max();
    loss += (m + std::log(arma::accu(arma::exp(eta_risk - m)))) - eta(idx);
  }
  return loss / n;
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


//' Compute Initial Step Size (1 / Lipschitz constant) based on family arg
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

// w.r.t. beta
inline arma::vec gradient_smooth_loss_beta(
    const arma::mat& X_l,
    const arma::vec& y,           // y or y_time
    const arma::vec& eta_l,
    int n,
    const std::string& family,
    const Rcpp::Nullable<arma::vec>& status = R_NilValue // Optional status for Cox
) {
  if (family == "cox") {
    arma::vec status_vec = Rcpp::as<arma::vec>(status);
    arma::vec gb(X_l.n_cols, arma::fill::zeros);
    arma::uvec events = arma::find(status_vec == 1);

    for (unsigned int i = 0; i < events.n_elem; ++i) {
      int idx = events(i);
      arma::uvec risk = arma::find(y >= y(idx));
      arma::vec w = arma::exp(eta_l.elem(risk) - eta_l.elem(risk).max());
      double sw = arma::accu(w);
      gb -= (X_l.row(idx).t() - (X_l.rows(risk).t() * w) / sw);
    }
    return gb / n;
  }

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
    const arma::vec& y,           // y or y_time
    const arma::vec& eta_l,
    int p_l,
    int K,
    int n,
    const std::string& family,
    const Rcpp::Nullable<arma::vec>& status = R_NilValue // Optional status for Cox
) {
  if (family == "cox") {
    arma::vec status_vec = Rcpp::as<arma::vec>(status);
    arma::mat gT(K, p_l, arma::fill::zeros);
    arma::uvec events = arma::find(status_vec == 1);

    for (unsigned int i = 0; i < events.n_elem; ++i) {
      int idx = events(i);
      arma::uvec risk = arma::find(y >= y(idx));
      arma::vec w = arma::exp(eta_l.elem(risk) - eta_l.elem(risk).max());
      double sw = arma::accu(w);

      arma::mat zxbar(K, p_l, arma::fill::zeros);
      for(unsigned int r = 0; r < risk.n_elem; ++r) {
        zxbar += w(r) * (Z.row(risk(r)).t() * X_l.row(risk(r)));
      }
      gT -= (Z.row(idx).t() * X_l.row(idx) - (zxbar / sw));
    }
    return gT / n;
  }

  arma::vec r_l(n);
  if (family == "gaussian") {
    r_l = y - eta_l;
  } else if (family == "binomial"){
    arma::vec prob_l = 1.0 / (1.0 + arma::exp(-eta_l));
    r_l = y - prob_l;
  }

  arma::mat gT(K, p_l);
  for (int jj = 0; jj < p_l; jj++) {
    gT.col(jj) = - (Z.t() * (X_l.col(jj) % r_l)) / n;
  }
  return gT;
}

#endif // SGPL_HELPERS_H
