#ifndef SGPL_HELPERS_H
#define SGPL_HELPERS_H

// Shared helpers for the SGPL solvers (fastSGPL.cpp and legacySGPL.cpp).
//
// The solvers implement Algorithm 1 of the paper (exact block proximal
// updates, "Option A"):
//   * blockwise coordinate descent over the predictor groups;
//   * proximal-gradient steps within each block, where the proximal operator
//     of the full block penalty is computed EXACTLY by the Dykstra-like
//     proximal algorithm (Bauschke and Combettes, 2008);
//   * backtracking with the quadratic upper-bound (sufficient-decrease) test;
//     a step that fails the test is never accepted;
//   * safe group screening (Corollary 1(ii) of the paper);
//   * gaussian: (beta0, theta0) profiled out; binomial / cox: unpenalized
//     intercept block updated once per outer iteration.
//
// Coefficient layout of a block l (p_l predictors, K modifiers):
//   v = (beta_l (p_l), vec(Theta_l) (K * p_l, column-major)),
// i.e. the K interaction coefficients of predictor jj are contiguous.

#include <RcppArmadillo.h>
#include <cmath>
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// Basic shrinkage operators
// ---------------------------------------------------------------------------

// Soft threshold for vectors and matrices
inline arma::vec soft_thresh_vec(const arma::vec& x, double t) {
  arma::vec out(x.n_elem);
  for (arma::uword i = 0; i < x.n_elem; i++) {
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
  for (arma::uword i = 0; i < x.n_rows; i++) {
    for (arma::uword j = 0; j < x.n_cols; j++) {
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

// block_soft: (1 - t / ||v||_2)_+ v
inline arma::vec block_soft(const arma::vec& v, double t) {
  double nv = arma::norm(v, 2);
  arma::vec out(v.n_elem);
  if (!std::isfinite(nv) || nv <= t) {
    out.zeros();
    return out;
  }
  out = v * (1.0 - t / nv);
  return out;
}

// In-place block soft-thresholding of the coordinates idx of v
inline void block_soft_inplace(arma::vec& v, const arma::uvec& idx, double t) {
  double nv = arma::norm(v.elem(idx), 2);
  if (!std::isfinite(nv) || nv <= t) {
    v.elem(idx).zeros();
  } else {
    v.elem(idx) *= (1.0 - t / nv);
  }
}

// ---------------------------------------------------------------------------
// Block structure and EXACT block proximal operator
// ---------------------------------------------------------------------------

struct SGPLBlock {
  int p_l;
  int K;
  std::vector<arma::uvec> pred;   // coordinates of (beta_j, theta_j)
  std::vector<arma::uvec> modg;   // coordinates of theta_lg
  arma::vec wg;                   // modifier-group weights w_lg = sqrt(p_l) sqrt(p_g) / sqrt(1 + K)
};

inline SGPLBlock make_sgpl_block(int p_l, int K,
                                 const std::vector<arma::uvec>& group_z_indices) {
  SGPLBlock B;
  B.p_l = p_l;
  B.K = K;
  for (int jj = 0; jj < p_l; ++jj) {
    arma::uvec idx(K + 1);
    idx(0) = jj;
    for (int k = 0; k < K; ++k) idx(k + 1) = p_l + jj * K + k;
    B.pred.push_back(idx);
  }
  B.wg.set_size(group_z_indices.size());
  for (std::size_t g = 0; g < group_z_indices.size(); ++g) {
    const arma::uvec& zg = group_z_indices[g];
    arma::uvec idx(p_l * zg.n_elem);
    arma::uword c = 0;
    for (int jj = 0; jj < p_l; ++jj) {
      for (arma::uword kk = 0; kk < zg.n_elem; ++kk) idx(c++) = p_l + jj * K + zg(kk);
    }
    B.modg.push_back(idx);
    B.wg(g) = std::sqrt((double)p_l) * std::sqrt((double)zg.n_elem) / std::sqrt(1.0 + K);
  }
  return B;
}

// Exact prox of the nested part
//   t2 * ||v||_1 + t1 * sum_j ||(beta_j, theta_j)||_2 + t1 * sqrt(p_l) * ||v||_2
// (leaf-to-root composition; exact because the groups are nested).
inline arma::vec prox_nested_part(arma::vec v, const SGPLBlock& B, double t1, double t2) {
  v = soft_thresh_vec(v, t2);
  for (const arma::uvec& idx : B.pred) block_soft_inplace(v, idx, t1);
  return block_soft(v, std::sqrt((double)B.p_l) * t1);
}

// Exact prox of the modifier-group part t1 * sum_g w_g ||theta_lg||_2
// (the modifier groups are disjoint).
inline arma::vec prox_modifier_part(arma::vec v, const SGPLBlock& B, double t1) {
  for (std::size_t g = 0; g < B.modg.size(); ++g) {
    block_soft_inplace(v, B.modg[g], B.wg(g) * t1);
  }
  return v;
}

// Exact prox of the full SGPL block penalty (t1 = t * lambda1, t2 = t * lambda2),
// computed with the Dykstra-like proximal algorithm. 'iters' returns the number
// of Dykstra iterations used.
inline arma::vec prox_sgpl_block(const arma::vec& v, const SGPLBlock& B,
                                 double t1, double t2,
                                 double tol, int maxit, int& iters) {
  arma::vec x = v;
  arma::vec c(v.n_elem, arma::fill::zeros);
  arma::vec e(v.n_elem, arma::fill::zeros);
  arma::vec y;
  iters = maxit;
  for (int it = 1; it <= maxit; ++it) {
    y = prox_modifier_part(x + c, B, t1);
    c = x + c - y;
    arma::vec xn = prox_nested_part(y + e, B, t1, t2);
    e = y + e - xn;
    double ch = std::max(arma::norm(xn - x, "inf"), arma::norm(xn - y, "inf"));
    x = xn;
    if (ch <= tol) {
      iters = it;
      break;
    }
  }
  return x;
}

// Penalty of one block (lam1 = lambda (1 - alpha), lam2 = lambda alpha)
inline double block_penalty_sgpl(const arma::vec& v, const SGPLBlock& B,
                                 double lam1, double lam2) {
  double s = 0.0;
  for (const arma::uvec& idx : B.pred) s += arma::norm(v.elem(idx), 2);
  s += std::sqrt((double)B.p_l) * arma::norm(v, 2);
  for (std::size_t g = 0; g < B.modg.size(); ++g) {
    s += B.wg(g) * arma::norm(v.elem(B.modg[g]), 2);
  }
  return lam1 * s + lam2 * arma::accu(arma::abs(v));
}

// Pack / unpack a block
inline arma::vec pack_block(const arma::vec& beta_l, const arma::mat& Theta_l) {
  return arma::join_cols(beta_l, arma::vectorise(Theta_l));
}
inline void unpack_block(const arma::vec& v, int p_l, int K,
                         arma::vec& beta_l, arma::mat& Theta_l) {
  beta_l = v.head(p_l);
  Theta_l = arma::reshape(v.tail((arma::uword)K * p_l), K, p_l);
}

// ---------------------------------------------------------------------------
// Linear predictor and block products
// ---------------------------------------------------------------------------

inline arma::vec predict_sgpl_cpp(
    const arma::mat& X,                     // n x p
    const arma::mat& Z,                     // n x K
    const arma::vec& beta,                  // p x 1
    const arma::mat& Theta,                 // K x p
    double beta0 = 0.0,
    const arma::vec& theta0 = arma::vec()   // not nullable
) {
  int n = X.n_rows;
  arma::vec eta = arma::vec(n, arma::fill::value(beta0));
  if (!theta0.is_empty() && theta0.n_elem > 0) {
    eta += Z * theta0;
  }
  arma::mat M = Z * Theta;                  // n x p
  M.each_row() += beta.t();
  eta += arma::sum(X % M, 1);
  return eta;
}

// Block contribution H_l v = X_l beta_l + rowSums(X_l % (Z Theta_l))
inline arma::vec block_fit(const arma::mat& X_l, const arma::mat& Z,
                           const arma::vec& v, int p_l, int K) {
  arma::vec beta_l;
  arma::mat Theta_l;
  unpack_block(v, p_l, K, beta_l, Theta_l);
  arma::vec u = X_l * beta_l;
  u += arma::sum(X_l % (Z * Theta_l), 1);
  return u;
}

// Block transpose product H_l^T w = (X_l^T w, vec(Z^T (X_l % w)))
inline arma::vec block_crossprod(const arma::mat& X_l, const arma::mat& Z,
                                 const arma::vec& w) {
  arma::vec gb = X_l.t() * w;
  arma::mat gT = Z.t() * (X_l.each_col() % w);
  return arma::join_cols(gb, arma::vectorise(gT));
}

// Orthogonal projection onto the complement of span(1, Z)
inline arma::vec project_out(const arma::mat& Q, const arma::vec& v) {
  return v - Q * (Q.t() * v);
}

// ---------------------------------------------------------------------------
// Cox partial likelihood (Breslow ties)
// ---------------------------------------------------------------------------

struct CoxData {
  arma::uvec ord;                  // indices sorted by increasing time
  std::vector<arma::uword> gstart; // tie groups in 'ord' (start positions)
  std::vector<arma::uword> gend;   // tie groups in 'ord' (end positions, inclusive)
  arma::vec status;                // event indicators (0/1)
  arma::vec time;
};

inline CoxData make_cox_data(const arma::vec& time, const arma::vec& status) {
  CoxData C;
  C.time = time;
  C.status = status;
  C.ord = arma::stable_sort_index(time, "ascend");
  arma::uword n = time.n_elem;
  arma::uword s = 0;
  while (s < n) {
    arma::uword e = s;
    while (e + 1 < n && time(C.ord(e + 1)) == time(C.ord(s))) ++e;
    C.gstart.push_back(s);
    C.gend.push_back(e);
    s = e + 1;
  }
  return C;
}

// Risk-set sums S_g = sum_{m: y_m >= t_g} exp(eta_m - shift), per tie group
inline arma::vec cox_risk_sums(const CoxData& C, const arma::vec& w) {
  std::size_t G = C.gstart.size();
  arma::vec S(G);
  double acc = 0.0;
  for (std::size_t g = G; g-- > 0;) {
    for (arma::uword k = C.gstart[g]; k <= C.gend[g]; ++k) acc += w(C.ord(k));
    S(g) = acc;
  }
  return S;
}

// (1/n) sum_{i in D} [ log sum_{m in R_i} exp(eta_m) - eta_i ]
inline double compute_cox_loss(const CoxData& C, const arma::vec& eta) {
  double shift = eta.max();
  arma::vec w = arma::exp(eta - shift);
  arma::vec S = cox_risk_sums(C, w);
  double loss = 0.0;
  for (std::size_t g = 0; g < C.gstart.size(); ++g) {
    for (arma::uword k = C.gstart[g]; k <= C.gend[g]; ++k) {
      arma::uword i = C.ord(k);
      if (C.status(i) > 0.5) loss += std::log(S(g)) + shift - eta(i);
    }
  }
  return loss / eta.n_elem;
}

// psi = d loss / d eta (n-vector)
inline arma::vec compute_cox_psi(const CoxData& C, const arma::vec& eta) {
  arma::uword n = eta.n_elem;
  double shift = eta.max();
  arma::vec w = arma::exp(eta - shift);
  arma::vec S = cox_risk_sums(C, w);
  arma::vec psi(n);
  double A = 0.0;                              // sum_{i in D, y_i <= t_g} 1 / S_{g(i)}
  for (std::size_t g = 0; g < C.gstart.size(); ++g) {
    double d_g = 0.0;
    for (arma::uword k = C.gstart[g]; k <= C.gend[g]; ++k) d_g += C.status(C.ord(k));
    A += d_g / S(g);
    for (arma::uword k = C.gstart[g]; k <= C.gend[g]; ++k) {
      arma::uword m = C.ord(k);
      psi(m) = (w(m) * A - C.status(m)) / n;
    }
  }
  return psi;
}

// ---------------------------------------------------------------------------
// Loss functions and the derivative psi = d loss / d eta
// ---------------------------------------------------------------------------

inline double compute_gaussian_loss(const arma::vec& y, const arma::vec& eta) {
  arma::vec r = y - eta;
  return 0.5 * arma::dot(r, r) / y.n_elem;
}

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

inline arma::vec compute_logistic_psi(const arma::vec& y, const arma::vec& eta) {
  arma::vec mu = 1.0 / (1.0 + arma::exp(-eta));
  return (mu - y) / (double)y.n_elem;
}

// Loss for binomial / cox from the full linear predictor eta
inline double glm_loss(const std::string& family, const arma::vec& y,
                       const CoxData* C, const arma::vec& eta) {
  if (family == "binomial") return compute_logistic_loss(y, eta);
  return compute_cox_loss(*C, eta);
}
inline arma::vec glm_psi(const std::string& family, const arma::vec& y,
                         const CoxData* C, const arma::vec& eta) {
  if (family == "binomial") return compute_logistic_psi(y, eta);
  return compute_cox_psi(*C, eta);
}

// ---------------------------------------------------------------------------
// SGPL penalty
// ---------------------------------------------------------------------------

// per_pred + joint_grp + z_grp + l1_b + l1_t
inline double compute_sgpl_penalties(
    const arma::vec& beta, const arma::mat& Theta,
    double lambda, double alpha,
    const arma::uvec& groups_x, const arma::uvec& groups_z, int K
) {
  int L = groups_x.max();
  int G = groups_z.max();
  arma::vec pl(L, arma::fill::zeros);
  arma::vec pg(G, arma::fill::zeros);
  for (arma::uword i = 0; i < groups_x.n_elem; i++) pl(groups_x(i) - 1) += 1.0;
  for (arma::uword i = 0; i < groups_z.n_elem; i++) pg(groups_z(i) - 1) += 1.0;

  double lam1 = lambda * (1.0 - alpha);
  double lam2 = lambda * alpha;
  arma::mat Theta2 = arma::square(Theta);

  double per_pred = lam1 * arma::accu(arma::sqrt(arma::square(beta) + arma::sum(Theta2, 0).t()));

  double joint_grp = 0.0;
  for (int l = 1; l <= L; l++) {
    arma::uvec idx = arma::find(groups_x == (arma::uword)l);
    if (idx.n_elem > 0) {
      arma::vec svec = arma::square(beta.elem(idx)) + arma::sum(Theta2.cols(idx), 0).t();
      joint_grp += std::sqrt(pl(l - 1)) * std::sqrt(arma::accu(svec));
    }
  }
  joint_grp *= lam1;

  double z_grp = 0.0;
  double w_denom = std::sqrt(1.0 + K);
  for (int l = 1; l <= L; l++) {
    arma::uvec idx_l = arma::find(groups_x == (arma::uword)l);
    if (idx_l.n_elem == 0) continue;
    for (int g = 1; g <= G; g++) {
      arma::uvec idx_g = arma::find(groups_z == (arma::uword)g);
      if (idx_g.n_elem == 0) continue;
      double s = arma::accu(Theta2.submat(idx_g, idx_l));
      // w_lg = sqrt(p_l) sqrt(p_g) / sqrt(1 + K)
      z_grp += std::sqrt(pl(l - 1)) * std::sqrt(pg(g - 1)) / w_denom * std::sqrt(s);
    }
  }
  z_grp *= lam1;

  double l1_b = lam2 * arma::accu(arma::abs(beta));
  double l1_t = lam2 * arma::accu(arma::abs(Theta));
  return per_pred + joint_grp + z_grp + l1_b + l1_t;
}

// ---------------------------------------------------------------------------
// Step sizes
// ---------------------------------------------------------------------------

// Largest eigenvalue of (1/n) A^T A for A = H_l (optionally projected),
// by power iteration using only products with X_l and Z.
inline double block_lipschitz(const arma::mat& X_l, const arma::mat& Z,
                              const arma::mat& Q, bool project,
                              int p_l, int K, int maxit = 100) {
  int n = X_l.n_rows;
  arma::vec v(p_l * (1 + K), arma::fill::ones);
  v /= arma::norm(v, 2);
  double val = 0.0, old = 0.0;
  for (int it = 0; it < maxit; ++it) {
    arma::vec u = block_fit(X_l, Z, v, p_l, K);
    if (project) u = project_out(Q, u);
    arma::vec w = block_crossprod(X_l, Z, u) / n;
    val = arma::dot(v, w);
    double nw = arma::norm(w, 2);
    if (nw <= 0.0) return 1e-12;
    v = w / nw;
    if (std::abs(val - old) <= 1e-8 * std::max(1.0, std::abs(val))) break;
    old = val;
  }
  return std::max(val, 1e-12);
}

//' Compute initial step size (1 / Lipschitz constant) based on family arg
//' (kept for backward compatibility; the solvers now use block-specific
//' Lipschitz constants computed by block_lipschitz()).
inline double compute_t_init(
    const arma::mat& X,
    const arma::mat& Z,
    const Rcpp::Nullable<double>& t_init_r,
    const std::string& family
) {
  if (t_init_r.isNotNull()) {
    return Rcpp::as<double>(t_init_r);
  }
  arma::rowvec x_norm2 = arma::sum(arma::square(X), 0);
  double max_xnorm2 = x_norm2.max();
  double z_op2 = arma::norm(Z, 2);
  z_op2 *= z_op2;
  double Lf = (max_xnorm2 * (1.0 + z_op2)) / X.n_rows;
  if (family == "logistic" || family == "binomial") {
    Lf *= 0.25;
  }
  return 1.0 / Lf;
}

// Curvature factor of the loss with respect to eta (upper bound)
inline double family_curvature(const std::string& family) {
  if (family == "binomial") return 0.25;
  return 1.0;   // gaussian; cox start value (backtracking adjusts it)
}

// ---------------------------------------------------------------------------
// Gradient functions (kept for backward compatibility)
// ---------------------------------------------------------------------------

inline arma::vec gradient_smooth_loss_beta(
    const arma::mat& X_l, const arma::vec& y, const arma::vec& eta_l,
    int n, const std::string& family
) {
  arma::vec r_l(n);
  if (family == "gaussian") {
    r_l = y - eta_l;
  } else {
    arma::vec prob_l = 1.0 / (1.0 + arma::exp(-eta_l));
    r_l = y - prob_l;
  }
  return - (X_l.t() * r_l) / n;
}

inline arma::mat gradient_smooth_loss_theta(
    const arma::mat& X_l, const arma::mat& Z, const arma::vec& y,
    const arma::vec& eta_l, int p_l, int K, int n, const std::string& family
) {
  arma::vec r_l(n);
  if (family == "gaussian") {
    r_l = y - eta_l;
  } else {
    arma::vec prob_l = 1.0 / (1.0 + arma::exp(-eta_l));
    r_l = y - prob_l;
  }
  arma::mat gT(K, p_l);
  for (int jj = 0; jj < p_l; jj++) {
    gT.col(jj) = - (Z.t() * (X_l.col(jj) % r_l)) / n;
  }
  return gT;
}

// ---------------------------------------------------------------------------
// Unpenalized intercept blocks
// ---------------------------------------------------------------------------

// Gaussian: least-squares regression of y - X~ gamma on (1, Z)
inline void update_intercepts_gaussian(
    const arma::mat& X, const arma::mat& Z, const arma::vec& y,
    const arma::vec& beta, const arma::mat& Theta,
    double& beta0, arma::vec& theta0
) {
  int n = X.n_rows;
  arma::mat M = Z * Theta;
  M.each_row() += beta.t();
  arma::vec fitted_pen = arma::sum(X % M, 1);
  arma::vec r0 = y - fitted_pen;
  arma::mat DesignMat = arma::join_rows(arma::ones<arma::vec>(n), Z);
  arma::vec coefs;
  bool success = arma::solve(coefs, DesignMat, r0);
  if (success) {
    beta0 = coefs(0);
    theta0 = coefs.subvec(1, Z.n_cols);
    theta0.replace(arma::datum::nan, 0.0);
  }
}

// Binomial: Newton (IRLS) steps for (beta0, theta0) with offset eta_pen.
// Step-halving guarantees that the logistic loss never increases.
inline void update_intercepts_logit(
    const arma::mat& Z, const arma::vec& y, const arma::vec& eta_pen,
    double& beta0, arma::vec& theta0, int max_irls = 25, double tol = 1e-8
) {
  int n = Z.n_rows;
  int K = Z.n_cols;
  arma::mat W_design = arma::join_rows(arma::ones<arma::vec>(n), Z);
  arma::vec coefs(1 + K, arma::fill::zeros);
  coefs(0) = beta0;
  if ((int)theta0.n_elem == K) coefs.subvec(1, K) = theta0;

  double loss = compute_logistic_loss(y, W_design * coefs + eta_pen);
  for (int iter = 0; iter < max_irls; iter++) {
    arma::vec eta = W_design * coefs + eta_pen;
    arma::vec p = 1.0 / (1.0 + arma::exp(-eta));
    arma::vec w = arma::clamp(p % (1.0 - p), 1e-10, 1.0);
    arma::vec grad = W_design.t() * (p - y) / n;
    arma::mat Hs = W_design.t() * (W_design.each_col() % w) / n;
    Hs.diag() += 1e-10;
    arma::vec step;
    if (!arma::solve(step, Hs, grad)) break;
    double s = 1.0;
    bool ok = false;
    arma::vec cand;
    double loss_c = loss;
    for (int h = 0; h < 30; ++h) {
      cand = coefs - s * step;
      loss_c = compute_logistic_loss(y, W_design * cand + eta_pen);
      if (loss_c <= loss + 1e-14) { ok = true; break; }
      s *= 0.5;
    }
    if (!ok) break;
    double ch = arma::max(arma::abs(cand - coefs));
    coefs = cand;
    loss = loss_c;
    if (ch < tol) break;
  }
  beta0 = coefs(0);
  theta0 = coefs.subvec(1, K);
  theta0.replace(arma::datum::nan, 0.0);
}

// Cox: Newton steps for theta0 with offset eta_pen (beta0 is not identifiable
// and is absorbed into the baseline hazard). Step-halving guarantees that the
// partial-likelihood loss never increases.
inline void update_intercepts_cox(
    const CoxData& C, const arma::mat& Z, const arma::vec& eta_pen,
    arma::vec& theta0, int max_newton = 25, double tol = 1e-8
) {
  int n = Z.n_rows;
  int K = Z.n_cols;
  if ((int)theta0.n_elem != K) theta0.zeros(K);
  double loss = compute_cox_loss(C, Z * theta0 + eta_pen);
  for (int iter = 0; iter < max_newton; ++iter) {
    arma::vec eta = Z * theta0 + eta_pen;
    double shift = eta.max();
    arma::vec w = arma::exp(eta - shift);
    // gradient
    arma::vec grad = Z.t() * compute_cox_psi(C, eta);
    // Hessian: (1/n) sum_{i in D} Cov_{R_i}(z)
    std::size_t G = C.gstart.size();
    double S0 = 0.0;
    arma::vec S1(K, arma::fill::zeros);
    arma::mat S2(K, K, arma::fill::zeros);
    arma::mat Hs(K, K, arma::fill::zeros);
    for (std::size_t g = G; g-- > 0;) {
      double d_g = 0.0;
      for (arma::uword k = C.gstart[g]; k <= C.gend[g]; ++k) {
        arma::uword m = C.ord(k);
        arma::rowvec zr = Z.row(m);
        S0 += w(m);
        S1 += w(m) * zr.t();
        S2 += w(m) * (zr.t() * zr);
        d_g += C.status(m);
      }
      if (d_g > 0.0) {
        arma::vec mean = S1 / S0;
        Hs += d_g * (S2 / S0 - mean * mean.t());
      }
    }
    Hs /= n;
    Hs.diag() += 1e-10;
    arma::vec step;
    if (!arma::solve(step, Hs, grad)) break;
    double s = 1.0;
    bool ok = false;
    arma::vec cand;
    double loss_c = loss;
    for (int h = 0; h < 30; ++h) {
      cand = theta0 - s * step;
      loss_c = compute_cox_loss(C, Z * cand + eta_pen);
      if (loss_c <= loss + 1e-14) { ok = true; break; }
      s *= 0.5;
    }
    if (!ok) break;
    double ch = arma::max(arma::abs(cand - theta0));
    theta0 = cand;
    loss = loss_c;
    if (ch < tol) break;
  }
  theta0.replace(arma::datum::nan, 0.0);
}

#endif // SGPL_HELPERS_H
