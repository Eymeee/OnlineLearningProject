"""
utils.py — Briques communes aux 5 TPs
======================================
Contenu :
  1. Covering Number (greedy epsilon-cover)
  2. Line Search : Armijo, Goldstein, Wolfe, adaptatif, self-adaptatif
  3. Validation croisée K-fold
  4. Régularisation L1 et L2 (sous-gradients)
  5. Métriques de classification
  6. Projection sur boule L2
  7. Normalisation des données
  8. Utilitaires d'affichage / plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ===========================================================================
# 1. COVERING NUMBER (greedy epsilon-cover)
# ===========================================================================

def greedy_epsilon_cover(A, epsilon, dist="euclidean"):
    """
    Algorithme glouton pour approximer le covering number N(A, epsilon).

    Paramètres
    ----------
    A       : np.ndarray, shape (n, m) — ensemble de points
    epsilon : float — rayon des boules
    dist    : str — "euclidean" (seul supporté pour l'instant)

    Retourne
    --------
    centers : np.ndarray — centres de la couverture
    n_cover : int — nombre de boules = N(A, epsilon)
    """
    A = np.array(A)
    uncovered = list(range(len(A)))
    centers_idx = []

    while uncovered:
        c_idx = uncovered[0]
        centers_idx.append(c_idx)
        c = A[c_idx]
        # Retirer tous les points dans la boule de rayon epsilon autour de c
        uncovered = [i for i in uncovered
                     if np.linalg.norm(A[i] - c) > epsilon]

    centers = A[centers_idx]
    return len(centers_idx)


def covering_number_curve(A, epsilons):
    """
    Calcule N(A, epsilon) pour une liste de valeurs de epsilon.

    Retourne
    --------
    counts : list[int] — N(A, epsilon) pour chaque epsilon
    """
    counts = []
    for eps in epsilons:
        _, n = greedy_epsilon_cover(A, eps)
        counts.append(n)
    return counts


def plot_covering_number(epsilons, counts, title="Covering number N(A, ε)"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epsilons, counts, marker='o', linewidth=2, color="#378ADD")
    ax.set_xlabel("ε (rayon)")
    ax.set_ylabel("N(A, ε)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


# ===========================================================================
# 2. LINE SEARCH
# ===========================================================================

def armijo(f, theta, d, g, alpha0=1.0, c1=1e-4, rho=0.5, max_iter=100):
    """
    Backtracking Armijo.

    f(theta) doit retourner un scalaire.
    d        : direction de descente
    g        : gradient/sous-gradient en theta
    """
    alpha = alpha0
    f0 = f(theta)
    slope = np.dot(g, d)
    for _ in range(max_iter):
        if f(theta + alpha * d) <= f0 + c1 * alpha * slope:
            break
        alpha *= rho
    return alpha


def goldstein(f, theta, d, g, alpha0=1.0, c=0.1, rho=0.5, max_iter=100):
    """
    Condition de Goldstein (encadrement).
    Cherche alpha tel que :
      f0 + (1-c)*alpha*slope <= f(theta + alpha*d) <= f0 + c*alpha*slope
    """
    alpha = alpha0
    f0 = f(theta)
    slope = np.dot(g, d)
    for _ in range(max_iter):
        f_new = f(theta + alpha * d)
        upper = f0 + c * alpha * slope
        lower = f0 + (1 - c) * alpha * slope
        if lower <= f_new <= upper:
            break
        if f_new > upper:
            alpha *= rho          # pas trop grand
        else:
            alpha /= rho          # pas trop petit
    return alpha


def wolfe(f, grad_f, theta, d, g, alpha0=1.0, c1=1e-4, c2=0.9,
          rho=0.5, max_iter=100):
    """
    Conditions de Wolfe (strong Wolfe avec sous-gradient au nouveau point).

    grad_f(theta) : fonction retournant un sous-gradient en theta
    """
    alpha = alpha0
    f0 = f(theta)
    slope = np.dot(g, d)
    for _ in range(max_iter):
        theta_new = theta + alpha * d
        f_new = f(theta_new)
        if f_new > f0 + c1 * alpha * slope:
            alpha *= rho
            continue
        g_new = grad_f(theta_new)
        curvature = np.dot(g_new, d)
        if abs(curvature) <= c2 * abs(slope):
            break
        if curvature < 0:
            alpha /= rho
        else:
            alpha *= rho
    return alpha


def adaptive_line_search(f, theta, d, alpha, success_threshold=0.01,
                          increase=1.2, decrease=0.5):
    """
    Pas adaptatif : augmente alpha si succès, le diminue sinon.

    Retourne (alpha_new, accepted)
    """
    f0 = f(theta)
    f_new = f(theta + alpha * d)
    relative_decrease = (f0 - f_new) / (abs(f0) + 1e-12)
    if relative_decrease >= success_threshold:
        return alpha * increase, True
    else:
        return alpha * decrease, False


class SelfAdaptiveLineSearch:
    """
    Pas self-adaptatif basé sur l'historique récent des succès/échecs.
    Ajuste automatiquement les facteurs d'augmentation et de diminution.
    """
    def __init__(self, alpha0=1.0, window=10, target_rate=0.5):
        self.alpha = alpha0
        self.window = window
        self.target_rate = target_rate   # taux de succès visé
        self.history = []                # 1 = succès, 0 = échec

    def step(self, f, theta, d, threshold=0.01):
        f0 = f(theta)
        f_new = f(theta + self.alpha * d)
        success = (f0 - f_new) / (abs(f0) + 1e-12) >= threshold
        self.history.append(int(success))

        if len(self.history) > self.window:
            self.history.pop(0)

        recent_rate = np.mean(self.history)
        # Ajuster les facteurs selon l'écart au taux cible
        if recent_rate > self.target_rate + 0.1:
            self.alpha *= 1.3    # trop de succès → augmenter davantage
        elif recent_rate < self.target_rate - 0.1:
            self.alpha *= 0.7    # trop d'échecs → réduire davantage

        return self.alpha, success


# ===========================================================================
# 3. VALIDATION CROISÉE K-FOLD
# ===========================================================================

def kfold_split(n, k=5, shuffle=True, seed=42):
    """
    Génère les indices de K-fold cross-validation.

    Retourne
    --------
    folds : list of (train_idx, val_idx)
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)

    fold_sizes = np.full(k, n // k)
    fold_sizes[:n % k] += 1
    folds = []
    current = 0
    for size in fold_sizes:
        val_idx = indices[current:current + size]
        train_idx = np.concatenate([indices[:current],
                                    indices[current + size:]])
        folds.append((train_idx, val_idx))
        current += size
    return folds


def kfold_cv(X, y, model_fn, score_fn, k=5, shuffle=True, seed=42, **kwargs):
    """
    Validation croisée K-fold générique.

    Paramètres
    ----------
    X, y       : données
    model_fn   : fonction(X_train, y_train, **kwargs) → modèle entraîné
    score_fn   : fonction(model, X_val, y_val) → float (score)
    k          : nombre de plis

    Retourne
    --------
    scores     : list[float] de longueur k
    mean_score : float
    std_score  : float
    """
    folds = kfold_split(len(y), k=k, shuffle=shuffle, seed=seed)
    scores = []
    for train_idx, val_idx in folds:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        model = model_fn(X_tr, y_tr, **kwargs)
        scores.append(score_fn(model, X_val, y_val))
    return scores, float(np.mean(scores)), float(np.std(scores))


def train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Découpe en train / validation / test.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return (X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx],
            X[test_idx],  y[test_idx])


# ===========================================================================
# 4. RÉGULARISATION L1 ET L2
# ===========================================================================

def l2_regularization(w, lambda_):
    """
    Retourne (terme de pénalité, sous-gradient de la régularisation L2).
    Pénalité : (lambda/2) * ||w||^2
    Gradient  : lambda * w
    """
    penalty = 0.5 * lambda_ * np.dot(w, w)
    grad    = lambda_ * w
    return penalty, grad


def l1_regularization(w, lambda_):
    """
    Retourne (terme de pénalité, sous-gradient de la régularisation L1).
    Pénalité : lambda * ||w||_1
    Sous-grad : lambda * sign(w)  (0 si w_j == 0)
    """
    penalty = lambda_ * np.sum(np.abs(w))
    subgrad = lambda_ * np.sign(w)
    return penalty, subgrad


def apply_l2_update(w, eta, lambda_):
    """Mise à jour avec régularisation L2 inline (OGD-style)."""
    return w * (1 - eta * lambda_)


def apply_l1_update(w, eta, lambda_):
    """
    Opérateur proximal L1 (soft-thresholding) pour OGD.
    Équivalent à la projection après un pas de sous-gradient L1.
    """
    threshold = eta * lambda_
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0.0)


# ===========================================================================
# 5. MÉTRIQUES DE CLASSIFICATION
# ===========================================================================

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[label_to_idx[yt], label_to_idx[yp]] += 1
    return cm, labels


def precision_recall_f1(y_true, y_pred, pos_label=1):
    """
    Calcule précision, rappel et F1 pour la classe positive.
    """
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    return float(precision), float(recall), float(f1)


def classification_report(y_true, y_pred, pos_label=1):
    acc = accuracy(y_true, y_pred)
    p, r, f1 = precision_recall_f1(y_true, y_pred, pos_label=pos_label)
    cm, labels = confusion_matrix(y_true, y_pred)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {p:.4f}")
    print(f"Recall    : {r:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"Confusion matrix (labels={list(labels)}):")
    print(cm)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "cm": cm}


def plot_confusion_matrix(cm, labels, title="Matrice de confusion"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


# ===========================================================================
# 6. PROJECTION SUR BOULE L2
# ===========================================================================

def project_l2_ball(w, radius=1.0):
    """
    Projette w sur la boule L2 de rayon `radius`.
    ΠB(w) = w * min(1, radius / ||w||)
    """
    norm = np.linalg.norm(w)
    if norm <= radius:
        return w.copy()
    return w * (radius / norm)


def project_l2_ball_inplace(w, radius=1.0):
    """Version in-place."""
    norm = np.linalg.norm(w)
    if norm > radius:
        w *= radius / norm


# ===========================================================================
# 7. NORMALISATION DES DONNÉES
# ===========================================================================

def standardize(X_train, X_val=None, X_test=None):
    """
    Standardise X_train (moyenne 0, écart-type 1).
    Applique la même transformation à X_val et X_test si fournis.
    """
    mu  = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_s = (X_train - mu) / std
    results = [X_train_s]
    for X in [X_val, X_test]:
        if X is not None:
            results.append((X - mu) / std)
    return results if len(results) > 1 else results[0], mu, std


def add_bias(X):
    """Ajoute une colonne de 1 (biais) à gauche de X."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


# ===========================================================================
# 8. UTILITAIRES DE PLOTTING
# ===========================================================================

def plot_losses(losses, label="Perte", color="#378ADD", title="Évolution de la perte"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, label=label, linewidth=2, color=color)
    ax.set_xlabel("Itération")
    ax.set_ylabel("Perte")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_multi_losses(losses_dict, title="Comparaison des pertes"):
    """
    losses_dict : dict {label: list_of_losses}
    """
    colors = ["#378ADD", "#D85A30", "#1D9E75", "#D4537E", "#EF9F27", "#7F77DD"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, losses) in enumerate(losses_dict.items()):
        ax.plot(losses, label=label, linewidth=2,
                color=colors[i % len(colors)])
    ax.set_xlabel("Itération")
    ax.set_ylabel("Perte")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_train_test(train_scores, test_scores, x_vals, xlabel="Degré d",
                    ylabel="Erreur", title="Biais-Variance"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_vals, train_scores, marker='o', label="Train",
            color="#378ADD", linewidth=2)
    ax.plot(x_vals, test_scores, marker='s', label="Test",
            color="#D85A30", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_decision_boundary_2d(w, b, X, y, title="Frontière de décision"):
    """
    Trace la frontière de décision pour un classifieur linéaire en 2D.
    w : np.ndarray de taille 2
    b : float
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {-1: "#378ADD", 1: "#D85A30"}
    for label in [-1, 1]:
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[label],
                   label=f"Classe {label}", alpha=0.7, edgecolors="none")

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    if abs(w[1]) > 1e-8:
        xx = np.linspace(x_min, x_max, 200)
        yy = -(w[0] * xx + b) / w[1]
        ax.plot(xx, yy, "k-", linewidth=2, label="Frontière")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig


def print_table(data, headers):
    """
    Affiche un tableau simple en console.
    data : list of list
    """
    col_widths = [max(len(str(row[i])) for row in [headers] + data)
                  for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in data:
        print(fmt.format(*[str(v) for v in row]))


# ===========================================================================
# 9. NORMES ET NORMES DUALES (TP5)
# ===========================================================================

def norm_l1(w):
    return np.sum(np.abs(w))

def norm_l2(w):
    return np.linalg.norm(w)

def norm_linf(w):
    return np.max(np.abs(w))

def dual_norm(w, primal="l2"):
    """
    Calcule la norme duale de w.
    dual(l1) = l_inf, dual(l2) = l2, dual(l_inf) = l1
    """
    mapping = {"l1": norm_linf, "l2": norm_l2, "linf": norm_l1}
    return mapping[primal](w)


# ===========================================================================
# 10. REGRET (TP4 / TP5)
# ===========================================================================

def compute_regret(cumulative_losses, best_fixed_loss):
    """
    Regret_T = sum_{t=1}^{T} l_t(w_t) - min_w sum_{t=1}^{T} l_t(w)

    cumulative_losses : liste ou array des pertes cumulées à chaque tour
    best_fixed_loss   : perte minimale d'un prédicteur fixe optimal
    """
    return np.array(cumulative_losses) - best_fixed_loss


def plot_regret(regrets_dict, title="Regret cumulé"):
    colors = ["#378ADD", "#D85A30", "#1D9E75", "#D4537E", "#EF9F27"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, regret) in enumerate(regrets_dict.items()):
        ax.plot(np.cumsum(regret), label=label, linewidth=2,
                color=colors[i % len(colors)])
    ax.set_xlabel("Rounds T")
    ax.set_ylabel("Regret cumulé")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig