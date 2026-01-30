#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implémentation *fidèle* à l'algorithme (Alg. 1) du papier FRIGO 2015
(stabilisation tonale via mouvement dominant + correction paramétrique).

Chemins demandés :
"""

import cv2
import numpy as np
import math

VIDEO_ORIG = "/Users/louisdorlencourt/Documents/Documents/3A/TIVO/graycard.mp4"
VIDEO_STAB = "/Users/louisdorlencourt/Documents/Documents/3A/TIVO/graycard_stabilized.mp4"


# =========================
# Paramètres du papier
# =========================
# largeur de travail pour l'estimation (papier : 120 px)
WORK_W = 120

# (6) seuil sur la similarité couleur centrée (dans [0,1])
SIGMA = 0.10

# (Alg.1 l.5) proportion minimale : |Ω_{t,k}| >= ω * |Ω|
OMEGA_FRAC = 0.70

# (7) lambda = lambda0 * exp(-||V_{t,k}|| / p)
LAMBDA0 = 0.90


# =========================
# Utils
# =========================
def resize_keep_aspect(bgr, w):
    h, W = bgr.shape[:2]
    if W == w:
        return bgr
    new_h = int(round(h * (w / float(W))))
    return cv2.resize(bgr, (w, new_h), interpolation=cv2.INTER_AREA)

def to_float01(bgr):
    return bgr.astype(np.float32) / 255.0

def estimate_affine_dominant_motion(prev_gray, curr_gray):
    """
    Estimation du mouvement dominant (affine) entre (t-1)->t.
    Retourne M 3x3 (homogène).
    """
    # points à suivre
    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=800,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7
    )
    if p0 is None:
        return np.eye(3, dtype=np.float32)

    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
    if p1 is None or st is None:
        return np.eye(3, dtype=np.float32)

    st = st.reshape(-1).astype(bool)
    p0 = p0[st].reshape(-1, 2)
    p1 = p1[st].reshape(-1, 2)

    if len(p0) < 12:
        return np.eye(3, dtype=np.float32)

    # affine partielle (rotation+translation+scale), robuste (RANSAC)
    A, _inl = cv2.estimateAffinePartial2D(
        p0, p1,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10
    )
    if A is None:
        return np.eye(3, dtype=np.float32)

    M = np.eye(3, dtype=np.float32)
    M[:2, :] = A.astype(np.float32)
    return M

def build_overlap_correspondences(u_k, u_t, A_tk):
    """
    Construit Ω et Ω_{t,k} exactement comme (5)-(6) du papier.

    u_k : image référence (float [0,1]) au temps k, shape (H,W,3)
    u_t : image courante (float [0,1]) au temps t, shape (H,W,3)
    A_tk : matrice 3x3 qui mappe coords de k -> t

    Retourne:
      - idx_x : indices linéaires des pixels x dans l'image k appartenant à Ω
      - y_int : coords (x,y) entières correspondantes dans l'image t
      - good_mask : booléen sur Ω, indiquant Ω_{t,k} (critère sigma)
    """
    Hk, Wk = u_k.shape[:2]
    Ht, Wt = u_t.shape[:2]

    # Grille de points x dans l'image k
    xs, ys = np.meshgrid(np.arange(Wk, dtype=np.float32),
                         np.arange(Hk, dtype=np.float32))
    ones = np.ones_like(xs)
    pts = np.stack([xs, ys, ones], axis=-1).reshape(-1, 3).T  # (3, N)

    # y = A_tk(x)
    q = (A_tk @ pts)  # (3, N)
    qx = q[0, :] / q[2, :]
    qy = q[1, :] / q[2, :]

    # Ω : points qui tombent dans l'image t
    inside = (qx >= 0) & (qx <= (Wt - 1)) & (qy >= 0) & (qy <= (Ht - 1))
    if not np.any(inside):
        idx_x = np.array([], dtype=np.int64)
        y_int = np.zeros((0, 2), dtype=np.int32)
        good_mask = np.array([], dtype=bool)
        return idx_x, y_int, good_mask

    idx = np.where(inside)[0]
    idx_x = idx.astype(np.int64)

    # coords entières (papier : correspondance par A_tk(x), on quantifie pour échantillonner)
    qx_i = np.rint(qx[inside]).astype(np.int32)
    qy_i = np.rint(qy[inside]).astype(np.int32)
    qx_i = np.clip(qx_i, 0, Wt - 1)
    qy_i = np.clip(qy_i, 0, Ht - 1)
    y_int = np.stack([qx_i, qy_i], axis=1)  # (M,2)

    # (6) critère sigma sur la différence des couleurs centrées
    # u_k^c(x) - mean(u_k^c)  vs  u_t^c(y) - mean(u_t^c)
    mu_k = u_k.reshape(-1, 3).mean(axis=0)
    mu_t = u_t.reshape(-1, 3).mean(axis=0)

    # récupérer u_k(x) et u_t(y)
    uk_flat = u_k.reshape(-1, 3)[idx_x]
    yt_lin = (y_int[:, 1] * Wt + y_int[:, 0]).astype(np.int64)
    ut_flat = u_t.reshape(-1, 3)[yt_lin]

    dk = uk_flat - mu_k
    dt = ut_flat - mu_t
    # (1/3) sum_c (...)^2 < sigma^2
    dist2 = np.mean((dk - dt) ** 2, axis=1)
    good_mask = dist2 < (SIGMA ** 2)

    return idx_x, y_int, good_mask

def estimate_alpha_gamma_logLS(uk_vals, ut_vals, eps=1e-6):
    """
    Résout (3)-(4) du papier (régression linéaire en domaine log):
      log(u_k) = gamma * log(u_t) + log(alpha)

    uk_vals, ut_vals : vecteurs float [0,1] de même taille (sur Ω_{t,k})
    """
    uk = np.clip(uk_vals, eps, 1.0)
    ut = np.clip(ut_vals, eps, 1.0)

    x = np.log(ut)
    y = np.log(uk)

    x_mean = float(x.mean())
    y_mean = float(y.mean())

    vx = float(((x - x_mean) ** 2).mean())
    if vx < 1e-12:
        gamma = 1.0
        alpha = float(np.exp(y_mean - gamma * x_mean))
        return alpha, gamma

    cov = float(((x - x_mean) * (y - y_mean)).mean())
    gamma = cov / vx
    alpha = float(np.exp(y_mean - gamma * x_mean))
    return alpha, gamma

def apply_parametric_correction_u8(frame_bgr_u8, alpha, gamma, lam):
    """
    Applique T'_t(u_t^c) = lam * alpha * (u_t^c)^gamma + (1-lam) * u_t^c
    via LUT (papier : LUT pour rapidité).
    alpha, gamma : arrays shape(3,) pour BGR
    """
    out = frame_bgr_u8.copy()
    # LUT sur [0..255] -> [0..255]
    x = np.arange(256, dtype=np.float32) / 255.0
    for c in range(3):
        y = lam * (alpha[c] * (x ** gamma[c])) + (1.0 - lam) * x
        y = np.clip(y, 0.0, 1.0)
        lut = (y * 255.0 + 0.5).astype(np.uint8)
        out[:, :, c] = cv2.LUT(out[:, :, c], lut)
    return out

def dominant_motion_vector(A_tk, W, H):
    """
    V_{t,k} : on prend le déplacement du centre (dominant motion) induit par A_tk.
    """
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5
    p = np.array([cx, cy, 1.0], dtype=np.float32)
    q = A_tk @ p
    qx = q[0] / q[2]
    qy = q[1] / q[2]
    vx = float(qx - cx)
    vy = float(qy - cy)
    return np.array([vx, vy], dtype=np.float32)


# =========================
# Algorithme 1 (papier)
# =========================
def stabilize_video():
    cap = cv2.VideoCapture(VIDEO_ORIG)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir {VIDEO_ORIG}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_STAB, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Impossible d'écrire {VIDEO_STAB}")

    # read first frame
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Vidéo vide ?")

    # Alg.1 l.1-2
    t = 1
    k = 1

    out_prev = frame0.copy()
    writer.write(out_prev)

    # référence u_k (papier : u_k, initial = u_1)
    u_k_full = out_prev.copy()

    # versions low-res
    u_k_lr = resize_keep_aspect(u_k_full, WORK_W)
    prev_lr = resize_keep_aspect(frame0, WORK_W)

    # A_{t,k} (k->t) en homogène
    A_tk = np.eye(3, dtype=np.float32)

    # param p de (7)
    p_norm = float(max(u_k_lr.shape[1], u_k_lr.shape[0]))

    # boucle principale (Alg.1)
    while True:
        ok, frame_t = cap.read()
        if not ok:
            break
        t += 1

        # low-res courant
        curr_lr = resize_keep_aspect(frame_t, WORK_W)

        # mouvement dominant entre (t-1)->t (sur low-res)
        prev_gray = cv2.cvtColor(prev_lr, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_lr, cv2.COLOR_BGR2GRAY)
        A_tt1 = estimate_affine_dominant_motion(prev_gray, curr_gray)

        # composition : A_{t,k} = A_{t,t-1} o A_{t-1,k}
        A_tk = (A_tt1 @ A_tk).astype(np.float32)

        # (4) Calculer Ω_{t,k} via (5)-(6)
        u_k_lr_f = to_float01(u_k_lr)
        u_t_lr_f = to_float01(curr_lr)

        idx_x, y_int, good = build_overlap_correspondences(u_k_lr_f, u_t_lr_f, A_tk)

        Omega_size = int(len(idx_x))                 # |Ω|
        Omega_tk_size = int(np.count_nonzero(good))  # |Ω_{t,k}|

        # (5) test |Ω_{t,k}| >= ω * |Ω|
        if Omega_size > 0 and Omega_tk_size >= int(math.ceil(OMEGA_FRAC * Omega_size)):
            # (6-9) estimer alpha/gamma par canal + appliquer correction
            # construire les paires (x,y) dans Ω_{t,k}
            idx_good = np.where(good)[0]
            idx_x_good = idx_x[idx_good]

            Ht_lr, Wt_lr = u_t_lr_f.shape[:2]
            yt = y_int[idx_good]
            yt_lin = (yt[:, 1] * Wt_lr + yt[:, 0]).astype(np.int64)

            uk_flat = u_k_lr_f.reshape(-1, 3)[idx_x_good]
            ut_flat = u_t_lr_f.reshape(-1, 3)[yt_lin]

            # (7) lambda
            V = dominant_motion_vector(A_tk, Wt_lr, Ht_lr)
            vnorm = float(np.linalg.norm(V))
            lam = float(LAMBDA0 * math.exp(-vnorm / max(1e-6, p_norm)))
            lam = max(0.0, min(1.0, lam))

            alpha = np.zeros(3, dtype=np.float32)
            gamma = np.zeros(3, dtype=np.float32)

            # IMPORTANT : OpenCV est en BGR, le papier note {r,g,b}
            # on applique canal par canal, l'ordre n'importe pas ici tant qu'on est cohérent
            for c in range(3):
                a, g = estimate_alpha_gamma_logLS(uk_flat[:, c], ut_flat[:, c])
                alpha[c] = a
                gamma[c] = g

            out_t = apply_parametric_correction_u8(frame_t, alpha, gamma, lam)
            writer.write(out_t)

            # mise à jour (t devient t-1 pour le prochain tour)
            out_prev = out_t
            prev_lr = curr_lr

        else:
            # (11-14) changement de référence
            # k <- t-1 ; u_k <- T'_{t-1}(u_{t-1})
            k = t - 1
            u_k_full = out_prev.copy()
            u_k_lr = resize_keep_aspect(u_k_full, WORK_W)

            # reset A_{t-1,k} = I puisque k=t-1 => pour recalculer Ω_{t,k},
            # on veut A_{t,k} = A_{t,t-1}
            A_tk = np.eye(3, dtype=np.float32)

            # on NE consomme PAS une nouvelle frame ici dans le papier :
            # on doit re-tester le *même* t avec le nouveau k.
            # Donc on annule les effets "avance" et on ré-évalue avec k=t-1:
            # -> on force A_tk = A_tt1 et on refait le test une fois.

            A_tk = A_tt1.copy()

            u_k_lr_f = to_float01(u_k_lr)
            u_t_lr_f = to_float01(curr_lr)

            idx_x, y_int, good = build_overlap_correspondences(u_k_lr_f, u_t_lr_f, A_tk)
            Omega_size = int(len(idx_x))
            Omega_tk_size = int(np.count_nonzero(good))

            if Omega_size > 0 and Omega_tk_size >= int(math.ceil(OMEGA_FRAC * Omega_size)):
                idx_good = np.where(good)[0]
                idx_x_good = idx_x[idx_good]

                Ht_lr, Wt_lr = u_t_lr_f.shape[:2]
                yt = y_int[idx_good]
                yt_lin = (yt[:, 1] * Wt_lr + yt[:, 0]).astype(np.int64)

                uk_flat = u_k_lr_f.reshape(-1, 3)[idx_x_good]
                ut_flat = u_t_lr_f.reshape(-1, 3)[yt_lin]

                V = dominant_motion_vector(A_tk, Wt_lr, Ht_lr)
                vnorm = float(np.linalg.norm(V))
                lam = float(LAMBDA0 * math.exp(-vnorm / max(1e-6, p_norm)))
                lam = max(0.0, min(1.0, lam))

                alpha = np.zeros(3, dtype=np.float32)
                gamma = np.zeros(3, dtype=np.float32)
                for c in range(3):
                    a, g = estimate_alpha_gamma_logLS(uk_flat[:, c], ut_flat[:, c])
                    alpha[c] = a
                    gamma[c] = g

                out_t = apply_parametric_correction_u8(frame_t, alpha, gamma, lam)
            else:
                # si même avec k=t-1 ça ne passe pas, on n'applique rien (cas dégénéré)
                out_t = frame_t.copy()

            writer.write(out_t)

            out_prev = out_t
            prev_lr = curr_lr

            # pour la suite, la référence est bien celle du papier :
            u_k_full = out_prev.copy()
            u_k_lr = resize_keep_aspect(u_k_full, WORK_W)
            A_tk = np.eye(3, dtype=np.float32)

    cap.release()
    writer.release()
    print("OK ->", VIDEO_STAB)


if __name__ == "__main__":
    stabilize_video()
