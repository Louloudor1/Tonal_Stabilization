import os
import glob
import cv2
import numpy as np


# -------------------------
# I/O: sequence d'images
# -------------------------
def list_images(folder, exts=("png", "jpg", "jpeg", "bmp", "tif", "tiff")):
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(folder, f"*.{ext}"))
        paths += glob.glob(os.path.join(folder, f"*.{ext.upper()}"))
    paths.sort()
    return paths


def read_image_float01(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img.astype(np.float32) / 255.0)


def write_mp4(frames_rgb01, out_path, fps):
    if not frames_rgb01:
        raise ValueError("Aucune frame à écrire.")
    h, w = frames_rgb01[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter mp4v ne s'ouvre pas. Essaie AVI/MJPG si besoin.")

    for f in frames_rgb01:
        f8 = np.clip(f * 255.0, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(f8, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    print(f"[OK] Vidéo écrite: {os.path.abspath(out_path)}")


# -------------------------
# Utils: resize estimation
# -------------------------
def resize_for_estimation(img_rgb01, target_width=160):
    h, w = img_rgb01.shape[:2]
    if w <= target_width:
        return img_rgb01
    scale = target_width / float(w)
    nh = int(round(h * scale))
    out = cv2.resize(img_rgb01, (target_width, nh), interpolation=cv2.INTER_AREA)
    return out


# -------------------------
# Mouvement: Harris + LK + RANSAC affine partielle
# -------------------------
def harris_points(gray, max_pts=400, quality=0.01, min_dist=7):
    """
    Utilise goodFeaturesToTrack (Shi-Tomasi) par défaut (très proche en pratique),
    robuste et rapide. Si tu veux strictement Harris, on met useHarrisDetector=True.
    """
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_pts,
        qualityLevel=quality,
        minDistance=min_dist,
        useHarrisDetector=True,
        k=0.04
    )
    if pts is None:
        return None
    return pts.reshape(-1, 2).astype(np.float32)


def estimate_affine_partial(prev_rgb01, cur_rgb01):
    """
    Retourne A (2x3) tel que: x_cur ~= A * x_prev (dans repère image)
    """
    prev_g = cv2.cvtColor((prev_rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    cur_g  = cv2.cvtColor((cur_rgb01  * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    pts_prev = harris_points(prev_g, max_pts=500, quality=0.01, min_dist=7)
    if pts_prev is None or len(pts_prev) < 20:
        return None, 0.0

    pts_cur, st, err = cv2.calcOpticalFlowPyrLK(
        prev_g, cur_g,
        pts_prev.reshape(-1, 1, 2),
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    if pts_cur is None:
        return None, 0.0

    st = st.reshape(-1).astype(bool)
    p0 = pts_prev[st]
    p1 = pts_cur.reshape(-1, 2)[st]

    if len(p0) < 15:
        return None, 0.0

    A, inliers = cv2.estimateAffinePartial2D(
        p0, p1,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10
    )
    if A is None or inliers is None:
        return None, 0.0

    inlier_ratio = float(inliers.sum()) / float(len(inliers))
    return A, inlier_ratio


def affine2x3_to_3x3(A):
    M = np.eye(3, dtype=np.float32)
    M[:2, :3] = A.astype(np.float32)
    return M


def warp_affine_rgb(img_rgb01, M_3x3, out_shape_hw):
    """
    Warpe img avec la transfo M (3x3) (affine),
    vers un canvas de taille out_shape_hw (h,w).
    """
    h, w = out_shape_hw
    A = M_3x3[:2, :3]
    out = cv2.warpAffine(
        img_rgb01,
        A,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return out


# -------------------------
# Masque Omega: correspondances robustes
# -------------------------
def omega_mask(ref_rgb01, warped_rgb01, sigma=0.04, eps=1e-6):
    """
    Masque Omega basé sur différence centrée.
    Implémentation simple: on centre chaque pixel par la moyenne globale (par canal),
    puis on garde si ||(ref-mean(ref)) - (warped-mean(warped))||^2 < sigma^2
    """
    ref = ref_rgb01
    w   = warped_rgb01

    mr = ref.reshape(-1, 3).mean(axis=0)
    mw = w.reshape(-1, 3).mean(axis=0)

    dr = ref - mr
    dw = w - mw
    diff2 = np.sum((dr - dw) ** 2, axis=2)

    mask = diff2 < (sigma ** 2)

    # enlève les zones "vides" dues au warp (noires)
    valid = np.sum(w, axis=2) > eps
    mask = mask & valid
    return mask


# -------------------------
# Estimation alpha/gamma par canal (régression en log)
# -------------------------
def fit_alpha_gamma(ref_rgb01, warped_rgb01, mask, eps=1e-6, gamma_clip=(0.2, 5.0)):
    """
    Pour chaque canal c:
      log(ref) = a + gamma*log(warped)  => alpha = exp(a)
    """
    params = []
    m = mask

    for c in range(3):
        y = ref_rgb01[..., c][m]
        x = warped_rgb01[..., c][m]

        if x.size < 200:  # trop peu d'échantillons
            params.append((1.0, 1.0))
            continue

        x = np.clip(x, eps, 1.0)
        y = np.clip(y, eps, 1.0)

        lx = np.log(x)
        ly = np.log(y)

        vx = np.var(lx)
        if vx < 1e-8:
            # fallback: gamma=1, alpha=ratio des moyennes (en linéaire)
            alpha = float(np.mean(y) / (np.mean(x) + eps))
            params.append((alpha, 1.0))
            continue

        cov = float(np.mean((lx - lx.mean()) * (ly - ly.mean())))
        gamma = cov / (vx + 1e-12)
        gamma = float(np.clip(gamma, gamma_clip[0], gamma_clip[1]))

        a = float(ly.mean() - gamma * lx.mean())
        alpha = float(np.exp(a))

        params.append((alpha, gamma))

    return params  # [(alphaR,gammaR), (alphaG,gammaG), (alphaB,gammaB)]


def apply_tonal_transform(frame_rgb01, params, lam=1.0):
    """
    Applique T'(s)=lam*(alpha*s^gamma) + (1-lam)*s par canal.
    """
    out = np.empty_like(frame_rgb01)
    s = frame_rgb01
    for c in range(3):
        alpha, gamma = params[c]
        corrected = alpha * np.power(np.clip(s[..., c], 0.0, 1.0), gamma)
        out[..., c] = lam * corrected + (1.0 - lam) * s[..., c]
    return np.clip(out, 0.0, 1.0)


# -------------------------
# Boucle principale: stabilisation tonale
# -------------------------
def stabilize_tone_from_images(
    folder,
    fps=30,
    out_path="stabilized.mp4",
    est_width=160,
    omega_thresh=0.25,     # ω
    sigma=0.04,
    lambda0=0.9,
    p_motion=30.0,         # + grand => moins sensible au mouvement
    max_frames=None,
):
    paths = list_images(folder)
    if not paths:
        raise FileNotFoundError(f"Aucune image trouvée dans {folder}")

    if max_frames is not None:
        paths = paths[:max_frames]

    # Lire première frame
    f0 = read_image_float01(paths[0])
    if f0 is None:
        raise RuntimeError(f"Impossible de lire {paths[0]}")

    # Référence k (corrigée) + version estimation
    ref_full = f0.copy()
    ref_est = resize_for_estimation(ref_full, est_width)

    # pour composer les affines (référence -> courant)
    # Ici on garde une matrice C telle que x_cur ~= C * x_ref (dans repère estimation)
    C_ref_to_prev = np.eye(3, dtype=np.float32)

    outputs = [ref_full.copy()]
    prev_est = ref_est.copy()
    prev_full = ref_full.copy()

    for i in range(1, len(paths)):
        cur_full = read_image_float01(paths[i])
        if cur_full is None:
            print(f"[WARN] Skip unreadable: {paths[i]}")
            continue

        cur_est = resize_for_estimation(cur_full, est_width)

        # 1) Mouvement entre prev_est et cur_est
        A_2x3, inlier_ratio = estimate_affine_partial(prev_est, cur_est)
        if A_2x3 is None:
            # si on n'a pas de mouvement, on tente sans warp
            A_3x3 = np.eye(3, dtype=np.float32)
        else:
            A_3x3 = affine2x3_to_3x3(A_2x3)

        # Compose: x_cur = A(prev->cur) * x_prev, et x_prev = C(ref->prev) * x_ref
        C_ref_to_cur = A_3x3 @ C_ref_to_prev

        # 2) Warp du courant vers ref: on veut x_ref = inv(C) * x_cur
        C_cur_to_ref = np.linalg.inv(C_ref_to_cur)

        h_est, w_est = ref_est.shape[:2]
        warped_est = warp_affine_rgb(cur_est, C_cur_to_ref, (h_est, w_est))

        # 3) Masque Omega
        mask = omega_mask(ref_est, warped_est, sigma=sigma)
        ratio = float(mask.mean())

        # 4) Si pas assez de correspondances, change la référence
        if ratio < omega_thresh:
            # nouvelle référence = frame précédente déjà corrigée
            ref_full = outputs[-1].copy()
            ref_est = resize_for_estimation(ref_full, est_width)

            # reset composition
            C_ref_to_cur = np.eye(3, dtype=np.float32)
            C_cur_to_ref = np.eye(3, dtype=np.float32)

            # recalcul warp/mask (optionnel, ici on continue direct)
            warped_est = cur_est.copy()
            mask = np.ones((ref_est.shape[0], ref_est.shape[1]), dtype=bool)
            ratio = 1.0

        # 5) Estimation alpha/gamma sur low-res (ref_est vs warped_est)
        params = fit_alpha_gamma(ref_est, warped_est, mask)

        # 6) Viscosité lambda en fonction du mouvement (approx: translation du C_ref_to_cur)
        tx = float(C_ref_to_cur[0, 2])
        ty = float(C_ref_to_cur[1, 2])
        motion_norm = np.sqrt(tx * tx + ty * ty)
        lam = float(lambda0 * np.exp(-motion_norm / max(p_motion, 1e-6)))
        lam = float(np.clip(lam, 0.0, 1.0))

        # 7) Appliquer T' sur full-res
        out_full = apply_tonal_transform(cur_full, params, lam=lam)

        outputs.append(out_full)

        # update prev
        prev_est = cur_est
        prev_full = cur_full
        C_ref_to_prev = C_ref_to_cur

        if i % 20 == 0 or i == len(paths) - 1:
            print(f"[{i}/{len(paths)-1}] Omega={ratio:.3f}  inliers={inlier_ratio:.2f}  lam={lam:.3f}")

    write_mp4(outputs, out_path, fps)
    return outputs


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    SESSION_FOLDER = "/Users/louisdorlencourt/Documents/Documents/3A/TIVO/Images_graycard"
    OUT_MP4 = "/Users/louisdorlencourt/Documents/Documents/3A/TIVO/graycard_stabilized.mp4"

    stabilize_tone_from_images(
        folder=SESSION_FOLDER,
        fps=50,
        out_path=OUT_MP4,
        est_width=160,
        omega_thresh=0.55,
        sigma=0.05,
        lambda0=0.9,
        p_motion=30.0,
        max_frames=None
    )
