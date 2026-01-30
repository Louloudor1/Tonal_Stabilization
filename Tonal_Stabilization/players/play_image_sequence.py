import os
import glob
import cv2


def list_images(folder, exts=("png", "jpg", "jpeg", "bmp", "tif", "tiff"), sort=True):
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*.{ext.upper()}")))
    if sort:
        # Tri lexicographique (marche si vos fichiers sont bien zero-padded: frame_0001.png)
        paths.sort()
    return paths


def play_image_sequence(folder, fps=25, window_name="Session Player", resize_to=None):
    """
    folder: dossier contenant les images
    fps: vitesse de lecture
    resize_to: (W, H) ou None
    """
    paths = list_images(folder)
    if not paths:
        raise FileNotFoundError(f"Aucune image trouvée dans: {folder}")

    delay_ms = max(1, int(1000 / fps))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    idx = 0
    paused = False

    while True:
        if not paused:
            img = cv2.imread(paths[idx], cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] Impossible de lire: {paths[idx]}")
                idx = (idx + 1) % len(paths)
                continue

            if resize_to is not None:
                img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)

            # Affiche un petit overlay
            overlay = img.copy()
            text = f"{idx+1}/{len(paths)} - {os.path.basename(paths[idx])}"
            cv2.putText(overlay, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, overlay)

            idx = (idx + 1) % len(paths)

        key = cv2.waitKey(delay_ms) & 0xFF

        # Contrôles
        if key in (27, ord('q')):   # ESC ou q
            break
        elif key == ord(' '):       # Espace = pause
            paused = not paused
        elif key == ord('a'):       # a = reculer
            idx = (idx - 2) % len(paths)  # -2 car on incrémente juste après
        elif key == ord('d'):       # d = avancer
            idx = idx % len(paths)
        elif key == ord('r'):       # r = restart
            idx = 0

    cv2.destroyAllWindows()


def export_to_mp4(folder, out_path="out.mp4", fps=25, resize_to=None):
    paths = list_images(folder)
    if not paths:
        raise FileNotFoundError(f"Aucune image trouvée dans: {folder}")

    first = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Impossible de lire la première image: {paths[0]}")

    if resize_to is not None:
        first = cv2.resize(first, resize_to, interpolation=cv2.INTER_AREA)

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Skip: {p}")
            continue
        if resize_to is not None:
            img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
        writer.write(img)

    writer.release()
    print(f"[OK] Export MP4: {out_path}")


if __name__ == "__main__":
    # === À MODIFIER ===
    SESSION_FOLDER = "/Users/louisdorlencourt/Documents/Documents/3A/TIVO/Images_graycard"
    FPS = 50

    # Lecture interactive
    play_image_sequence(SESSION_FOLDER, fps=FPS, resize_to=None)

    # Optionnel: exporter
    export_to_mp4(SESSION_FOLDER, out_path="/Users/louisdorlencourt/Documents/Documents/3A/TIVO/graycard.mp4", fps=FPS, resize_to=None)
