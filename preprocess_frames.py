import cv2
import numpy as np
import os
import sys

def preprocess_image(img_path, out_path):
    img = cv2.imread(img_path)
    if img is None: return

    # 1. Denoising (Bilateral Filter)
    # d=9: Pixel neighborhood, 75: Color sigma, 75: Space sigma
    # This smooths flat areas but keeps object boundaries sharp
    denoised = cv2.bilateralFilter(img, 9, 75, 75)

    # 2. Subtle Sharpening (Unsharp Masking)
    # We don't use a simple kernel because it adds too much noise.
    # We blur the image and subtract it from the original.
    gaussian_3 = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian_3, -0.5, 0)

    # 3. Histogram Equalization (Optional - helps with lighting)
    # Convert to YUV to only equalize the brightness (Y), not colors
    img_yuv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    final = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    cv2.imwrite(out_path, final)

if __name__ == "__main__":
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    for f in sorted(os.listdir(in_dir)):
        if f.endswith(".png"):
            print(f"Preprocessing {f}...")
            preprocess_image(os.path.join(in_dir, f), os.path.join(out_dir, f))
