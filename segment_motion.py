import numpy as np
import cv2
import os
import sys

def run_segmentation(input_path, output_path, threshold=3.0):
    # Check if the flow file exists
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run your LiteFlowNet script first.")
        return

    print(f"Processing {input_path} for Motion Segmentation...")

    # 1. Load the dense field (H x W x 2)
    flow = np.load(input_path)
    u = flow[..., 0]
    v = flow[..., 1]

    # 2. Estimate Background Motion (Median-based Ego-Motion)
    # As per your proposal: median is robust against dynamic object outliers
    u_bg = np.median(u)
    v_bg = np.median(v)
    
    print(f"Estimated Ego-Motion: u_bg={u_bg:.4f}, v_bg={v_bg:.4f}")

    # 3. Compute Motion Residuals (Eq 3.2)
    u_diff = u - u_bg
    v_diff = v - v_bg
    residual = np.sqrt(u_diff**2 + v_diff**2)

    # 4. Apply Threshold (Eq 3.3)
    # Typical values 2-5 pixels/frame
    binary_mask = (residual > threshold).astype(np.uint8) * 255

    # 5. Morphological Operations
    # Opening: removes small noise dots
    # Closing: fills small holes inside the object
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)

    # 6. Save Results
    cv2.imwrite(output_path, refined_mask)
    
    # Optional: Save a heatmap of the residuals for your thesis figures
    heatmap_path = output_path.replace('mask_', 'residual_')
    res_heatmap = cv2.applyColorMap((residual * 10).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_path, res_heatmap)

    print(f"SUCCESS: Generated '{output_path}' and '{heatmap_path}'")

if __name__ == '__main__':
    # You can adjust the threshold here (2.0 to 5.0)
    # These lines catch the arguments from the .sh script
    # Usage: python segment_motion.py [input_npy] [output_png] [threshold]
    
    in_file = sys.argv[1] if len(sys.argv) > 1 else 'flow_result.npy'
    out_file = sys.argv[2] if len(sys.argv) > 2 else 'motion_mask.png'
    
    try:
        thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5
    except:
        thresh = 3.5

    run_segmentation(input_path=in_file, output_path=out_file, threshold=thresh)
