import os
import sys

def read_tum_txt(filename):
    """Reads an official TUM text file and returns a dict of {timestamp_str: file_path}"""
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                # Keep timestamp as a string to preserve exact text formatting from the dataset
                data[parts[0]] = parts[1]
    return data

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 associate_5col.py <mask_folder_path>")
        print("Example: python3 associate_5col.py ../../results_fr3_walking_xyz3/masks")
        sys.exit(1)

    mask_folder = sys.argv[1]
    # Convert the mask folder to a clean, absolute path on your hard drive
    absolute_mask_path = os.path.abspath(mask_folder)

    # Check for official files in the current dataset directory
    if not (os.path.exists("rgb.txt") and os.path.exists("depth.txt")):
        print("❌ Error: Cannot find 'rgb.txt' or 'depth.txt' in this directory.")
        print("Make sure you run this script inside the specific dataset folder.")
        return

    # Read the pristine official dataset files
    print("Reading official rgb.txt and depth.txt...")
    rgb_data = read_tum_txt("rgb.txt")
    depth_data = read_tum_txt("depth.txt")

    # We use the matched pairs file if it exists, or match them directly
    # To keep it bulletproof and match the official association tool:
    matches = []
    
    # Simple associate logic based on closest float value for the official txt files
    rgb_ts_floats = sorted([float(ts) for ts in rgb_data.keys()])
    depth_ts_floats = sorted([float(ts) for ts in depth_data.keys()])

    for r_float in rgb_ts_floats:
        if not depth_ts_floats:
            break
        # Find closest depth timestamp within 0.02s
        closest_d_float = min(depth_ts_floats, key=lambda x: abs(x - r_float))
        if abs(closest_d_float - r_float) <= 0.02:
            # Convert back to the exact string representation keys
            r_str = f"{r_float:.6f}"
            d_str = f"{closest_d_float:.6f}"
            
            # Re-verify keys exist in original data (handling floating precision edge cases)
            matching_r_key = min(rgb_data.keys(), key=lambda x: abs(float(x) - r_float))
            matching_d_key = min(depth_data.keys(), key=lambda x: abs(float(x) - closest_d_float))
            
            matches.append((matching_r_key, rgb_data[matching_r_key], matching_d_key, depth_data[matching_d_key]))

    # Save to the 5-column format
    output_file = 'associations_opticalScene.txt'
    with open(output_file, 'w') as f:
        for r_ts, r_path, d_ts, d_path in matches:
            # Construct the mask filename using the exact RGB timestamp string + .png
            mask_file_name = f"{r_ts}.png"
            full_mask_path = os.path.join(absolute_mask_path, mask_file_name)
            
            # Write 5 columns: ts_rgb rgb_path ts_depth depth_path absolute_mask_path
            f.write(f"{r_ts} {r_path} {d_ts} {d_path} {full_mask_path}\n")

    print(f"✅ Created 5-column 'associations.txt' with {len(matches)} pairs.")
    print(f"   Mask paths point directly to: {absolute_mask_path}/<timestamp_rgb>.png")

if __name__ == "__main__":
    main()

