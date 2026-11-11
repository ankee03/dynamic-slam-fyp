import os
import glob
import sys
import subprocess

# 1. Configuration & Robust Path Resolution
# os.path.expanduser converts '~' to '/home/fyp' automatically
raw_input = sys.argv[1] if len(sys.argv) > 1 else './images'
input_dir = os.path.abspath(os.path.expanduser(raw_input))
output_dir = os.path.abspath('./output_flows')

# DEBUG SECTION: This will tell you exactly why it might find 0 images
print(f"--- Environment Debug ---")
print(f"Searching in: {input_dir}")
print(f"Folder exists: {os.path.exists(input_dir)}")

# 2. Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# 3. Get images (Search for both .png and .PNG)
# Python glob is case-sensitive on Linux, so we check both
search_pattern = os.path.join(input_dir, "*.[pP][nN][gG]")
images = sorted(glob.glob(search_pattern))

if len(images) < 2:
    print(f"\nError: Found {len(images)} images in {input_dir}")
    print(f"Search pattern used: {search_pattern}")
    
    # Troubleshooting aid: list what IS actually in the folder
    if os.path.exists(input_dir):
        contents = os.listdir(input_dir)
        print(f"Files found in folder (first 5): {contents[:5]}")
    sys.exit(1)

print(f"\n✅ Success! Found {len(images)} images. Starting processing...")

# 4. Loop through the sequence
for i in range(len(images) - 1):
    img1 = images[i]
    img2 = images[i+1]

    # Create a unique output name
    base_name = os.path.basename(img1).replace('.png', '').replace('.PNG', '')
    out_flo = os.path.join(output_dir, f"{base_name}.flo")

    print(f"[{i+1}/{len(images)-1}] Processing: {os.path.basename(img1)} -> {os.path.basename(img2)}")

    # 5. Call the LiteFlowNet run.py command
    cmd = [
        "python", "run.py",
        "--model", "kitti",
        "--one", img1,
        "--two", img2,
        "--out", out_flo
    ]

    # subprocess.run executes the command and waits for it to finish
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running run.py for {base_name}: {e}")
        continue

print(f"\nDone! All flow files are saved in: {output_dir}")
