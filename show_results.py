import matplotlib.pyplot as plt
import flowiz as fz
import os
import sys

# 1. Path setup
# Usage: python show_results.py ./images/one.png (to use as a title)
image_label = sys.argv[1] if len(sys.argv) > 1 else 'out.flo'
file_flo = 'out.flo'

# 2. Check if the .flo file exists
if not os.path.exists(file_flo):
    print(f"Error: {file_flo} not found. Run the model first!")
    sys.exit(1)

# 3. Convert .flo to a viewable color image
# flowiz converts the motion vectors into the standard color wheel
flow_img = fz.convert_from_file(file_flo)

# 4. Display only the flow result
plt.figure(figsize=(10, 8))
plt.imshow(flow_img)
plt.title(f'Optical Flow: {os.path.basename(image_label)}')
plt.axis('off')  # Hide axes for a clean "image only" look

plt.tight_layout()
plt.show()
