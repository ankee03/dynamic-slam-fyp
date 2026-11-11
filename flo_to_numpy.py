import numpy as np
import os
import sys

def load_flo(filename):
    """
    Standard Middlebury .flo file reader
    """
    if not os.path.isfile(filename):
        print(f"Error: {filename} not found.")
        return None

    with open(filename, 'rb') as f:
        # Check magic number 'PIEH' (80, 73, 69, 72)
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            print('Magic number incorrect. Invalid .flo file.')
            return None
        
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        
        data = np.fromfile(f, np.float32, count=2 * w * h)
        # Reshape to (height, width, 2) where 2 is (u, v) flow vectors
        return data.reshape((h, w, 2))

if __name__ == '__main__':
    # Default input/output paths
    input_flo = sys.argv[1] if len(sys.argv) > 1 else './out.flo'
    output_npy = sys.argv[2] if len(sys.argv) > 2 else './flow_result.npy'

    print(f"Reading {input_flo}...")
    flow_data = load_flo(input_flo)

    if flow_data is not None:
        np.save(output_npy, flow_data)
        print(f"Successfully saved flow to: {output_npy}")
        print(f"Array shape: {flow_data.shape}")
