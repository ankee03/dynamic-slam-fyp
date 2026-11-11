import flowiz as fz
import sys
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_flow.py <input.flo> <output.png>")
        sys.exit(1)

    file_flo = sys.argv[1]
    file_png = sys.argv[2]

    if not os.path.exists(file_flo):
        print(f"Error: {file_flo} not found.")
        sys.exit(1)

    try:
        # Use flowiz to convert the .flo file to a numpy image (RGB)
        # flowiz handles the Middlebury color logic automatically
        flow_img = fz.convert_from_file(file_flo)
        
        # Save the result
        import cv2
        # flowiz returns RGB, OpenCV expects BGR for imwrite
        cv2.imwrite(file_png, cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR))
        
        print(f"Successfully converted {os.path.basename(file_flo)} to {os.path.basename(file_png)}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
