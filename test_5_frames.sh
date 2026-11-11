#!/bin/bash

# Check if the user provided enough arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: ./process_data.sh [input_dir] [output_dir] [optional_threshold]"
    echo "Example: ./process_data.sh ./data/fr3_walking_xyz/rgb ./results_xyz 3.5"
    exit 1
fi

# Configuration from arguments
DATA_DIR=$1
OUT_DIR=$2
THRESHOLD=${3:-3.5}  # Uses 3.5 as default if 3rd argument is missing
# Create a temporary folder for cleaned images
PRE_DIR="$OUT_DIR/preprocessed_temp"
MODEL="kitti"

mkdir -p "$OUT_DIR"
mkdir -p "$PRE_DIR"

echo "Step 0: Preprocessing images for noise reduction..."
# Run the preprocessing script on the whole input directory
python preprocess_frames.py "$DATA_DIR" "$PRE_DIR"

# Get sorted list of PREPROCESSED files now
files=($(ls "$PRE_DIR"/*.png | sort))

# Get sorted list of PNG files
#files=($(ls "$DATA_DIR"/*.png | sort))

echo "Processing data from: $DATA_DIR"
echo "Saving results to: $OUT_DIR"
echo "Using threshold: $THRESHOLD"

# Loop through the files
for ((i=0; i<5; i++)); do
    FRAME1=${files[$i]}
    FRAME2=${files[$i+1]}
    BASENAME=$(basename "$FRAME1" .png)
    
    echo "------------------------------------------------"
    echo "Processing Pair $((i+1)): $BASENAME"

    # Step 1: Flow (Saves to OUT_DIR)
    python run.py --model $MODEL --one "$FRAME1" --two "$FRAME2" --out "$OUT_DIR/$BASENAME.flo"

    # NEW STEP: Convert .flo to a color .png for your report/thesis
    python visualize_flow.py "$OUT_DIR/$BASENAME.flo" "$OUT_DIR/flow_viz_$BASENAME.png"

    # Step 2: NumPy (Saves to OUT_DIR)
    python flo_to_numpy.py "$OUT_DIR/$BASENAME.flo" "$OUT_DIR/$BASENAME.npy"

    # Step 3: Mask (Now correctly saves to OUT_DIR with a unique name)
    python segment_motion.py "$OUT_DIR/$BASENAME.npy" "$OUT_DIR/mask_$BASENAME.png" $THRESHOLD
done

echo "Done! All results are in $OUT_DIR"
