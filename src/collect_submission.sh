#!/bin/bash
# Script to zip the contents of the submission folder
# When unzipped, you get all contents from inside submission/ directly

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_DIR="$SCRIPT_DIR/submission"
OUTPUT_ZIP="$SCRIPT_DIR/submission.zip"

# Check if submission directory exists
if [ ! -d "$SUBMISSION_DIR" ]; then
    echo "Error: submission directory not found at $SUBMISSION_DIR"
    exit 1
fi

# Remove existing zip if it exists
if [ -f "$OUTPUT_ZIP" ]; then
    rm "$OUTPUT_ZIP"
    echo "Removed existing $OUTPUT_ZIP"
fi

# Change to submission directory and zip contents
# Note: We exclude most of meta_rl/experiments/ as it can be very large/noisy.
# We only include experiments/dream/tensorboard/episode which contains the
# TensorBoard logs needed by the autograder to verify DREAM training results.
cd "$SUBMISSION_DIR"

# First, zip everything except the experiments directory
zip -r "$OUTPUT_ZIP" . \
    -x "*.pyc" \
    -x "__pycache__/*" \
    -x "*.DS_Store" \
    -x ".*" \
    -x "meta_rl/experiments/*"

# Add back only the DREAM tensorboard/episode logs needed for autograding
if [ -d "meta_rl/experiments/dream/tensorboard/episode" ]; then
    zip -r "$OUTPUT_ZIP" "meta_rl/experiments/dream/tensorboard/episode"
    echo "Added DREAM tensorboard logs to submission"
else
    echo "Warning: meta_rl/experiments/dream/tensorboard/episode not found - skipping DREAM logs"
fi

echo "Created $OUTPUT_ZIP"
echo "Contents:"
unzip -l "$OUTPUT_ZIP" | head -20