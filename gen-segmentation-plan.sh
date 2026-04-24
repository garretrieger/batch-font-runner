#!/bin/bash

FONT=$1
QUALITY=$2
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR/bazel-batch-font-runner/external/ift_encoder_data+
$SCRIPT_DIR/bazel-bin/external/ift_encoder+/util/gen_ift_segmentation_plan \
  --input_font="$FONT" \
  --auto_config_quality="$QUALITY" \
  --nooutput_segmentation_plan \
  --output_segmentation_analysis
