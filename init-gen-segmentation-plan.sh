#!/bin/bash
bazel build -c opt  @ift_encoder//util:gen_ift_segmentation_plan
