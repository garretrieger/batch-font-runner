#!/usr/bin/env python3

import argparse
import subprocess
import concurrent.futures
import re
import sys
import os
import traceback
import itertools


QUALITY_LEVELS = [
  1, 2, 3, 4, 5, 6, 7, 8
]

COST_PATTERN = re.compile(r"^ift_total_cost = ([0-9]+)$", re.MULTILINE)
TIME_PATTERN = re.compile(r"^CodepointToGlyphSegments took: ([0-9.]+) seconds$", re.MULTILINE)

def check_font(font_path, quality_level):
  """
  Runs the hb-depend-closure-parity check on a single font.
  Returns a tuple of (font_path, list_of_test_ids_with_under_approx)
  """

  cmd = [
    "./gen-segmentation-plan.sh",
    font_path,
    str(quality_level),
  ]

  total_cost = 0
  total_time = 0

  try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
      print(f"Error processing {font_path}: command line return code {result.returncode}", file=sys.stderr)
      return font_path, total_cost, total_time

    stderr_output = result.stderr

    match = COST_PATTERN.search(stderr_output)
    if match:
      total_cost = int(match.group(1))
    else:
      print(f"Error processing {font_path}: total cost not found in output", file=sys.stderr)

    match = TIME_PATTERN.search(stderr_output)
    if match:
      total_time = float(match.group(1))
    else:
      print(f"Error processing {font_path}: total time not found in output", file=sys.stderr)

  except FileNotFoundError:
    print(f"Error: segmenter executable not found at './gen-segmentation-plan.sh'.", file=sys.stderr)
    sys.exit(1)
  except Exception as e:
    print(f"Error processing {font_path}: {e} {traceback.format_exc()}", file=sys.stderr)

  return font_path, quality_level, total_cost, total_time

def main():
  parser = argparse.ArgumentParser(description="Coordinate running the gen_ift_segmentation_plan utility across a collection of fonts.")
  parser.add_argument("input_file", help="Text file with one font file path per line.")
  parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4, help="Number of concurrent executions (default: number of CPUs).")

  args = parser.parse_args()

  if not os.path.isfile(args.input_file):
    print(f"Error: Input file '{args.input_file}' not found.", file=sys.stderr)
    sys.exit(1)

  with open(args.input_file, 'r') as f:
    font_paths = [line.strip() for line in f if line.strip()]

  if not font_paths:
    print("No font paths found in the input file.", file=sys.stderr)
    sys.exit(0)

  result = subprocess.run(["./init-gen-segmentation-plan.sh"], capture_output=True, text=True)
  if result.returncode != 0:
    print("Failed to init gen_ift_segmentation_plan.")
    sys.exit(1)

  print("font_path, quality_level, total_cost, total_time")
  with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
    future_to_path = {
      executor.submit(check_font, path, quality): path
      for path, quality in itertools.product(font_paths, QUALITY_LEVELS)
    }

    processed_count = 0
    total_fonts = len(font_paths)
    for future in concurrent.futures.as_completed(future_to_path):
      path = future_to_path[future]
      processed_count += 1
      if processed_count % 100 == 0:
        print(f"Processed {processed_count}/{total_fonts} fonts...", file=sys.stderr)

      try:
        result = future.result()
        if result is not None:
          font_path, quality_level, total_cost, total_time = result
          # TODO XXXX also include quality level
          print(f"{font_path}, {quality_level}, {total_cost}, {total_time}")

      except Exception as exc:
        print(f"{path} generated an exception: {exc}", file=sys.stderr)

if __name__ == "__main__":
  main()
