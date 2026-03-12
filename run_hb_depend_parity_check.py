#!/usr/bin/env python3

import argparse
import subprocess
import concurrent.futures
import re
import sys
import os

def check_font(font_path, parity_checker_path, iterations):
  """
  Runs the hb-depend-closure-parity check on a single font.
  Returns a tuple of (font_path, list_of_test_ids_with_under_approx)
  """
  cmd = [
    parity_checker_path,
    "--report-under-approximation",
    "-n", f"{iterations}",
    font_path
  ]

  under_approx_ids = []

  try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
      print(f"Error processing {font_path}: command line return code {result.returncode}", file=sys.stderr)

    # Sample error message:
    # Test 0xd4cdf94ae35986d6: UNDER-APPROX - Depend missed 1 glyphs: 73
    stderr_output = result.stderr
    pattern = re.compile(r"Test\s+(0x[0-9a-fA-F]+):\s+UNDER-APPROX")
    under_approx_ids = pattern.findall(stderr_output)

  except FileNotFoundError:
    print(f"Error: Parity checker executable not found at '{parity_checker_path}'.", file=sys.stderr)
    sys.exit(1)
  except Exception as e:
    print(f"Error processing {font_path}: {e}", file=sys.stderr)

  return font_path, under_approx_ids

def main():
  parser = argparse.ArgumentParser(description="Coordinate running the hb-depend-closure-parity utility across a collection of fonts.")
  parser.add_argument("input_file", help="Text file with one font file path per line.")
  parser.add_argument("--checker-path", default="./hb-depend-closure-parity", help="Path to the hb-depend-closure-parity executable.")
  parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4, help="Number of concurrent executions (default: number of CPUs).")
  parser.add_argument("-n", "--iterations", type=int, default=1000, help="Number of iterations to run on each parity checker execution.")

  args = parser.parse_args()

  if not os.path.isfile(args.input_file):
    print(f"Error: Input file '{args.input_file}' not found.", file=sys.stderr)
    sys.exit(1)

  with open(args.input_file, 'r') as f:
    font_paths = [line.strip() for line in f if line.strip()]

  if not font_paths:
    print("No font paths found in the input file.", file=sys.stderr)
    sys.exit(0)

  with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
    future_to_path = {
      executor.submit(check_font, path, args.checker_path, args.iterations): path
      for path in font_paths
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
          font_path, under_approx_ids = result

          for test_id in under_approx_ids:
            print(f"UNDER-APPROX on {font_path} for test ID {test_id}", file=sys.stdout)

      except Exception as exc:
        print(f"{path} generated an exception: {exc}", file=sys.stderr)

if __name__ == "__main__":
  main()
