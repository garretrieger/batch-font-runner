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
  0, 1, 2, 3, 4, 5, 6, 7, 8
]


TIME_PATTERN = re.compile(r"^CodepointToGlyphSegments took: ([0-9.]+) seconds$", re.MULTILINE)

CODEPOINT_COUNT_PATTERN = re.compile(r"^number_of_codepoints = ([0-9]+)$", re.MULTILINE)
IDEAL_COST_PATTERN = re.compile(r"^ideal_total_cost = ([0-9]+)$", re.MULTILINE)
IFT_COST_PATTERN = re.compile(r"^ift_total_cost = ([0-9]+)$", re.MULTILINE)
NON_IFT_COST_PATTERN = re.compile(r"^non_ift_total_cost = ([0-9]+)$", re.MULTILINE)

def get_cjk_script_flag(font_path):
  """
  Checks METADATA.pb for CJK subsets and returns the appropriate flag if applicable.
  """
  font_dir = os.path.dirname(font_path)
  metadata_path = os.path.join(font_dir, "METADATA.pb")

  if not os.path.isfile(metadata_path):
    return []

  subsets = []
  try:
    with open(metadata_path, 'r', encoding='utf-8') as f:
      for line in f:
        line = line.strip()
        if line.startswith("subsets:"):
          match = re.search(r'subsets:\s*"([^"]+)"', line)
          if match:
            subsets.append(match.group(1))
  except Exception as e:
    print(f"Error reading {metadata_path}: {e}", file=sys.stderr)
    return []

  target_subsets = {"chinese-simplified", "chinese-hongkong", "chinese-traditional", "japanese", "korean"}
  found_target_subsets = [s for s in subsets if s in target_subsets]

  if len(found_target_subsets) == 1:
    subset = found_target_subsets[0]
    script_map = {
      "chinese-simplified": "Script_chinese-simplified",
      "chinese-hongkong": "Script_chinese-traditional",
      "chinese-traditional": "Script_chinese-traditional",
      "japanese": "Script_japanese",
      "korean": "Script_korean"
    }
    script_value = script_map.get(subset)
    if script_value:
      return [f"--auto_config_primary_script={script_value}"]

  return []

def check_font(font_path, quality_level, timeout=None):
  """
  Runs the gen-segmentation-plan.sh on a single font.
  Returns a tuple of (font_path, quality_level, ideal_total_cost, ift_total_cost, non_ift_total_cost, total_time, codepoint_count)
  """

  extra_args = get_cjk_script_flag(font_path)

  cmd = [
    "./gen-segmentation-plan.sh",
    font_path,
    str(quality_level),
  ] + extra_args

  total_cost = 0
  total_time = 0

  try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
      print(f"Error processing {font_path}, quality {quality_level}: command line return code {result.returncode}", file=sys.stderr)
      return None

    stderr_output = result.stderr

    match = CODEPOINT_COUNT_PATTERN.search(stderr_output)
    if match:
      codepoint_count = int(match.group(1))
    else:
      print(f"Error processing {font_path}: total cost not found in output", file=sys.stderr)

    match = IDEAL_COST_PATTERN.search(stderr_output)
    if match:
      ideal_total_cost = int(match.group(1))
    else:
      print(f"Error processing {font_path}: total cost not found in output", file=sys.stderr)

    match = IFT_COST_PATTERN.search(stderr_output)
    if match:
      ift_total_cost = int(match.group(1))
    else:
      print(f"Error processing {font_path}: total cost not found in output", file=sys.stderr)

    match = NON_IFT_COST_PATTERN.search(stderr_output)
    if match:
      non_ift_total_cost = int(match.group(1))
    else:
      print(f"Error processing {font_path}: total cost not found in output", file=sys.stderr)

    match = TIME_PATTERN.search(stderr_output)
    if match:
      total_time = float(match.group(1))
    else:
      print(f"Error processing {font_path}: total time not found in output", file=sys.stderr)

  except subprocess.TimeoutExpired:
    print(f"Timeout occurred for font: {font_path} at quality level: {quality_level}", file=sys.stderr)
    return None
  except FileNotFoundError:
    print(f"Error: segmenter executable not found at './gen-segmentation-plan.sh'.", file=sys.stderr)
    sys.exit(1)
  except Exception as e:
    print(f"Error processing {font_path}: {e} {traceback.format_exc()}", file=sys.stderr)

  return font_path, quality_level, ideal_total_cost, ift_total_cost, non_ift_total_cost, total_time, codepoint_count

def main():
  parser = argparse.ArgumentParser(description="Coordinate running the gen_ift_segmentation_plan utility across a collection of fonts.")
  parser.add_argument("input_file", help="Text file with one font file path per line.")
  parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4, help="Number of concurrent executions (default: number of CPUs).")
  parser.add_argument("-t", "--timeout", type=float, help="Timeout in seconds for each execution.")
  parser.add_argument("-r", "--resume", help="Path to a previous stdout log file to resume from. Skips already processed font/quality pairs.")

  args = parser.parse_args()

  if not os.path.isfile(args.input_file):
    print(f"Error: Input file '{args.input_file}' not found.", file=sys.stderr)
    sys.exit(1)

  with open(args.input_file, 'r') as f:
    font_paths = [line.strip() for line in f if line.strip()]

  if not font_paths:
    print("No font paths found in the input file.", file=sys.stderr)
    sys.exit(0)

  skip_set = set()
  if args.resume:
    if os.path.isfile(args.resume):
      with open(args.resume, 'r') as f:
        for line in f:
          parts = [p.strip() for p in line.split(';')]
          if len(parts) >= 2 and parts[0] != "font_path":
            try:
              font_path = parts[0]
              quality_level = int(parts[1])
              skip_set.add((font_path, quality_level))
            except ValueError:
              pass
    else:
      print(f"Warning: Resume file '{args.resume}' not found. Starting from scratch.", file=sys.stderr)

  result = subprocess.run(["./init-gen-segmentation-plan.sh"], capture_output=True, text=True)
  if result.returncode != 0:
    print("Failed to init gen_ift_segmentation_plan.")
    sys.exit(1)

  tasks_to_run = [
    (path, quality)
    for path, quality in itertools.product(font_paths, QUALITY_LEVELS)
    if (path, quality) not in skip_set
  ]

  print("font_path; quality_level; ideal_total_cost; ift_total_cost; non_ift_total_cost; total_time_s; codepoint_count")
  with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
    future_to_path = {
      executor.submit(check_font, path, quality, args.timeout): path
      for path, quality in tasks_to_run
    }

    processed_count = 0
    total_fonts = len(tasks_to_run)
    for future in concurrent.futures.as_completed(future_to_path):
      path = future_to_path[future]
      processed_count += 1
      if processed_count % 100 == 0:
        print(f"Processed {processed_count}/{total_fonts} fonts...", file=sys.stderr)

      try:
        result = future.result()
        if result is not None:
          font_path, quality_level, ideal_total_cost, ift_total_cost, non_ift_total_cost, total_time, codepoint_count = result
          print(f"{font_path}; {quality_level}; {ideal_total_cost}; {ift_total_cost}; {non_ift_total_cost}; {total_time}; {codepoint_count}")

      except Exception as exc:
        print(f"{path} generated an exception: {exc}", file=sys.stderr)

if __name__ == "__main__":
  main()
