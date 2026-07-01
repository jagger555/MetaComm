import argparse
import csv
import os
from collections import defaultdict


def read_scalar_rows(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = int(float(row["step"]))
                tag = row["tag"]
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append((step, tag, value))
    return rows


def filter_rows(rows, prefixes):
    if not prefixes:
        return rows
    return [row for row in rows if any(row[1].startswith(prefix) for prefix in prefixes)]


def write_long_csv(rows, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "tag", "value"])
        writer.writerows(rows)


def write_wide_csv(rows, output_path):
    grouped = defaultdict(dict)
    tags = sorted({tag for _, tag, _ in rows})
    for step, tag, value in rows:
        grouped[step][tag] = value

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + tags)
        for step in sorted(grouped.keys()):
            writer.writerow([step] + [grouped[step].get(tag, "") for tag in tags])


def main():
    parser = argparse.ArgumentParser(description="Filter/pivot logged scalar CSV files.")
    parser.add_argument("--run_dir", type=str, required=True, help="Experiment output directory.")
    parser.add_argument(
        "--prefix",
        nargs="*",
        default=["graph_"],
        help="Only keep scalar tags starting with these prefixes. Default: graph_.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="scalars.csv",
        help="Input scalar CSV file name under run_dir.",
    )
    parser.add_argument(
        "--output_long",
        type=str,
        default="graph_metrics_long.csv",
        help="Filtered long-format CSV output file name under run_dir.",
    )
    parser.add_argument(
        "--output_wide",
        type=str,
        default="graph_metrics_wide.csv",
        help="Filtered wide-format CSV output file name under run_dir.",
    )
    args = parser.parse_args()

    input_path = os.path.join(args.run_dir, args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cannot find scalar CSV: {input_path}")

    rows = read_scalar_rows(input_path)
    rows = filter_rows(rows, args.prefix)
    if not rows:
        raise ValueError("No scalar rows matched the requested prefixes.")

    write_long_csv(rows, os.path.join(args.run_dir, args.output_long))
    write_wide_csv(rows, os.path.join(args.run_dir, args.output_wide))


if __name__ == "__main__":
    main()
