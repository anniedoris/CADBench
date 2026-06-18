import argparse
import json
from pathlib import Path


def load_rows(path: str):
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    else:
        with open(path, "r") as f:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Dump the 'generated' CadQuery code from each row of a jsonl or json file into individual .py files named by file_id.")
    parser.add_argument("--jsonl_path", required=True, help="Path to the jsonl or json file (e.g. tested_models/cadevolve/mesh/r1/benchB.jsonl or data/cadevolve/bench0_random_easy.json)")
    parser.add_argument("--output_dir", required=True, help="Directory to write the .py files to")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for row in load_rows(args.jsonl_path):
        file_id = row["file_id"]
        code = row["generated"]
        (output_dir / f"{file_id}.py").write_text(code)
        written += 1

    print(f"Wrote {written} .py file(s) to {output_dir}")


if __name__ == "__main__":
    main()
