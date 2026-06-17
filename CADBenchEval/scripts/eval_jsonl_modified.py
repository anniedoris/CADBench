import argparse
import json
import os
import sys
import pandas as pd
from tabulate import tabulate
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
from datasets import load_dataset
from CADBench import pretty_print_metrics
from CADBench.Eval import Evaluator


def _id_label_at(args):
    dataset, idx = args
    row = dataset[idx]
    return row["file_id"], row.get("label", "unknown")

def calculate_advanced_per_label_metrics(per_item_logs, base_output_path):
    """
    Aggregates results by label focusing on Medians for the Dashboard,
    calculates VSR, and exports to JSON, TXT, and Excel.
    """
    if not per_item_logs:
        return

    df = pd.DataFrame(per_item_logs)
    if 'label' not in df.columns:
        df['label'] = 'unknown'

    df['status'] = pd.to_numeric(df['status'], errors='coerce').fillna(0)

    all_metrics = [
        "Aligned IoU", "Aligned Chamfer Distance", "Aligned Surface IoU",
        "Naive IoU", "Naive Chamfer Distance", "Naive Surface IoU",
        "token_count", "line_count", "total_operations"
    ]
    
    existing_metrics = [m for m in all_metrics if m in df.columns]
    for m in existing_metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce').fillna(0.0)

    # Prepare Datasets
    df_success = df[df['status'] == 1.0].copy()
    df_adjusted = df.copy()
    df_adjusted.loc[df_adjusted['status'] != 1.0, existing_metrics] = 0.0

    # Aggregations
    success_stats = df_success.groupby('label')[existing_metrics].agg(['mean', 'median', 'std', 'count'])
    adjusted_stats = df_adjusted.groupby('label')[existing_metrics].agg(['mean', 'median', 'std', 'count'])

    output_dict = {}
    standard_table_rows = []
    adj_primary_rows_for_df = [] 
    adj_primary_blocks_txt = []
    
    label_order = {"easy": 0, "medium": 1, "hard": 2}
    sorted_labels = sorted(df['label'].unique(), key=lambda x: label_order.get(x.lower(), 99))

    def get_val(stat_df, lbl, metric, measure):
        try:
            val = stat_df.loc[lbl, (metric, measure)]
            return float(val) if pd.notnull(val) else 0.0
        except (KeyError, IndexError):
            return 0.0

    for label in sorted_labels:
        output_dict[label] = {}
        first_row_for_label = True
        
        geo_vals = []
        code_vals = []
        
        # DASHBOARD HEADERS: Only Medians and VSR
        ap_headers = [
            "Adj Med\nAligned IoU", 
            "Adj Med\nAligned CD", 
            "Adj Med\nAligned SIoU", 
            "VSR\n(%)",  
            "Adj Med\nTokens", "Adj Med\nLines", "Adj Med\nOps"
        ]
        
        excel_primary_row = {"Label": label.upper()}
        s_cnt_val, a_cnt_val = 0, 0

        for m in existing_metrics:
            s_mean = get_val(success_stats, label, m, 'mean')
            s_med  = get_val(success_stats, label, m, 'median')
            s_std  = get_val(success_stats, label, m, 'std')
            s_cnt  = int(get_val(success_stats, label, m, 'count'))
            
            a_mean = get_val(adjusted_stats, label, m, 'mean')
            a_med  = get_val(adjusted_stats, label, m, 'median')
            a_std  = get_val(adjusted_stats, label, m, 'std')
            a_cnt  = int(get_val(adjusted_stats, label, m, 'count'))

            if m == "Aligned IoU":
                s_cnt_val, a_cnt_val = s_cnt, a_cnt

            output_dict[label][m] = {
                "success_only": {"mean": s_mean, "median": s_med, "std": s_std, "count": s_cnt},
                "adjusted": {"mean": a_mean, "median": a_med, "std": a_std, "count": a_cnt}
            }

            # Keep full data for the Detailed Section
            standard_table_rows.append({
                "Label": label.upper() if first_row_for_label else "",
                "Metric": m,
                "Mean": s_mean, "Med": s_med, "SD": s_std, "Count": s_cnt,
                "Adj Mean": a_mean, "Adj Med": a_med, "Adj SD": a_std, "Total Count": a_cnt
            })
            first_row_for_label = False

            # Collect MEDIANS for Dashboard
            if m == "Aligned IoU":
                geo_vals.append(f"{a_med:.4f}")
                excel_primary_row["Adj Med Aligned IoU"] = a_med
            elif m == "Aligned Chamfer Distance":
                geo_vals.append(f"{a_med:.4f}")
                excel_primary_row["Adj Med Aligned CD"] = a_med
            elif m == "Aligned Surface IoU":
                geo_vals.append(f"{a_med:.4f}")
                excel_primary_row["Adj Med Aligned SIoU"] = a_med
            elif m == "token_count":
                code_vals.append(f"{a_med:.2f}")
                excel_primary_row["Adj Med Tokens"] = a_med
            elif m == "line_count":
                code_vals.append(f"{a_med:.2f}")
                excel_primary_row["Adj Med Lines"] = a_med
            elif m == "total_operations":
                code_vals.append(f"{a_med:.2f}")
                excel_primary_row["Adj Med Ops"] = a_med

        vsr = (s_cnt_val / a_cnt_val * 100) if a_cnt_val > 0 else 0.0
        vsr_str = f"{vsr:.2f}%"
        ap_values = geo_vals + [vsr_str] + code_vals
        
        # Excel Formatting
        excel_primary_row_ordered = {
            "Label": excel_primary_row["Label"],
            "Adj Med Aligned IoU": excel_primary_row.get("Adj Med Aligned IoU"),
            "Adj Med Aligned CD": excel_primary_row.get("Adj Med Aligned CD"),
            "Adj Med Aligned SIoU": excel_primary_row.get("Adj Med Aligned SIoU"),
            "VSR (%)": vsr_str,
            "Adj Med Tokens": excel_primary_row.get("Adj Med Tokens"),
            "Adj Med Lines": excel_primary_row.get("Adj Med Lines"),
            "Adj Med Ops": excel_primary_row.get("Adj Med Ops"),
        }
        adj_primary_rows_for_df.append(excel_primary_row_ordered)

        label_title = f"\n{label.upper()} (Total Samples: {a_cnt_val})\n"
        adj_primary_blocks_txt.append(label_title + tabulate([ap_values], headers=ap_headers, tablefmt="fancy_grid"))

    # Save logic
    output_json = f"{base_output_path}_per_label_metrics.json"
    output_summary = f"{base_output_path}_per_label_metrics.txt"
    output_excel = f"{base_output_path}_per_label_metrics.xlsx"

    with open(output_json, 'w') as f:
        json.dump(output_dict, f, indent=4)
    with open(output_summary, 'w') as f:
        f.write("PER-LABEL METRICS REPORT\n" + "="*100 + "\n\n")
        f.write("SECTION 1: DASHBOARD (ADJUSTED MEDIANS + VSR)\n" + "-" * 45 + "\n")
        f.write("\n".join(adj_primary_blocks_txt))
        f.write("\n\n" + "="*100 + "\nSECTION 2: DETAILED STATS (MEANS, MEDS, SD)\n" + "-" * 45 + "\n")
        f.write(tabulate(pd.DataFrame(standard_table_rows), headers='keys', tablefmt="fancy_grid", showindex=False))

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        pd.DataFrame(adj_primary_rows_for_df).to_excel(writer, sheet_name='Primary Dashboard', index=False)
        pd.DataFrame(standard_table_rows).to_excel(writer, sheet_name='Detailed Stats', index=False)

    print(f"\n[Done] Reports generated at: {output_excel}")

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("jsonl", type=str)
    parser.add_argument("--dataset", type=str, default="DeCoDELab/CADBench")
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--variable-name", type=str, required=True)
    parser.add_argument("--timeout", type=int, default=30) # 30 second default timeout
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-print", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    return parser.parse_args()

def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip(): continue
            rows.append(json.loads(line.rstrip(',')))
    return rows

def main():
    args = parse_args()
    input_path = os.path.abspath(args.jsonl)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_subdir = os.path.join(os.path.dirname(input_path), f"{base_name}_results")
    
    metrics_out = os.path.join(output_subdir, f"{base_name}_metrics.json")
    logs_out = os.path.join(output_subdir, f"{base_name}_logs.json")
    base_report_prefix = os.path.join(output_subdir, base_name)

    if args.skip_if_exists and os.path.exists(logs_out):
        print(f"Skipping: {logs_out} exists.")
        return

    os.makedirs(output_subdir, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)
    id_to_label = {}
    n = len(dataset)
    with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        for fid, label in tqdm(
            executor.map(_id_label_at, ((dataset, i) for i in range(n))),
            total=n,
            desc="Building Label Dictionary",
            disable=args.quiet,
        ):
            id_to_label[fid] = label
    outputs = load_jsonl(args.jsonl)

    evaluator = Evaluator(dataset=dataset, num_workers=args.num_workers, verbose=not args.quiet, variable_name=args.variable_name, timeout=args.timeout)
    metrics, per_item = evaluator.run(outputs)

    for item in per_item:
        item["label"] = id_to_label.get(item.get("file_id"), "unknown")

    with open(metrics_out, "w") as f: json.dump(metrics, f, indent=2)
    with open(logs_out, "w") as f: json.dump(per_item, f, indent=2)

    if not args.no_print:
        print("\n=== GLOBAL METRICS ===")
        pretty_print_metrics(metrics)
        
    calculate_advanced_per_label_metrics(per_item, base_report_prefix)

if __name__ == "__main__":
    main()