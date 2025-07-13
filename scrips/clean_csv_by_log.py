
import os
import pandas as pd

csv_path = "../data/images/birds.csv"
log_path = "../data/missing_or_mismatch_log.txt"
base_dir = "../data/images"

df = pd.read_csv(csv_path)

paths_to_remove = []
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        if "→" in line:
            rel_path = line.split("→")[0].split("]")[-1].strip()
            paths_to_remove.append(rel_path)

paths_to_remove = list(set(paths_to_remove))

for rel_path in paths_to_remove:
    abs_path = os.path.join(base_dir, rel_path).replace("\\\\", "/").replace("\\", "/")
    if os.path.exists(abs_path):
        os.remove(abs_path)
        print(f"[DELETED] {abs_path}")
    else:
        print(f"[NOT FOUND] {abs_path}")

df_filtered = df[~df['filepaths'].isin(paths_to_remove)]

filtered_path = "/3DBirdReconstruction/data/images/birds.csv"
df_filtered.to_csv(filtered_path, index=False, encoding="utf-8")
print(f"[DONE] Filtered CSV saved to: {filtered_path}")
