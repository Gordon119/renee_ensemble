from pathlib import Path
import re
import pandas as pd

ROOT = Path("mem_profiles")

# Matches filenames like: mem_LF-Amazon-131K_0.1.csv  -> ratio = 0.1
RATIO_RE = re.compile(r"_(\d+(?:\.\d+)?)\.csv$", re.IGNORECASE)

def short_model_name(tf: str) -> str:
    """Convert full tf string into a compact tag like 'distilbert' or 'roberta-large'."""
    if not isinstance(tf, str) or not tf:
        return "unknown"
    s = tf.lower()

    if "distilbert" in s:
        return "distilbert"
    if "roberta-large" in s:
        return "roberta-large"

    # fallback: last segment, remove trailing version like -v1, -v2...
    name = tf.split("/")[-1]
    name = re.sub(r"-v\d+$", "", name, flags=re.IGNORECASE)
    return name

rows = []

for csv_path in ROOT.rglob("*.csv"):
    # Infer ratio from filename (if missing, treat as "full" = 1.0)
    if "Amazon-670K" in csv_path.name:
        continue
    m = RATIO_RE.search(csv_path.name)
    ratio_num = float(m.group(1)) if m else 1.0
    ratio_label = f"{ratio_num:g}" if m else "full"

    df = pd.read_csv(csv_path)

    if "epoch0_peak_reserved_mib" not in df.columns:
        continue

    dataset = csv_path.parent.name

    tf = df["tf"].iloc[0] if "tf" in df.columns and len(df) > 0 else "unknown"
    model_tag = short_model_name(tf)  # (modified)

    # Pull batch_size + maxlen from columns (preferred) with safe fallbacks
    batch_size = df["batch_size"].iloc[0] if "batch_size" in df.columns and len(df) > 0 else "unknown"  # (modified)
    max_length = df["maxlen"].iloc[0] if "maxlen" in df.columns and len(df) > 0 else "unknown"          # (modified)

    # New label: data/model/batch_size/max_length
    model = f"{dataset}/{model_tag}/{batch_size}/{max_length}"  # (modified)

    # If multiple rows exist, take the maximum peak as representative
    peak_mib = pd.to_numeric(df["epoch0_peak_reserved_mib"], errors="coerce").max()

    rows.append(
        {
            "model": model,
            "ratio": ratio_label,
            "ratio_num": ratio_num,
            "epoch0_peak_reserved_gib": peak_mib / 1024,  # MiB -> GiB
        }
    )

out = pd.DataFrame(rows).dropna(subset=["epoch0_peak_reserved_gib"])
out = out.sort_values(["model", "ratio_num"]).drop(columns=["ratio_num"])

print("\n=== Long table (GiB) ===")
print(out.to_string(index=False))

pivot = out.pivot_table(
    index="model",
    columns="ratio",
    values="epoch0_peak_reserved_gib",
    aggfunc="max",
).sort_index()

print("\n=== Pivot table (GiB) ===")
print(pivot.round(3).to_markdown())

out.to_csv("epoch0_peak_reserved_gib_long.csv", index=False)
pivot.round(6).to_csv("epoch0_peak_reserved_gib_pivot.csv")
print("\nSaved: epoch0_peak_reserved_gib_long.csv, epoch0_peak_reserved_gib_pivot.csv")
