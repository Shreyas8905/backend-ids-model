from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/ton_iot")
OUT = Path("data/processed/ton_iot")
WINDOW_SECONDS = 10

FILES = {
    "network": "Network_dataset_1.csv",
    "process": "Linux_process_1.csv",
    "memory": "linux_memory1.csv",
    "disk": "linux_disk_1.csv",
}


def load_csv(source, required):
    path = RAW / FILES[source]
    df = pd.read_csv(path, usecols=lambda c: c in required,
                     low_memory=False)

    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"]).copy()
    df["window_start_ts"] = (
        df["ts"].astype("int64") // WINDOW_SECONDS
    ) * WINDOW_SECONDS

    if "label" in df:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0)

    if "type" in df:
        df["type"] = df["type"].fillna("unknown").astype(str).str.lower()

    return df


def aggregate(source, numeric, categorical=None):
    categorical = categorical or []

    required = {"ts", "label", "type"} | set(numeric) | set(categorical)
    df = load_csv(source, required)

    grouped = df.groupby("window_start_ts", sort=True)
    result = pd.DataFrame(index=grouped.size().index)

    result[f"{source}_record_count"] = grouped.size()

    for col in numeric:
        values = pd.to_numeric(df[col], errors="coerce")
        df[f"__{col}"] = values
        df[f"__{col}_missing"] = values.isna().astype("int8")

        g = df.groupby("window_start_ts", sort=True)

        result[f"{source}_{col}_mean"] = g[f"__{col}"].mean()
        result[f"{source}_{col}_max"] = g[f"__{col}"].max()
        result[f"{source}_{col}_sum"] = g[f"__{col}"].sum(min_count=1)
        result[f"{source}_{col}_missing_rate"] = (
            g[f"__{col}_missing"].mean()
        )

    # Labels are targets, never model features.
    result["label"] = grouped["label"].max().astype("int8")

    def resolve_type(values):
        attacks = sorted({
            str(v) for v in values
            if str(v) not in ("normal", "unknown", "nan")
        })
        if not attacks:
            return "normal"
        return attacks[0] if len(attacks) == 1 else "multi_attack"

    result["attack_type"] = grouped["type"].agg(resolve_type)
    result = result.reset_index()

    return result


def build_network():
    return aggregate(
        "network",
        numeric=[
            "duration", "src_bytes", "dst_bytes",
            "src_pkts", "dst_pkts",
            "src_ip_bytes", "dst_ip_bytes",
            "missed_bytes",
        ],
    )


def build_linux_host():
    process = aggregate(
        "process",
        numeric=["CPU", "TRUN", "TSLPI", "TSLPU", "EXC"],
    )

    memory = aggregate(
        "memory",
        numeric=[
            "MINFLT", "MAJFLT", "VSIZE",
            "RSIZE", "VGROW", "RGROW", "MEM",
        ],
    )

    disk = aggregate(
        "disk",
        numeric=["RDDSK", "WRDSK", "WCANCL", "DSK"],
    )

    # Outer joins preserve windows even when a source has no records.
    merged = process.merge(
        memory, on="window_start_ts", how="outer",
        suffixes=("_process", "_memory")
    )

    merged = merged.merge(
        disk, on="window_start_ts", how="outer",
        suffixes=("", "_disk")
    )

    label_cols = [
        c for c in merged.columns
        if c == "label" or c.startswith("label_")
    ]

    merged["label"] = (
        merged[label_cols].max(axis=1).fillna(0).astype("int8")
    )

    type_cols = [
        c for c in merged.columns
        if c == "attack_type" or c.startswith("attack_type_")
    ]

    def resolve_merged_type(row):
        attacks = sorted({
            str(row[c]) for c in type_cols
            if pd.notna(row[c])
            and str(row[c]) not in ("normal", "unknown", "nan")
        })
        if not attacks:
            return "normal"
        return attacks[0] if len(attacks) == 1 else "multi_attack"

    merged["attack_type"] = merged.apply(resolve_merged_type, axis=1)

    # Remove source-specific labels and intermediate type columns.
    merged = merged.drop(
        columns=[
            c for c in label_cols if c != "label"
        ] + [
            c for c in type_cols if c != "attack_type"
        ]
    )

    return merged.sort_values("window_start_ts").reset_index(drop=True)


def save_profile(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    df.to_csv(path, index=False)

    print(f"\n{name}")
    print("Windows:", len(df))
    print("Attack windows:", int(df["label"].sum()))
    print("Normal windows:", int((df["label"] == 0).sum()))
    print("Attack types:")
    print(df["attack_type"].value_counts().to_string())
    print("Saved:", path)


def main():
    print("Preparing ToN_IoT 10-second windows...")

    network = build_network()
    linux_host = build_linux_host()

    save_profile(network, "network_v1")
    save_profile(linux_host, "linux_host_v1")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
