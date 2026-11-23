
#!/usr/bin/env python3
"""Read-only inventory of dataset files and archive members; archives are not extracted."""
import argparse, csv, gzip, io, json, tarfile, zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MAX_SAMPLE_BYTES = 2 * 1024 * 1024
HINTS = ("time", "timestamp", "date", "datetime", "event_time", "start_time")

def detect_timestamp_columns(columns):
    return [c for c in columns if any(h in c.lower() for h in HINTS)]

def infer_type(values):
    vals = [str(v).strip() for v in values if v is not None and str(v).strip()]
    if not vals: return "empty_or_null"
    if {v.lower() for v in vals} <= {"true","false","yes","no"}: return "boolean"
    try:
        [int(v) for v in vals]; return "integer"
    except ValueError: pass
    try:
        [float(v) for v in vals]; return "number"
    except ValueError: return "string"

def inspect_csv_stream(stream, name, sample_rows=100):
    raw = stream.read(MAX_SAMPLE_BYTES); stream.seek(0)
    delimiter = "\\t" if Path(name).suffix.lower() == ".tsv" else ","
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig", errors="replace")), delimiter=delimiter)
    columns = reader.fieldnames or []
    rows = [row for _, row in zip(range(sample_rows), reader)]
    return {"columns":columns,"sampled_rows":len(rows),
      "column_types":{c:infer_type([r.get(c,"") for r in rows]) for c in columns},
      "timestamp_candidates":detect_timestamp_columns(columns),
      "sample_truncated_at_bytes":len(raw)==MAX_SAMPLE_BYTES}

def inspect_json_stream(stream, name):
    raw = stream.read(MAX_SAMPLE_BYTES)
    result = {"sample_bytes":len(raw),"sample_truncated":len(raw)==MAX_SAMPLE_BYTES}
    try:
        txt = raw.decode("utf-8-sig", errors="replace")
        obj = json.loads(next((x for x in txt.splitlines() if x.strip()), "{}")) if name.lower().endswith((".jsonl",".ndjson")) else json.loads(txt)
        if isinstance(obj,dict): result["top_level_keys"] = list(obj)
        elif isinstance(obj,list):
            result["top_level_type"]="array"
            result["first_item_keys"]=list(obj[0]) if obj and isinstance(obj[0],dict) else []
        else: result["top_level_type"]=type(obj).__name__
    except Exception as exc: result["parse_note"]=str(exc)
    return result

def inspect_member(stream, name, sample_rows):
    low=name.lower()
    if low.endswith((".csv",".tsv")): return inspect_csv_stream(stream,name,sample_rows)
    if low.endswith((".json",".jsonl",".ndjson")): return inspect_json_stream(stream,name)
    if low.endswith(".parquet"): return {"note":"Parquet member not decoded in archive; inspect after selective extraction."}
    return {"note":"Member not deeply inspected."}

def inspect_zip(path, sample_rows):
    members=[]
    with zipfile.ZipFile(path) as z:
        for m in z.infolist():
            if m.is_dir(): continue
            e={"name":m.filename,"size_bytes":m.file_size}
            try:
                if m.filename.lower().endswith((".csv",".tsv",".json",".jsonl",".ndjson")):
                    with z.open(m) as s: e.update(inspect_member(s,m.filename,sample_rows))
                elif m.filename.lower().endswith(".parquet"):
                    e["note"]="Parquet member not decoded; inspect after selective extraction."
            except Exception as exc: e["error"]=f"{type(exc).__name__}: {exc}"
            members.append(e)
    return {"member_count":len(members),"members":members}

def inspect_tar(path, sample_rows):
    members=[]
    with tarfile.open(path,"r:*") as t:
        for m in t:
            if not m.isfile(): continue
            e={"name":m.name,"size_bytes":m.size}
            try:
                if m.name.lower().endswith((".csv",".tsv",".json",".jsonl",".ndjson")):
                    s=t.extractfile(m)
                    if s: e.update(inspect_member(s,m.name,sample_rows))
                elif m.name.lower().endswith(".parquet"):
                    e["note"]="Parquet member not decoded; inspect after selective extraction."
            except Exception as exc: e["error"]=f"{type(exc).__name__}: {exc}"
            members.append(e)
    return {"member_count":len(members),"members":members}

def inspect_file(path, sample_rows):
    ext=path.suffix.lower()
    e={"path":str(path),"size_bytes":path.stat().st_size,"kind":ext.lstrip(".")}
    try:
        if ext in (".csv",".tsv"):
            with path.open("rb") as s: e.update(inspect_csv_stream(s,path.name,sample_rows))
        elif ext in (".json",".jsonl",".ndjson"):
            with path.open("rb") as s: e.update(inspect_json_stream(s,path.name))
        elif ext==".parquet":
            try:
                import pyarrow.parquet as pq
                pf=pq.ParquetFile(path); names=pq.read_schema(path).names
                e.update({"rows":pf.metadata.num_rows,"row_groups":pf.metadata.num_row_groups,
                          "columns":names,"timestamp_candidates":detect_timestamp_columns(names)})
            except ImportError: e["note"]="Install pyarrow to inspect Parquet metadata."
        elif ext==".zip": e["archive"]=inspect_zip(path,sample_rows)
        elif ext in (".tar",".tgz",".gz",".bz2",".xz") and tarfile.is_tarfile(path):
            e["archive"]=inspect_tar(path,sample_rows)
        elif ext==".gz" and path.name.lower().endswith((".csv.gz",".tsv.gz")):
            with gzip.open(path,"rb") as s: e["compressed_table"]=inspect_csv_stream(s,path.name[:-3],sample_rows)
        else: e["note"]="File type not deeply inspected."
    except Exception as exc: e["error"]=f"{type(exc).__name__}: {exc}"
    return e

def inventory_path(path,sample_rows):
    if not path.exists(): raise FileNotFoundError(path)
    if path.is_file(): return [inspect_file(path,sample_rows)]
    return [inspect_file(p,sample_rows) for p in sorted(path.rglob("*")) if p.is_file()]

def summarize(entries):
    return {"file_count":len(entries),"total_size_bytes":sum(e.get("size_bytes",0) for e in entries),
            "extensions":dict(sorted(Counter(Path(e["path"]).suffix.lower() or "<none>" for e in entries).items())),
            "files_with_errors":sum("error" in e for e in entries)}

def write_reports(dataset,entries,out):
    out.mkdir(parents=True,exist_ok=True)
    payload={"dataset":dataset,"generated_at_utc":datetime.now(timezone.utc).isoformat(),
             "summary":summarize(entries),"files":entries}
    jp=out/f"{dataset}_inventory.json"; cp=out/f"{dataset}_inventory.csv"
    jp.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8")
    with cp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["path","size_bytes","kind","columns","timestamp_candidates","error"]); w.writeheader()
        for e in entries:
            w.writerow({"path":e.get("path"),"size_bytes":e.get("size_bytes"),"kind":e.get("kind"),
              "columns":";".join(e.get("columns",[])),"timestamp_candidates":";".join(e.get("timestamp_candidates",[])),
              "error":e.get("error","")})
    return jp,cp

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path",required=True,type=Path); p.add_argument("--dataset",required=True,choices=["ton_iot","lo2v2","rcaeval"])
    p.add_argument("--output",type=Path,default=Path("audit")); p.add_argument("--sample-rows",type=int,default=100)
    a=p.parse_args()
    if a.sample_rows<1: p.error("--sample-rows must be >= 1")
    entries=inventory_path(a.path,a.sample_rows); jp,cp=write_reports(a.dataset,entries,a.output)
    print(json.dumps(summarize(entries),indent=2)); print(f"JSON report: {jp}"); print(f"CSV report: {cp}")
if __name__=="__main__": main()
