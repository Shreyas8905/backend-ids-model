
import io, zipfile
from scripts.dataset_inventory import detect_timestamp_columns, infer_type, inspect_csv_stream, inventory_path, write_reports

def test_timestamp_candidates():
    assert detect_timestamp_columns(["pid","event_timestamp","date"])==["event_timestamp","date"]
def test_type_inference():
    assert infer_type(["1","2"])=="integer"
    assert infer_type(["1.2","2.4"])=="number"
    assert infer_type(["true","false"])=="boolean"
    assert infer_type(["x","y"])=="string"
def test_csv_inspection():
    r=inspect_csv_stream(io.BytesIO(b"timestamp,pid\n2026-01-01,2\n"),"x.csv",10)
    assert r["columns"]==["timestamp","pid"] and r["sampled_rows"]==1
def test_directory_inventory(tmp_path):
    (tmp_path/"x.csv").write_text("time,value\n1,2\n")
    assert inventory_path(tmp_path,10)[0]["columns"]==["time","value"]
def test_zip_inventory_no_extract(tmp_path):
    z=tmp_path/"x.zip"
    with zipfile.ZipFile(z,"w") as f: f.writestr("inner/a.csv","timestamp,n\n2026-01-01,1\n")
    result=inventory_path(z,10)[0]["archive"]["members"][0]
    assert result["columns"]==["timestamp","n"] and not (tmp_path/"inner").exists()
def test_report_output(tmp_path):
    jp,cp=write_reports("ton_iot",[{"path":"x.csv","size_bytes":5,"kind":"csv"}],tmp_path)
    assert jp.exists() and cp.exists()
