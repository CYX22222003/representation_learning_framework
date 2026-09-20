"""Freeze or explicitly execute the Phase 3 framework price matrix."""

from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
from typing import Sequence
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase3_encoder_common import DEFAULT_FEATURE_NPZ,DEFAULT_PRICE_LABELS,DEFAULT_PROCESSED_NPZ,PHASE3_EXPERIMENT_ROOT,sha256_file,write_json  # noqa:E402
from phase3_price_common import PRICE_CONFIGS,run_root  # noqa:E402
from extract_phase3_features import validate_features  # noqa:E402
from prepare_phase3_price_labels import validate_labels  # noqa:E402
MANIFEST=PHASE3_EXPERIMENT_ROOT/"manifests"/"price_framework_seed0.json"

def commands(processed,features,labels,device):
    python=str((ROOT/".venv/bin/python3").resolve()); script=str((ROOT/"scripts_v2/train_phase3_price.py").resolve())
    return [[python,script,"--config-id",cid,"--processed-npz",str(processed.resolve()),"--features-npz",str(features.resolve()),"--labels-npz",str(labels.resolve()),"--run-root",str(run_root(cid).resolve()),"--device",device] for cid in PRICE_CONFIGS]
def freeze(processed,features,labels,device):
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    validate_features(features,processed)
    validate_labels(labels,processed)
    payload={"phase":3,"task":"price_prediction","status":"frozen_not_executed","processed_sha256":sha256_file(processed),"features_sha256":sha256_file(features),"labels_sha256":sha256_file(labels),"configurations":list(PRICE_CONFIGS),"commands":commands(processed,features,labels,device)}; write_json(MANIFEST,payload); return payload
def execute(processed,features,labels):
    if not MANIFEST.is_file(): raise FileNotFoundError("freeze and review the price matrix first")
    m=json.loads(MANIFEST.read_text()); current=(sha256_file(processed),sha256_file(features),sha256_file(labels)); expected=(m["processed_sha256"],m["features_sha256"],m["labels_sha256"])
    if current!=expected: raise ValueError("price prerequisite hashes differ from frozen matrix")
    for command in m["commands"]:
        if Path(command[1]).resolve().parent!=(ROOT/"scripts_v2").resolve(): raise ValueError("non-scripts_v2 command rejected")
        subprocess.run(command,cwd=ROOT,check=True)
def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--processed-npz",type=Path,default=DEFAULT_PROCESSED_NPZ); p.add_argument("--features-npz",type=Path,default=DEFAULT_FEATURE_NPZ); p.add_argument("--labels-npz",type=Path,default=DEFAULT_PRICE_LABELS); p.add_argument("--device",default="cuda"); p.add_argument("--execute",action="store_true"); a=p.parse_args(argv)
    try:
        if a.execute: execute(a.processed_npz,a.features_npz,a.labels_npz); print("Phase 3 price matrix complete")
        else: m=freeze(a.processed_npz,a.features_npz,a.labels_npz,a.device); print(f"Price matrix frozen, not executed: {MANIFEST} ({len(m['commands'])} runs)")
        return 0
    except FileExistsError as exc: print(exc,file=sys.stderr); return 2
    except Exception as exc: print(f"Phase 3 price launcher failed: {exc}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
