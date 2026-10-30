"""Grand-Challenge style entry point (offline): /input -> /output.

Reads  structured-prompt.json, prostate-biopsy-decision-clinical-data.json,
       prostate-modality-level-neural-representations.json   (the last one is read but not used)
Writes prostate-biopsy-decision.json            -> "yes" | "no"
       prostate-biopsy-decision-reasoning.json  -> {confidence, variable_weights, reveal_sequence, free_text}
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import load_bundle, predict_case

INPUT = Path(os.environ.get('CHIMERA_INPUT', '/input')); OUTPUT = Path(os.environ.get('CHIMERA_OUTPUT', '/output'))
MODEL_DIR = os.environ.get('CHIMERA_MODEL_DIR', str(Path(__file__).parent / 'model'))

def _load(name):
    f = INPUT / name
    return json.load(open(f)) if f.exists() else {}

def run():
    structured = _load('structured-prompt.json')
    clinical = _load('prostate-biopsy-decision-clinical-data.json')
    _ = _load('prostate-modality-level-neural-representations.json')      # intentionally unused
    decision, reasoning, _p = predict_case(structured, clinical, load_bundle(MODEL_DIR))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    json.dump(decision, open(OUTPUT / 'prostate-biopsy-decision.json', 'w'))
    json.dump(reasoning, open(OUTPUT / 'prostate-biopsy-decision-reasoning.json', 'w'), indent=2)
    return 0

if __name__ == '__main__':
    raise SystemExit(run())
