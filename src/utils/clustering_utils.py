import re, json, math, shutil
from pathlib import Path
from itertools import permutations
from typing import Dict, Tuple, Optional
import json

_ASSET_RE = re.compile(
    r'^(?P<label>.+?)_(?P<x>\d+(?:\.\d+)?)_(?P<y>\d+(?:\.\d+)?)_(?P<z>\d+(?:\.\d+)?)_object_(?P<id>\d+)$'
)

def parse_asset_name(name: str) -> Tuple[str, Tuple[float,float,float], int]:
    m = _ASSET_RE.match(name)
    if not m:
        raise ValueError(f"Unparseable asset name: {name}")
    label = m.group('label')
    dims = (float(m.group('x')), float(m.group('y')), float(m.group('z')))
    oid = int(m.group('id'))
    return label, dims, oid

def l2_dim_distance(a: Tuple[float,float,float], b: Tuple[float,float,float]) -> float:
    best = math.inf
    for perm in permutations(b, 3):
        d = (a[0]-perm[0])**2 + (a[1]-perm[1])**2 + (a[2]-perm[2])**2
        if d < best:
            best = d
    return best

def neutralize_lr(label: str) -> str:
    """Remove left/right tokens but keep everything else (e.g., 'lower_left_cabinet' -> 'lower_cabinet')."""
    tokens = [t for t in label.split('_') if t not in ('left', 'right')]
    return '_'.join(tokens)

def load_objects(asset_info: Dict[str, str]):
    out = {}
    for key, val in asset_info.items():
        m = re.match(r'^object_(\d+)$', key)
        if not m:
            continue
        obj_id = int(m.group(1))
        label, dims, _ = parse_asset_name(val)
        out[obj_id] = {'label': label, 'dims': dims, 'label_lrless': neutralize_lr(label)}
    return out

def existing_mesh_ids(output_dir: Path):
    ids = set()
    if not output_dir.exists():
        return ids
    for p in output_dir.iterdir():
        if p.is_dir():
            m = re.match(r'^object_(\d+)$', p.name)
            if m:
                ids.add(int(m.group(1)))
    return ids

def _best_source_for(dst_id: int, objs: Dict[int, dict], candidates: list[int]) -> Optional[int]:
    """Choose closest-by-dimensions source from candidates; ties → smaller id."""
    if not candidates:
        return None
    tgt_dims = objs[dst_id]['dims']
    return min(candidates, key=lambda sid: (l2_dim_distance(tgt_dims, objs[sid]['dims']), sid))

def choose_source_id(dst_id: int, objs: Dict[int, dict], sources: set) -> Optional[int]:
    """
    1) Exact label match
    2) If none, match after removing left/right (e.g., left_cabinet ↔ right_cabinet)
    """
    tgt = objs[dst_id]
    # 1) Exact label
    exact = [sid for sid in sources if objs[sid]['label'] == tgt['label']]
    pick = _best_source_for(dst_id, objs, exact)
    if pick is not None:
        return pick
    # 2) Left/Right-neutralized label
    lrless = [sid for sid in sources if objs[sid]['label_lrless'] == tgt['label_lrless']]
    pick = _best_source_for(dst_id, objs, lrless)
    if pick is not None:
        return pick
    # No match found
    return None


def _rename_object_files_in_place(folder: Path, src_id: int, dst_id: int):
    """
    In the copied 'object_<dst_id>' folder, rename any files whose names
    start with 'object_<src_id>_' to 'object_<dst_id>_'.
    Also fix text references inside .obj/.mtl from object_<src_id> -> object_<dst_id>.
    """
    old_prefix = f"object_{src_id}_"
    new_prefix = f"object_{dst_id}_"

    # 1) Rename files that carry the old id in their filename
    #    e.g., object_9_mesh.obj -> object_27_mesh.obj, object_9_sample_gs.mp4 -> object_27_sample_gs.mp4
    for p in folder.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if name.startswith(old_prefix):
            new_name = new_prefix + name[len(old_prefix):]
            target = p.with_name(new_name)
            if target.exists():
                target.unlink()        # be safe if something already exists
            p.rename(target)

    # 2) Patch text inside .obj and .mtl
    #    Replace any token "object_<src_id>" with "object_<dst_id>".
    token_old = f"object_{src_id}"
    token_new = f"object_{dst_id}"
    for ext in (".obj", ".mtl"):
        for f in folder.rglob(f"*{ext}"):
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                # fallback without explicit encoding
                txt = f.read_text(errors="ignore")
            patched = txt.replace(token_old, token_new)
            if patched != txt:
                f.write_text(patched, encoding="utf-8", errors="ignore")

def copy_object_folder(output_dir: Path, src_id: int, dst_id: int, overwrite=False):
    """
    Copy output/object_<src_id> -> output/object_<dst_id>, then rename internal
    files & references to use <dst_id>.
    """
    src = output_dir / f"object_{src_id}"
    dst = output_dir / f"object_{dst_id}"
    if not src.exists():
        print(f"[warn] Source missing: {src}")
        return False
    if dst.exists():
        if not overwrite:
            print(f"[skip] {dst} exists. Use overwrite=True to replace.")
            return False
        shutil.rmtree(dst)

    shutil.copytree(src, dst)

    # After copying, fix filenames and references in the destination folder
    _rename_object_files_in_place(dst, src_id, dst_id)
    return True

def assign_missing_meshes(asset_info: Dict[str,str], output_dir: str, overwrite=False, dry_run=False):
    output_path = Path(output_dir)
    objs = load_objects(asset_info)
    have = existing_mesh_ids(output_path)
    all_ids = set(objs.keys())
    need = sorted(all_ids - have)
    plan = {}

    for dst_id in need:
        src_id = choose_source_id(dst_id, objs, have)
        if src_id is None:
            print(f"[info] No source match for object_{dst_id} "
                  f"(label='{objs[dst_id]['label']}', lrless='{objs[dst_id]['label_lrless']}'). Skipping.")
            continue
        plan[dst_id] = src_id
        if dry_run:
            print(f"[plan] object_{dst_id} <= object_{src_id} "
                  f"(label='{objs[dst_id]['label']}', via='{objs[src_id]['label']}')")
        else:
            ok = copy_object_folder(output_path, src_id, dst_id, overwrite=overwrite)
            print(f"[{'copied' if ok else 'failed'}] object_{dst_id} <= object_{src_id}")
    return plan

# ---------- Example ----------
if __name__ == "__main__":
    # Paste your full asset_info dict here
    with open("data/basement_test_2/cached_asset_info.json") as f:
        asset_info = json.load(f)
    assert isinstance(asset_info, dict), f"Expected dict, got {type(asset_info)}"
    OUTPUT_DIR = "data/basement_test_2/output"

    # Dry run first to verify choices:
    assign_missing_meshes(asset_info, OUTPUT_DIR, overwrite=False, dry_run=False)

    # Then actually copy:
    # assign_missing_meshes(asset_info, OUTPUT_DIR, overwrite=False, dry_run=False)
