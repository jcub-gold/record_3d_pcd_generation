import re, json, math, shutil
from pathlib import Path
from itertools import permutations
from typing import Dict, Tuple, Optional
import json

_ASSET_RE = re.compile(
    r'^(?P<label>.+?)_(?P<x>\d+(?:\.\d+)?)_(?P<y>\d+(?:\.\d+)?)_(?P<z>\d+(?:\.\d+)?)_object_(?P<id>\d+)$'
)

"""
Function: parse_asset_name
--------------------------
name: asset name string

Parses asset name into label, dimensions, and object ID.
Returns (label, (x, y, z), id).
"""
def parse_asset_name(name: str) -> Tuple[str, Tuple[float,float,float], int]:
    m = _ASSET_RE.match(name)
    if not m:
        raise ValueError(f"Unparseable asset name: {name}")
    label = m.group('label')
    dims = (float(m.group('x')), float(m.group('y')), float(m.group('z')))
    oid = int(m.group('id'))
    return label, dims, oid

"""
Function: l2_dim_distance
------------------------
a: tuple of dimensions (x, y, z)
b: tuple of dimensions (x, y, z)

Computes the minimum squared L2 distance between a and all permutations of b.
Returns the best (minimum) distance.
"""
def l2_dim_distance(a: Tuple[float,float,float], b: Tuple[float,float,float]) -> float:
    best = math.inf
    for perm in permutations(b, 3):
        d = (a[0]-perm[0])**2 + (a[1]-perm[1])**2 + (a[2]-perm[2])**2 # NOTE: it may not be best to look at all permutations of b, as x,y,z should already coorelate to width, height, depth
        if d < best:
            best = d
    return best

"""
Function: neutralize_lr
-----------------------
label: asset label string

Removes 'left' and 'right' tokens from the label.
Returns the neutralized label.
"""
def neutralize_lr(label: str) -> str:
    tokens = [t for t in label.split('_') if t not in ('left', 'right')]
    return '_'.join(tokens)

"""
Function: load_objects
----------------------
asset_info: dictionary mapping object keys to asset names

Loads object info from asset_info, parsing label, dimensions, and left/right-neutralized label.
Returns a dictionary mapping object IDs to their info.
"""
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

"""
Function: existing_mesh_ids
--------------------------
output_dir: Path to the output directory

Returns a set of object IDs for which mesh folders exist in output_dir.
"""
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

"""
Function: _best_source_for
--------------------------
dst_id: destination object ID
objs: dictionary of object info
candidates: list of candidate source IDs

Chooses the closest-by-dimensions source from candidates; ties go to smaller ID.
Returns the best source ID or None.
"""
def _best_source_for(dst_id: int, objs: Dict[int, dict], candidates: list[int]) -> Optional[int]:
    if not candidates:
        return None
    tgt_dims = objs[dst_id]['dims']
    return min(candidates, key=lambda sid: (l2_dim_distance(tgt_dims, objs[sid]['dims']), sid)) # TODO: A tie should look at the average color of the object or other features from the objects segmented frame

"""
Function: choose_source_id
-------------------------
dst_id: destination object ID
objs: dictionary of object info
sources: set of available source IDs

Chooses a source ID for dst_id by:
    1) Exact label match
    2) If none, match after removing left/right tokens (via _best_source_for)
Returns the best matching source ID or None.
"""
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

"""
Function: _rename_object_files_in_place
---------------------------------------
folder: Path to the destination object folder
src_id: source object ID
dst_id: destination object ID

Renames files and internal references from src_id to dst_id inside the folder.
"""
def _rename_object_files_in_place(folder: Path, src_id: int, dst_id: int):
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

"""
Function: copy_object_folder
---------------------------
output_dir: Path to the output directory
src_id: source object ID
dst_id: destination object ID
overwrite: whether to overwrite existing destination folder

Copies the source object folder to the destination, renaming files and references (via _rename_object_files_in_place).
Returns True if copied, False otherwise.
"""
def copy_object_folder(output_dir: Path, src_id: int, dst_id: int, overwrite=False):
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
    _rename_object_files_in_place(dst, src_id, dst_id)
    return True

"""
Function: assign_missing_meshes
-------------------------------
asset_info: dictionary mapping object keys to asset names
output_dir: path to the output directory containing object folders
overwrite: whether to overwrite existing folders (default False)
dry_run: if True, only print planned assignments without copying (default False)

Assigns missing mesh folders for objects by copying the closest matching mesh from existing objects.
Uses label and dimension matching, including left/right-neutralized labels.
Prints info about each assignment and returns a mapping from destination IDs to source IDs.
"""
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
