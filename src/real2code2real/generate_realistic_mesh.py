import os
from argparse import ArgumentParser
from PIL import Image
from submodules.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from src.utils.mesh_utils import save_object

def load_images(img_dir):
    exts = {".png", ".jpg", ".jpeg"}
    files = sorted(
        [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts]
    )
    return [Image.open(os.path.join(img_dir, f)).convert("RGB") for f in files]

def main():
    parser = ArgumentParser("TRELLIS mesh generator")
    parser.add_argument("--scene_name", "-s", required=True, type=str)
    args = parser.parse_args()
    
    source_dir = f"data/{args.scene_name}/multiview"
    output_path = f"data/{args.scene_name}/output"


    os.makedirs(output_path, exist_ok=True)

    pipe = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipe.cuda()

    object_dirs = [
        d for d in sorted(os.listdir(source_dir))
        if os.path.isdir(os.path.join(source_dir, d))
    ]

    if not object_dirs:
        object_dirs = ["."]
        base_dir = source_dir
    else:
        base_dir = source_dir

    for obj_name in object_dirs:
        in_dir = base_dir if obj_name == "." else os.path.join(base_dir, obj_name)

        gen_dir = os.path.join(in_dir, "generation_state")
        img_dir = gen_dir if os.path.isdir(gen_dir) else in_dir

        images = load_images(img_dir)
        if not images:
            print(f"[skip] {obj_name}: no images found in {img_dir}")
            continue
        
        if len(images) == 1:
            output = pipe.run(images[0], seed=1)
        else:
            output = pipe.run_multi_image(images, seed=1)

        out_name = obj_name if obj_name != "." else "object"
        out_dir = output_path if obj_name == "." else os.path.join(output_path, obj_name)
        os.makedirs(out_dir, exist_ok=True)

        save_object(output, out_dir, out_name)

        print(f"[done] Saved mesh for {out_name} to {out_dir}")
        return


if __name__ == "__main__":
    main()