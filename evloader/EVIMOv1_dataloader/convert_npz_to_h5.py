import os, ast, shutil
from pathlib import Path
import numpy as np
import h5py


def convert_all_evimo_npz_to_h5(evimo_root, output_root):
    """
    Recursively converts all .npz files under EVIMO root to .h5 format.
    
    Args:
        evimo_root (str): Root directory of EVIMO dataset (e.g., "EVIMO1/")
        output_root (str): Output root directory (e.g., "EVIMO1_h5/")
    """
    for dirpath, _, filenames in os.walk(evimo_root):
        filenames = sorted(filenames)
        for fname in filenames:
            if not fname.endswith(".npz"):
                continue
            npz_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(npz_path, evimo_root)
            rel_dir = os.path.dirname(rel_path).replace("npz", "")  # drop "npz" level
            base_name = os.path.splitext(fname)[0]
            h5_dir = os.path.join(output_root, rel_dir)
            os.makedirs(h5_dir, exist_ok=True)
            h5_path = os.path.join(h5_dir, f"{base_name}.h5")

            print(f"Converting {npz_path} -> {h5_path}")
            try:
                archive = np.load(npz_path, allow_pickle=True)

                with h5py.File(h5_path, "w") as f:
                    # Events
                    if "events" in list(archive.keys()):
                        events = archive["events"]
                        ts = events[:, 0]
                        x = events[:, 1]
                        y = events[:, 2]
                        p = events[:, 3]
                        events_ = np.stack((x, y, ts, p), axis=-1)
                        f.create_dataset("events", data=events_)
                        f.create_dataset("events_t", data=ts)
                    else:
                        print(f"⚠️ Skipping (no event data): {npz_path}")
                        continue

                    # Optional extras
                    for key in ["index", "discretization", "depth", "classical", 'mask']:
                        if key in list(archive.keys()):
                            f.create_dataset(key, data=archive[key])
                    
                    if "meta" in list(archive.keys()):
                        meta = archive["meta"].item()
                        height = meta['meta']["res_x"]
                        width = meta['meta']["res_y"]
                        fx = meta['meta']["fx"]
                        fy = meta['meta']["fy"]
                        cx = meta['meta']["cx"]
                        cy = meta['meta']["cy"]
                        
                        f.create_dataset("height", data=height)
                        f.create_dataset("width", data=width)
                        f.create_dataset("fx", data=fx)
                        f.create_dataset("fy", data=fy)
                        f.create_dataset("cx", data=cx)
                        f.create_dataset("cy", data=cy)

                        timestamps = []
                        frames = meta['frames']
                        for element in frames:
                            timestamps.append(element['ts'])
                        f.create_dataset("ts", data=np.array(timestamps))



                        f.create_dataset("meta", data=np.string_(str(meta)))

                    else:
                        raise ValueError("Meta data not found in the archive.")
                    
                    # Save metadata
                
                print(f"✅ Saved {h5_path}")
            except Exception as e:
                print(f"❌ Error converting {npz_path}: {e}")


def extract_images_poses_timestamps_txt(evimo_txt_root, output_root):
    for dirpath, _, filenames in os.walk(evimo_txt_root):
        meta_path = os.path.join(dirpath, "meta.txt")
        if not os.path.isfile(meta_path):
            continue

        with open(meta_path, 'r') as f:
            data = ast.literal_eval(f.read())

        rel_path = os.path.relpath(dirpath, evimo_txt_root)
        rel_path_ = rel_path.replace('/txt', '')
        output_seq_dir = os.path.join(output_root, rel_path_)
        os.makedirs(output_seq_dir, exist_ok=True)

        # make sure images subfolder exists
        images_out = os.path.join(output_seq_dir, 'images')
        os.makedirs(images_out, exist_ok=True)

        # move + dump
        poses_ = []
        timestamps_ = []
        for frame in data['frames']:
            img_name = frame['classical_frame']
            src = os.path.join(dirpath, 'img', img_name)
            dst = os.path.join(images_out, img_name)

            if os.path.isfile(src):
                shutil.move(src, dst)
            else:
                print(f"Warning: {src} not found, skipping.")

            # timestamp
            ts = frame['ts']
            timestamps_.append(ts)

            # camera pose
            t = frame['cam']['pos']['t']
            q = frame['cam']['pos']['q']
            poses_.append([t['x'], t['y'], t['z'],
                        q['w'], q['x'], q['y'], q['z']])
        
        # convert to numpy arrays
        timestamps = np.array(timestamps_, dtype=np.float64)     # shape (N,)
        poses      = np.array(poses_,      dtype=np.float64)     # shape (N,7)

        np.save(os.path.join(output_seq_dir, 'timestamps.npy'), timestamps)
        np.save(os.path.join(output_seq_dir, 'poses.npy'),      poses)


        print(f"Saved {dirpath} in {output_seq_dir}")




def inspect_h5_file(h5_path, print_sample=True):
    """
    Print contents of an HDF5 file (dataset names, shapes, dtypes).
    
    Args:
        h5_path (str): Path to .h5 file to inspect.
        print_sample (bool): If True, prints the first few entries of each dataset.
    """
    print(f"\n📂 Inspecting HDF5 file: {h5_path}\n{'=' * 50}")
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            dset = f[key]
            print(f"🗃️ Dataset: '{key}'")
            print(f"   ↪ Shape: {dset.shape}")
            print(f"   ↪ Dtype: {dset.dtype}")
            if print_sample:
                try:
                    print(f"   ↪ Sample: {dset[0:2]}\n")
                except:
                    print("   ↪ Sample: (not printable)\n")
    print("=" * 50)

# Example usage:
if __name__ == "__main__":
    # convert_all_evimo_npz_to_h5(
    #     evimo_root="/data/scratch/pellerito/datasets/EVIMO1_npz",
    #     output_root="/data/scratch/pellerito/datasets/EVIMO1_"
    # )
    extract_images_poses_timestamps_txt(
        evimo_txt_root="/data/scratch/pellerito/datasets/EVIMO1_txt",
        output_root="/data/scratch/pellerito/datasets/EVIMO1_"
    )
    inspect_h5_file("/home/rpg/Downloads/EVIMO1/train/tabletop-egomotion/seq_00.h5")
    
