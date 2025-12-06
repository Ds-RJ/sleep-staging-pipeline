
import os
import argparse
import yaml
import pickle
from typing import Dict, Any, List

from utils import find_pairs, read_hyp_labels, read_raw_signal, epoch_raw_30s, aasm_from_rk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .rec/.edf and .hyp files")
    parser.add_argument("--out_pickle", type=str, required=True, help="Output pickle path for synced epochs")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    pairs = find_pairs(args.data_dir)
    if not pairs:
        print("No paired signal+hyp files found.")
        return

    synced: Dict[str, Any] = {}
    valid_codes = set(int(k) for k in cfg["rk_codes_map"].keys())
    rk_map = {int(k): v for k, v in cfg["rk_codes_map"].items()}

    for sig_path, hyp_path in pairs:
        try:
            raw = read_raw_signal(sig_path)
            epochs_data, n_epochs_sig, samples_per_epoch, sfreq = epoch_raw_30s(raw, cfg["epoch_len_sec"])
            labels = read_hyp_labels(hyp_path, valid_codes)
            n_epochs_hyp = len(labels)

            if n_epochs_sig != n_epochs_hyp:
                n_min = min(n_epochs_sig, n_epochs_hyp)
                print(f"Aligning {os.path.basename(sig_path)}: signal {n_epochs_sig} vs hyp {n_epochs_hyp} -> {n_min}")
                epochs_data = epochs_data[:n_min]
                labels = labels[:n_min]

            label_names = [rk_map.get(code, "Unknown") for code in labels]
            if cfg.get("use_aasm", True):
                label_names = [aasm_from_rk(n) for n in label_names]

            # Optionally drop Movement/Undefined
            keep_idx: List[int] = []
            for i, name in enumerate(label_names):
                if cfg.get("drop_movement", True) and name.lower().startswith("movement"):
                    continue
                if cfg.get("drop_undefined", True) and name in ["Unknown", "Undefined"]:
                    continue
                keep_idx.append(i)

            if len(keep_idx) < len(label_names):
                epochs_data = epochs_data[keep_idx]
                labels = [labels[i] for i in keep_idx]
                label_names = [label_names[i] for i in keep_idx]

            synced[os.path.basename(sig_path)] = {
                "channel_names": raw.ch_names,
                "sfreq": sfreq,
                "samples_per_epoch": samples_per_epoch,
                "epoch_len_sec": cfg["epoch_len_sec"],
                "epochs_data_shape": (len(epochs_data), len(raw.ch_names), samples_per_epoch),
                "epochs_data": epochs_data,      # ndarray (n_epochs, n_channels, samples)
                "label_names": label_names,      # AASM or R&K names per epoch
                "label_codes": labels            # raw codes per epoch
            }
            raw.close()
        except Exception as e:
            print(f"Error processing {sig_path}: {e}")

    os.makedirs(os.path.dirname(args.out_pickle), exist_ok=True)
    with open(args.out_pickle, "wb") as f:
        pickle.dump(synced, f)
    print(f"Saved synced epochs -> {args.out_pickle}")


if __name__ == "__main__":
    main()
