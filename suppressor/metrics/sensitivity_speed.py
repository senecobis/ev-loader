import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from matplotlib import pyplot as plt

from MaskFlow import MaskFlow
from suppressor.loss.IoUs import IoUs
from suppressor.DSEC_dataloader.provider import DatasetProvider
from suppressor.utils.utils import open_config_json,get_device


class Validate:
    def __init__(self, config):
        self.config = config

        self.dataset_provider = DatasetProvider(
            dataset_path=self.config["data"]["path"],
            num_bins=self.config["data"]["voxel_bins"],
            delta_t_ms=self.config["loader"]["event_dt_ms"],
            representation=self.config["data"]["representation"]
        )
        
        self.iou_metric = IoUs()
        self.device = get_device()

    def _evaluate_sequence(self, sequence):
        """Evaluate a sequence of data
        Args:
            sequence (list): list of data
            plot_path (str): path to save the plots
        """
        maskflow = MaskFlow()
        objects_speed = defaultdict(list)
        for ind, data in tqdm(enumerate(sequence)):
            gt = data["mask"].astype(int)
            flows = maskflow.flow_calculation_with_lables(dynamic_mask_t1=gt)
            if flows is None:
                continue
            
            for label in flows:
                current_flow = flows[label]["flow"]
                objects_speed[label].append(np.linalg.norm(current_flow))

        return objects_speed

    def _print_stats(self, sequence_id, stats):
        print(f"Sequence: {sequence_id}")
        for k, v in stats.items():
            print(f"{k}: {v}")

    def _write_json(self, data, path):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def _get_dataset(self):
        return self.dataset_provider.get_ev_imo_test_dataset(
            sequence_len=self.config["data"]["sequence_len"],
            dt=self.config["data"]["dt_100_ms"],
            mask_future_events=self.config["data"]["mask_future_events"]
            )

    def validate_model(self, save_path):
        """Validate the dynamic masker model on the test dataset

        Args:
            save_path (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.dataset = self._get_dataset()

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        results = {}

        for sequence in tqdm(self.dataset):
            sequence_id = sequence[0]["sequence_id"]
            save_sequence_path = save_path / sequence_id
            save_sequence_path.mkdir(exist_ok=True)

            stats = self._evaluate_sequence(sequence)
            for label in stats:
                plt.clf()
                speed_curve = stats[label]
                plt.plot(speed_curve)
                label_save_path = save_sequence_path / f"{label}.png"
                plt.savefig(label_save_path)

        return results

    
if __name__ == "__main__":
    config = open_config_json("suppressor/metrics/config/sensitivity_test.json")
    validator = Validate(config=config)
    validator.validate_model(
        save_path="suppressor/metrics/sensitivity_plots"
        )
    # validate_models(
    #     config_path="configs/validate.json", 
    #     models_dir="checkpoints/2025-03-03_09-51-29",
    #     save_path="results"
    #     )
