import json
from collections import defaultdict
import numpy as np
import argparse


def aggregate_results(path_to_json, save_path):
    # Load your JSON
    with open(path_to_json, 'r') as f:
        data = json.load(f)

    # Create structures to accumulate sums
    accumulators = defaultdict(lambda: defaultdict(list))

    # Loop over each sequence
    for seq_name, metrics in data.items():
        # The object type is before '_seq', e.g., 'box', 'floor'
        object_type = seq_name.split('_seq')[0]
        
        for metric, value in metrics.items():
            if isinstance(value, list):
                # For IoU lists, take the nanmean of the list before appending
                name = metric.split('/')
                if len(name) < 2:
                    name.append('')
                metric = name[0] + '_positive_class/' + name[1]
                accumulators[object_type][metric].append(value[0])

                metric_2 = name[0] + '_mean_of_IoU/' + name[1]
                accumulators[object_type][metric_2].append(np.nanmean(value))
            else:
                # For scalar values (mIoU, pIoU)
                accumulators[object_type][metric].append(value)

    # Now compute the nanmean for each object type
    final_means = {}

    for object_type, metric_values in accumulators.items():
        final_means[object_type] = {}
        for metric, values in metric_values.items():
            final_means[object_type][metric] = float(np.nanmean(values))

    # Save the result to a new JSON
    with open(save_path, 'w') as f:
        json.dump(final_means, f, indent=4)

    print(json.dumps(final_means, indent=4))

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Aggregate results from a JSON file.")
    parser.add_argument("path", type=str, help="Path to input JSON file")
    args = parser.parse_args()
    
    aggregate_results(path_to_json=f"{args.path}/results.json", save_path=f"{args.path}/aggregated_results.json")
    
    # Example usage:
    # python suppressor/utils/aggregate_results.py /home/pellerito/suppressor_/results/EV-IMO_2025-04-30_14-08-02/model_epoch_20