import torch
from pathlib import Path
from .HydraSequence import HydraSequence

class CachedHydraSequence(HydraSequence):
    """
    A class that extends HydraSequence to cache the results of the sequence.
    This is useful for performance optimization, especially when dealing with large datasets.
    """
    
    def __init__(self, use_cache=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = use_cache
        if self.use_cache:
            self.cache_dir = Path(self.seq_path) / "cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_file_path(self, index):
        return self.cache_dir / f"sample_{index:06d}.pt"
    
    def clear_cache(self):
        if self.cache_dir.exists():
            for file in self.cache_dir.glob("*.pt"):
                file.unlink()

    def __getitem__(self, index):
        index += 1 if index == 0 else 0
        self.indices = self._get_sequence_indices(index)

        # Use the final index in the sequence to define a single cache file
        cache_file = self._cache_file_path(index)

        if cache_file.exists():
            return torch.load(cache_file)

        events = self.preload_events(self.indices)
        sequence_data = []
        for i, idx in enumerate(self.indices):
            data = self._get_single_datapoint_by_events(idx=idx, events=events[i])
            sequence_data.append(data)

        if len(sequence_data) != self.sequence_len + 1:
            raise ValueError("Size mismatch in HydraSequence between sequence length and sequence data")

        # Save to cache
        torch.save(sequence_data, cache_file)

        return sequence_data

if __name__ == '__main__':
    import time
    seq_abs_path = Path("/data/scratch/pellerito/datasets/DSEC/train/thun_00_a")
    dsec_seq = HydraSequence(
        seq_path=seq_abs_path, 
        num_bins=2, 
        sequence_len=2, 
        max_num_grad_events=10000, 
        dt=[1, 10, 100]
        )

    for ind in range(len(dsec_seq)):
        loader_outputs = dsec_seq[ind]
        print(f"indices: {dsec_seq.indices}")
        print(f"index: {ind}")

        for loader_output in loader_outputs:
            tm = time.time()
            file_index = loader_output['file_index']
            sequence_id = loader_output['sequence_id']
            event_representation = loader_output['representation']
            dt = loader_output['sampled_dt']
            forward_flow_gt = loader_output['forward_flow_gt']
            backward_flow_gt = loader_output['backward_flow_gt']
            dynamic_mask = loader_output['dynamic_mask']

            event_list = loader_output['event_list']
            polarity_mask = loader_output['polarity_mask']
            d_event_list = loader_output['d_event_list']
            d_polarity_mask = loader_output['d_polarity_mask']
            
            frame = loader_output['frame']
            print(f"-------------------------{time.time()-tm/1000}-----------------------------")


