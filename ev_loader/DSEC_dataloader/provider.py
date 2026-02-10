import time
import torch
from pathlib import Path
from .Sequence import Sequence
from .TestSequence import TestSequence
from .RecurrentSequence import RecurrentSequence
from .FlowSequence import FlowSequence
from .FlowTestSequence import FlowTestSequence
from .SemanticSequence import SemanticSequence
from .RawSequence import RawSequence
from .ByEvIdxSequence import ByEvIdxSequence
from .TimeSurfaceSequence import TimeSurfaceSequence
class DatasetProvider:
    def __init__(self, dataset_path: Path, delta_t_ms: int=50, num_bins=15, representation: str = ''):
        dataset_path = Path(dataset_path)
        self.train_path = dataset_path / 'train'
        self.test_path = dataset_path / 'test'
        self.validation_path = dataset_path / 'validation'
        self.delta_t_ms = delta_t_ms
        self.num_bins = num_bins
        self.representation = representation
        
        assert dataset_path.is_dir(), str(dataset_path)
        
    def get_train_dataset(self, load_opt_flow: bool = True):
        assert self.train_path.is_dir(), str(self.train_path)
        train_sequences = list()
        for child in sorted(self.train_path.iterdir()):
            train_sequences.append(Sequence(seq_path=child, 
                                            mode='train', 
                                            delta_t_ms=self.delta_t_ms, 
                                            num_bins=self.num_bins, 
                                            representation=self.representation,
                                            load_opt_flow=load_opt_flow
                                            ))
        return torch.utils.data.ConcatDataset(train_sequences)
    
    def get_recurrent_train_dataset(self, sequence_len:int = 2):
        assert self.train_path.is_dir(), str(self.train_path)
        train_sequences = list()
        for child in sorted(self.train_path.iterdir()):
            train_sequences.append(RecurrentSequence(seq_path=child, mode='train', delta_t_ms=self.delta_t_ms, num_bins=self.num_bins, sequence_len=sequence_len, representation=self.representation))
        return torch.utils.data.ConcatDataset(train_sequences)
    
    def get_flow_train_dataset(self, sequence_len:int = 2, max_num_grad_events:int = 10000, dt:int = 0):
        assert self.train_path.is_dir(), str(self.train_path)
        train_sequences = list()
        for child in sorted(self.train_path.iterdir()):
            train_sequences.append(FlowSequence(seq_path=child, mode='train', delta_t_ms=self.delta_t_ms, num_bins=self.num_bins, sequence_len=sequence_len, representation=self.representation, max_num_grad_events=max_num_grad_events, dt=dt))
        return torch.utils.data.ConcatDataset(train_sequences)
    
    def get_flow_train_dataset_astest(self, sequence_len:int = 2, max_num_grad_events:int = 10000, dt:list = [0.0]):
        assert self.train_path.is_dir(), str(self.train_path)
        train_sequences = list()
        for child in sorted(self.train_path.iterdir()):
            train_sequences.append(FlowSequence(seq_path=child, mode='train', delta_t_ms=self.delta_t_ms, num_bins=self.num_bins, sequence_len=sequence_len, representation=self.representation, max_num_grad_events=max_num_grad_events, dt=dt))
        return train_sequences
    
    def get_flow_test_dataset(self):
        assert self.test_path.is_dir(), str(self.test_path)
        test_sequences = list()
        for child in sorted(self.test_path.iterdir()):
            test_sequences.append(FlowTestSequence(seq_path=child, mode='test', delta_t_ms=self.delta_t_ms, num_bins=self.num_bins, representation=self.representation))
        return test_sequences
    
    def get_test_dataset(self, FPS=None):
        assert self.test_path.is_dir(), str(self.test_path)
        test_sequences = list()
        for child in sorted(self.test_path.iterdir()):
            test_sequences.append(TestSequence(
                seq_path=child, mode='test', delta_t_ms=self.delta_t_ms, num_bins=self.num_bins, representation=self.representation, FPS=FPS))
        return test_sequences
    
    def get_semantic_test_dataset(self, class_format='19'):
        """
        Get semantic segmentation datasets from the test directory.
        
        Args:
            class_format (str): Which class format to use ('11' or '19')
            
        Returns:
            list: List of SemanticSequence objects.
        """
        assert self.test_path.is_dir(), str(self.test_path)
        test_sequences = list()
        for child in sorted(self.test_path.iterdir()):
            test_sequences.append(SemanticSequence(
                seq_path=child,
                mode='test',
                delta_t_ms=self.delta_t_ms,
                num_bins=self.num_bins,
                representation=self.representation,
                class_format=class_format
            ))
            if not test_sequences[-1].semantics_exists:
                test_sequences.pop()
        return test_sequences
    
    def get_raw_train_dataset(self, num_events: int):
        assert self.train_path.is_dir(), str(self.train_path)
        train_sequences = list()
        for child in sorted(self.train_path.iterdir()):
            train_sequences.append(RawSequence(seq_path=child, 
                                            mode='train', 
                                            delta_t_ms=self.delta_t_ms, 
                                            num_bins=self.num_bins, 
                                            representation=self.representation,
                                            num_events=num_events
                                            ))
        return torch.utils.data.ConcatDataset(train_sequences)
    
    def get_byIdx_train_dataset(self, num_events: int, voxels_subsample_factor: int):
        assert self.train_path.is_dir(), str(self.train_path)
        train_sequences = list()
        for child in sorted(self.train_path.iterdir()):
            train_sequences.append(ByEvIdxSequence(seq_path=child, 
                                            mode='train', 
                                            num_bins=self.num_bins, 
                                            representation=self.representation,
                                            num_events=num_events,
                                            voxels_subsample_factor=voxels_subsample_factor
                                            ))
        return torch.utils.data.ConcatDataset(train_sequences)
    
    def get_byIdx_test_dataset(self, num_events: int):
        assert self.test_path.is_dir(), str(self.test_path)
        test_sequences = list()
        for child in sorted(self.test_path.iterdir()):
            test_sequences.append(ByEvIdxSequence(seq_path=child, 
                                            mode='test', 
                                            num_bins=self.num_bins, 
                                            representation=self.representation,
                                            num_events=num_events,
                                            voxels_subsample_factor=-1
                                            ))
        return test_sequences
    
    def get_time_surface_train_dataset(self, num_events: int, rep_subsample_factor: int):
        assert self.train_path.is_dir(), str(self.train_path)
        train_sequences = list()
        for child in sorted(self.train_path.iterdir()):
            train_sequences.append(TimeSurfaceSequence(seq_path=child, 
                                            mode='train', 
                                            num_bins=self.num_bins, 
                                            representation=self.representation,
                                            num_events=num_events,
                                            rep_subsample_factor=rep_subsample_factor
                                            ))
        return torch.utils.data.ConcatDataset(train_sequences)
    
    def get_time_surface_test_dataset(self, num_events: int):
        assert self.test_path.is_dir(), str(self.test_path)
        test_sequences = list()
        for child in sorted(self.test_path.iterdir()):
            test_sequences.append(TimeSurfaceSequence(seq_path=child, 
                                            mode='test', 
                                            num_bins=self.num_bins, 
                                            representation=self.representation,
                                            num_events=num_events,
                                            rep_subsample_factor=-1
                                            ))
        return test_sequences

if __name__ == "__main__":
    dsec_dir = "/data/scratch/pellerito/datasets/DSEC"
    num_bins = 15
    dataset_provider = DatasetProvider(dsec_dir, num_bins=num_bins, representation="stack")
    test_dataset = dataset_provider.get_high_freq_hydra_train_dataset()
    for i, seq in enumerate(test_dataset):
        print(f"Sequence {i}: {seq}")