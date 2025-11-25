import time
import torch
from pathlib import Path
from .sequence import Sequence
from .TestSequence import TestSequence
from .RecurrentSequence import RecurrentSequence
from .FlowSequence import FlowSequence
from .FlowTestSequence import FlowTestSequence
from .HydraSequence import HydraSequence
from .HydraTestSequence import HydraTestSequence
from .SemanticSequence import SemanticSequence
from ..EVIMOv1_dataloader.EVIMOSequence import EVIMOSequence
from ..EVIMOv1_dataloader.EVIMOTestSequence import EVIMOTestSequence
from ..EVIMOv1_dataloader.EV_IMOSequence import EV_IMOSequence
from ..EVIMOv1_dataloader.EVIMOFramesSequence import EVIMOFramesSequence
from ..EVIMOv1_dataloader.EVIMOTestSequenceImportByNumber import EVIMOTestSequenceImportByNumber
from ..EVIMOv1_dataloader.EVIMOSequenceRandByNumber import EVIMOSequenceRandByNumber
from ..EVIMOv1_dataloader.EVIMOTestSequenceByNumber import EVIMOTestSequenceByNumber
from .HighFreqHydraSequence import HighFreqHydraSequence

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
        
    def get_train_dataset(self):
        assert self.train_path.is_dir(), str(self.train_path)
        train_sequences = list()
        for child in sorted(self.train_path.iterdir()):
            train_sequences.append(Sequence(seq_path=child, mode='train', delta_t_ms=self.delta_t_ms, num_bins=self.num_bins, representation=self.representation))
        return train_sequences
    
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
            test_sequences.append(TestSequence(seq_path=child, mode='test', delta_t_ms=self.delta_t_ms, num_bins=self.num_bins, representation=self.representation, FPS=FPS))
        return test_sequences
    
    def get_val_dataset(self):
        assert self.validation_path.is_dir(), str(self.validation_path)
        validation_sequences = list()
        for child in sorted(self.validation_path.iterdir()):
            validation_sequences.append(TestSequence(seq_path=child, mode='validation', delta_t_ms=self.delta_t_ms, num_bins=self.num_bins, representation=self.representation))
        return validation_sequences
    
    def get_hydra_train_dataset(
            self, 
            sequence_len:int = 2, 
            max_num_grad_events:int = 10000, 
            dt:int = [0,100], 
            augment:list = [], 
            augment_prob:list = [],
            multiple_batches:bool = True
            ):
        assert self.train_path.is_dir(), str(self.train_path)
        train_sequences = list()
        for child in sorted(self.train_path.iterdir()):
            train_sequences.append(HydraSequence(
                seq_path=child, 
                mode='train', 
                delta_t_ms=self.delta_t_ms, 
                num_bins=self.num_bins, 
                sequence_len=sequence_len, 
                representation=self.representation, 
                max_num_grad_events=max_num_grad_events, 
                dt=dt,
                augment=augment,
                augment_prob=augment_prob,
                multiple_batches=multiple_batches
                ))
        return torch.utils.data.ConcatDataset(train_sequences)
    
    def get_hydra_test_dataset(self):
        assert self.test_path.is_dir(), str(self.test_path)
        test_sequences = list()
        for child in sorted(self.test_path.iterdir()):
            test_sequences.append(HydraTestSequence(
                seq_path=child, 
                mode='test', 
                delta_t_ms=self.delta_t_ms, 
                num_bins=self.num_bins, 
                representation=self.representation
                ))
        return test_sequences
    
    def get_evimo_train_dataset(self, sequence_len:int, batch_size:int):
        assert self.train_path.is_dir(), str(self.train_path)
        train_sequences = list()
        for child in sorted(self.train_path.iterdir()):
            for grandchild in sorted(child.iterdir()):
                if not str(grandchild).endswith('.h5'):
                    continue
                print(f"Loading sequence {grandchild}")
                train_sequences.append(EVIMOSequence(
                    h5_path=grandchild, 
                    window_ms=self.delta_t_ms, 
                    num_bins=self.num_bins, 
                    sequence_len=sequence_len,
                    batch_size=batch_size
                    ))
        return torch.utils.data.ConcatDataset(train_sequences)
    
    def get_evimo_test_dataset(self):
        assert self.test_path.is_dir(), str(self.test_path)
        test_sequences = list()
        for child in sorted(self.test_path.iterdir()):
            for grandchild in sorted(child.iterdir()):
                if not str(grandchild).endswith('.h5'):
                    continue
                print(f"Loading sequence {grandchild}")
                test_sequences.append(EVIMOTestSequence(
                    h5_path=grandchild, 
                    window_ms=self.delta_t_ms, 
                    num_bins=self.num_bins, 
                    ))
        return test_sequences
    
    def get_ev_imo_train_dataset(self, sequence_len:int, dt:int = 0, mask_future_events:bool = False):
        assert self.train_path.is_dir(), str(self.train_path)
        train_sequences = list()
        for child in sorted(self.train_path.iterdir()):
            for grandchild in sorted(child.iterdir()):
                print(f"Loading sequence {grandchild}")
                train_sequences.append(EV_IMOSequence(
                    h5_path=grandchild, 
                    window_ms=self.delta_t_ms, 
                    num_bins=self.num_bins, 
                    sequence_len=sequence_len,
                    dt=dt,
                    mask_future_events=mask_future_events
                    ))
        return torch.utils.data.ConcatDataset(train_sequences)
    
    def get_ev_imo_test_dataset(self, sequence_len:int, dt:int = 0, mask_future_events:bool = False):
        assert self.test_path.is_dir(), str(self.test_path)
        test_sequences = list()
        for child in sorted(self.test_path.iterdir()):
            for grandchild in sorted(child.iterdir()):
                if not str(grandchild).endswith('.h5'):
                    continue
                print(f"Loading sequence {grandchild}")
                test_sequences.append(EV_IMOSequence(
                    h5_path=grandchild, 
                    window_ms=self.delta_t_ms, 
                    num_bins=self.num_bins, 
                    sequence_len=sequence_len,
                    dt=dt,
                    mask_future_events=mask_future_events
                    ))
        return test_sequences
    
    def get_rampvo_evimo_dataset(self):
        assert self.test_path.is_dir(), str(self.test_path)
        test_sequences = list()
        for child in sorted(self.test_path.iterdir()):
            for grandchild in sorted(child.iterdir()):
                if not str(grandchild).endswith('.h5'):
                    continue
                print(f"Loading sequence {grandchild}")
                test_sequences.append(EVIMOFramesSequence(
                    h5_path=grandchild, 
                    window_ms=self.delta_t_ms, 
                    num_bins=self.num_bins, 
                    ))
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
    
    # Load evimo dataset by event number and if only_sequence is set, 
    # load only the sequence that corresponds to the string e.g. only_sequence="box" loads only sequences with "box" in their path
    def get_evimo_test_dataset_by_event(self, window_ev_num, only_sequence: str =""):
        assert self.test_path.is_dir(), str(self.test_path)
        test_sequences = list()
        for child in sorted(self.test_path.iterdir()):
            if only_sequence != "" and only_sequence not in str(child):
                continue
            for grandchild in sorted(child.iterdir()):
                if not str(grandchild).endswith('.h5'):
                    continue
                print(f"Loading sequence {grandchild}")
                test_sequences.append(EVIMOTestSequenceImportByNumber(
                    h5_path=grandchild, 
                    window_ms=self.delta_t_ms, 
                    num_bins=self.num_bins,
                    window_ev_num=window_ev_num 
                    ))
        return test_sequences
    
    def get_high_freq_hydra_train_dataset(
        self, 
        sequence_len:int = 2, 
        max_num_grad_events:int = 10000, 
        dt:int = [0,100], 
        augment:list = [], 
        augment_prob:list = [],
        multiple_batches:bool = True
        ):
        assert self.train_path.is_dir(), str(self.train_path)
        train_sequences = list()
        for child in sorted(self.train_path.iterdir()):
            train_sequences.append(HighFreqHydraSequence(
                seq_path=child, 
                mode='train', 
                delta_t_ms=self.delta_t_ms, 
                num_bins=self.num_bins, 
                sequence_len=sequence_len, 
                representation=self.representation, 
                max_num_grad_events=max_num_grad_events, 
                dt=dt,
                augment=augment,
                augment_prob=augment_prob,
                multiple_batches=multiple_batches
                ))
        return torch.utils.data.ConcatDataset(train_sequences)

    def get_evimo_train_dataset_by_varying_event_window(self, sequence_len:int, batch_size:int):
        assert self.train_path.is_dir(), str(self.train_path)
        train_sequences = list()
        for child in sorted(self.train_path.iterdir()):
            for grandchild in sorted(child.iterdir()):
                if not str(grandchild).endswith('.h5'):
                    continue
                print(f"Loading sequence {grandchild}")
                train_sequences.append(EVIMOSequenceRandByNumber(
                    h5_path=grandchild, 
                    window_ms=self.delta_t_ms, 
                    num_bins=self.num_bins, 
                    sequence_len=sequence_len,
                    batch_size=batch_size
                    ))
        return torch.utils.data.ConcatDataset(train_sequences)

    def get_evimo_test_dataset_by_varying_event_window(self, dt_pred_time_ms: float):
        assert self.test_path.is_dir(), str(self.test_path)
        test_sequences = list()
        for child in sorted(self.test_path.iterdir()):
            for grandchild in sorted(child.iterdir()):
                if not str(grandchild).endswith('.h5'):
                    continue
                print(f"Loading sequence {grandchild}")
                test_sequences.append(EVIMOTestSequenceByNumber(
                    h5_path=grandchild, 
                    window_ms=self.delta_t_ms, 
                    num_bins=self.num_bins,
                    dt_pred_time_ms=dt_pred_time_ms
                    ))
        return test_sequences


if __name__ == "__main__":
    # dsec_dir = "/home/rpg/Downloads/DSEC"
    dsec_dir = "/data/scratch/pellerito/datasets/DSEC"
    num_bins = 15
    dataset_provider = DatasetProvider(dsec_dir, num_bins=num_bins, representation="stack")
    test_dataset = dataset_provider.get_high_freq_hydra_train_dataset()
    for i, seq in enumerate(test_dataset):
        print(f"Sequence {i}: {seq}")

    # evimo_dir = "/home/rpg/Downloads/EVIMO1"
    # evimo_dir = "/data/scratch/pellerito/datasets/EVIMO1"
    # num_bins = 2
    # dataset_provider = DatasetProvider(evimo_dir, num_bins=num_bins, representation="voxel")
    # # evimo_dataset = dataset_provider.get_evimo_train_dataset(sequence_len=5)
    # start = time.time()
    # evimo_dataset = dataset_provider.get_rampvo_evimo_dataset()
    # print(f"Number of sequences: {len(evimo_dataset)}, time {time.time()-start:.2f}s")