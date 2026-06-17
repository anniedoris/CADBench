import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datasets import Dataset

class Processor:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
    
    def __call__(self, idx: int):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        return self.__call__(idx)
    