from abc import ABC, abstractmethod

from pymodelextractor.utils.data_loader import DataLoader
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import SymbolStr

import joblib

class PickleDataLoader(DataLoader):    
    def __init__(self, data_path):
        self.data_path = data_path        
        self._load()    

    def _load(self):
        #Load files        
        self._data = joblib.load(self.data_path)

    def get_data(self):
        return self._data
