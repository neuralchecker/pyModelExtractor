from abc import ABC, abstractmethod

from pymodelextractor.utils.data_loader import DataLoader
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import SymbolStr

from pymodelextractor.utils.csv_data_loader import CsvDataLoader

class CsvProbabilisticDataLoader(CsvDataLoader):    
    @abstractmethod
    def value_to_output(self, value):  
        return np.array(value)
