from abc import ABC, abstractmethod

from pymodelextractor.utils.data_loader import DataLoader
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import SymbolStr

import pandas as pd
from abc import ABC, abstractmethod

class CsvDataLoader(DataLoader):    
    def __init__(self, data_path):
        self.data_path = data_path
        self.alphabet = set()
        self._load()

    
    def word_to_sequence(self, word):
        seq_list = []
        for elem in word:
            symbol = SymbolStr(str(elem))
            self.alphabet.add(symbol)
            seq_list.append(symbol)
        return Sequence(seq_list)

    @abstractmethod
    def value_to_output(self, value):  
        raise NotImplementedError      

    def _load(self):
        #Load files
        df = pd.read_csv(self.data_path, header = None)
        df.columns = ['seq','value']  
        sequences = list(df['seq'])
        values = list(df['values'])
        sequences = [self.word_to_sequence(query) for query in sequences]
        values =  [self.value_to_output(value) for value in values]
        self._data = dict(zip(sequences, values))


    def get_data(self):
        return self._data
