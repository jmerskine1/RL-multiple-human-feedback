# -*- coding: utf-8 -*-
"""
 RL monitor class
 Monitor RL results 

 Note:  
"""
import numpy as np

"""
Episodic RL monitor
"""
class RLmon(object):
    
    def __init__(self, names=[]):
        self.data = {}
        self.temp = {}
        for name in names:
            self.data[name] = []
            self.temp[name] = []
        return
    
    def store(self, name=None, d=None, done=False):
        if name not in self.data.keys() and name is not None:
            self.data[name] = []
            self.temp[name] = []
        if d is not None:
            self.temp[name].append(d)
        # if doen, store the data
        if done:
            if name is None:
                for n in self.data.keys():
                    self.data[n].append(self.temp[n])
                    self.temp[n] = []
            else:
                self.data[name].append(self.temp[name])
                self.temp[name] = []
        return
    
    def get_avestd(self, name):
        """Get the average and standard deviation of the data
        Args:
            name (str): The name of the data to get the average of.
        """
        if name not in self.data:
            return None
        if len(self.data[name]) == 0:
            return None
        
        ave = []
        std = []
        n = 0
        while True:
            data = [d[n] for d in self.data[name] if len(d) > n]
            if len(data) == 0:
                break
            ave.append(np.mean(data, axis=0))
            std.append(np.std(data, axis=0))
            n += 1
        return np.array(ave), np.array(std)
    
    def get_data(self, name):
        """Get the percentile of the data
        Args:
            name (str): The name of the data to get the percentile of.
            pct (float): The percentile to get.
        """
        if name not in self.data:
            return None
        if len(self.data[name]) == 0:
            return None
        
        data = []
        n = 0
        while True:
            tmp_data = [d[n] for d in self.data[name] if len(d) > n]
            data.append(tmp_data)
            if len(tmp_data) == 0:
                break
            
            n += 1
        return data
    
    def get_pct(self, name, pct=0.5):
        """Get the percentile of the data
        Args:
            name (str): The name of the data to get the percentile of.
            pct (float): The percentile to get.
        """
        if name not in self.data:
            return None
        if len(self.data[name]) == 0:
            return None
        
        pct_data = []
        n = 0
        while True:
            data = [d[n] for d in self.data[name] if len(d) > n]
            if len(data) == 0:
                break
            pct_data.append(np.percentile(data, pct, axis=0))
            n += 1
        return np.array(pct_data)

    def loadData(self, fname):
        data = np.load(fname, allow_pickle=True)
        for name in data.keys():
            self.data[name] = data[name].tolist()
        return

    def saveData(self, fname):
        # Convert all keys to strings
        string_data = {str(key): value for key, value in self.data.items()}
        np.savez(fname, **string_data)
        return
