"""
 CS-410: SigMF class that stores sigmf metadata info
 @file SigMF.py
 @authors Jun Cho, Will Cho, Grace Johnson, Connor Whynott
 @collaborators None
"""

import json
import Annotation

class SigMF():
    
    def __init__(self, sigmf_file):
        """
            Description: Initializes SigMF object
            :param sigmf_file: the sigmf metadata file to be parsed
        """
        sigmf = sigmf_file.read()
        data = json.loads(sigmf)

        self.datatype = data['global'].get('core:datatype')
        self.sample_rate = data['global'].get('core:sample_rate', 0)
        self.author = data['global'].get('core:author')
        self.hardware = data['global'].get('core:hw')
        self.offset = data['global'].get('core:offset')
        self.recorder = data['global'].get('core:recorder')

        self.datetime = data['captures'][0].get('core:datetime')
        self.center_frequency = data['captures'][0].get('core:frequency', 0)
        self.sample_start = data['captures'][0].get('core:sample_start')

        self.annotations = []
        #at this point, original code expects annotations of sigmf-meta to be empty; should handle when not empty
        for annotation in data['annotations']:
            sample_start = annotation.get('sample_start', 0)
            sample_count = annotation.get('sample_count', 0)
            freq_lower = annotation.get('freq_lower_edge', 0)
            freq_upper = annotation.get('freq_upper_edge', 0)
            label = annotation.get('label')
            comment = annotation.get('comment')
            self.annotations.append(Annotation(sample_start, sample_count, freq_lower, freq_upper, label, comment))

        # Calculate statistics
        self.duration = self._calculate_duration()
        self.min_frequency = self._calculate_min_frequency()
        self.max_frequency = self._calculate_max_frequency()
        self.frequency_range = self.max_frequency - self.min_frequency if self.max_frequency and self.min_frequency else 0
        self.signal_power = self._calculate_signal_power()

    def _calculate_duration(self):
        """ Compute duration based on available metadata """
        if self.sample_start is not None and self.sample_rate:
            return self.sample_start / self.sample_rate
        return 0

    def _calculate_min_frequency(self):
        """ Compute min frequency from annotations """
        if self.annotations:
            return min(ann.freq_lower for ann in self.annotations if ann.freq_lower is not None)
        return self.center_frequency - (self.sample_rate / 2) if self.center_frequency and self.sample_rate else 0

    def _calculate_max_frequency(self):
        """ Compute max frequency from annotations """
        if self.annotations:
            return max(ann.freq_upper for ann in self.annotations if ann.freq_upper is not None)
        return self.center_frequency + (self.sample_rate / 2) if self.center_frequency and self.sample_rate else 0

    def _calculate_signal_power(self):
        """ Compute an estimated signal power (dummy implementation, replace if needed) """
        return 10 * (self.sample_rate / 1e6)  # Example placeholder logic
