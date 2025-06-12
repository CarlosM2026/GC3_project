"""
 CS-410: Annotation class that holds information for a single spectrogram information
 @file Annotation.py
 @authors Jun Cho, Will Cho, Grace Johnson, Connor Whynott
 @collaborators None
"""

class Annotation():

    def __init__(self, start_count, sample_count, freq_lower_edge, freq_upper_edge, label, comment):
        """
            Description: Initializes annotation object
            :param start_count: sample at which the annotation begins
            :param sample_count: number of samples over which the annotation spans
            :param freq_lower_edge: annotation lower frequency boundary
            :param freq_upper_edge: annotation upper frequency boundary
            :param label: label to be displayed with annotation
            :param comment: additional comment or info about annotation
        """
        self.start_count = start_count
        self.sample_count = sample_count
        self.freq_lower_edge = freq_lower_edge
        self.freq_upper_edge = freq_upper_edge
        self.label = label
        self.comment = comment