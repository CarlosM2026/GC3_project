# M. Zheleva, P. Bogdanov, T. Larock and P. Schmitt, "AirVIEW: Unsupervised transmitter detection for next generation spectrum sensing," IEEE INFOCOM 2018 - IEEE Conference on Computer Communications, Honolulu, HI, USA, 2018, pp. 1673-1681, doi: 10.1109/INFOCOM.2018.8485946.

import numpy as np
import json
from pydantic.dataclasses import dataclass
import math
import copy

def findTransmitters(input, scale, beta, jaccard_threshold, max_gap_rows, fft_size):
    '''
    Gets parameters, calculates threshold, and runs algorithm.

    return:
    list[Transmitter]: detected transmitters
    '''
    # params[0] = mean of the pairwise difference of multiscale products
    # params[1] = std of the pairwise differences of multiscale products
    # beta is a threshold-scaling parameter that determines how many standard deviations from the mean should pairwise differences be in order to be ranked as a outlier local maxima
    params, regions = findAvgAdjDiffCoarse(input, math.log2(input.shape[1]) - scale,
				math.log2(input.shape[1]) - (scale + 1))
    # num_rows = len(regions)
    # num_columns = len(regions[0])  # Assuming all inner lists have the same length
    # num_items = len(regions[0][0])
    # print("Dimensions:", (num_rows, num_columns, num_items))
    # print(regions[0])
    threshold = params[0] + params[1]*beta # threshold = mean + stdev*beta
    
    detected = findTransmittersMultiScale(input, regions, jaccard_threshold, scale, threshold, max_gap_rows)
    return detected # output annotations in main run function
    

@dataclass
class Plugin:
    sample_rate: int = 0
    center_freq: int = 0
    
    # custom params
    run_parameter_optimization: str = 'no'
    beta: float = 2.0
    scale: int = 9

    def run(self, samples):
        print(samples[0:10])
        print(self.sample_rate)
        print(self.center_freq)
        print(self.beta)
        print(self.scale)

        # Your Plugin (and optionally, classification) code here

        # turn samples into 2d matrix from 1d array
        fft_size = 1024
        num_rows = int(np.floor(len(samples)/fft_size))
        spectrogram = np.zeros((num_rows, fft_size))
        for i in range(num_rows):
            spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i*fft_size:(i+1)*fft_size])))**2)

        print(spectrogram.shape)
        time_for_fft = fft_size * (1/self.sample_rate) *1000 # time it takes to traverse in ms
        max_gap_rows = math.ceil(0.0/time_for_fft)
        jaccard_threshold = 0.5 # if they are at least halfway overlapping, considered aligned

        if self.run_parameter_optimization[0].lower() == 'y':
            print('finding optimal paramters')
            airview_beta_scale = findOptimalParams(spectrogram)
            print(f"Optimal Beta and Scale: beta: {airview_beta_scale[0]}, scale: {airview_beta_scale[1]}")
        else:
            detected = findTransmitters(spectrogram, self.scale, self.beta, jaccard_threshold, max_gap_rows, fft_size)
            print("FOUND")
            # When making a detector, for the return, make a list, then for each detected emission, add one of these dicts to the list:
            annotations = []
            for transmitter in detected:
                # print('*** detected: ', transmitter)

                start_row = transmitter.start_row
                start_col = transmitter.start_col
                end_row = transmitter.end_row
                end_col = transmitter.end_col

                # print('*** what java would output:')
                print(f"{transmitter.start_col},{num_rows - transmitter.end_row},{transmitter.end_col - transmitter.start_col},{transmitter.end_row-transmitter.start_row}\n")

                x = start_col
                y = start_row
                width = end_col - start_col
                height = end_row - start_row
                
                if height > 0:
                    an = {}
                    an['core:freq_lower_edge'] = int(x / fft_size * self.sample_rate - (self.sample_rate / 2) + self.center_freq) # Hz
                    an['core:freq_upper_edge'] = int((x + width) / fft_size * self.sample_rate - (self.sample_rate / 2) + self.center_freq) # Hz
                    an['core:sample_start'] = int(y * fft_size)
                    an['core:sample_count'] = int(height * fft_size)
                    an["core:label"] = "Transmitter"# NOTE should we should set this to transmitters, since that is what AirView looks for?
                    annotations.append(an)
        
        if self.run_parameter_optimization[0].lower() == 'y':
            return {
                "data_output" : [],
                "airview_beta_scale" : airview_beta_scale
            }
        else:
            return {
                "data_output" : [],
                "airview_annotations" : annotations
            }
    

# MULTISCALE FUNCTIONS

def adjacentOrOverlapping(edges, start, end):
    '''
    Check if the indices are directly adjacent to or overlapping with an
	already established region.
    '''
    for e in edges:
        if (e[1][0] + 1 == start or e[0][0] -1 == end or
        (e[0][0] <= start and e[1][0] >= end) or
        (e[1][0] >= start and e[1][0] <= end) or
        (e[0][0] >= start and e[0][0] <= end) or
        (e[0][0] >= start and e[1][0] <= end)):
            return True
    return False

def isSorted(regions, indexToSort):
    for i in range(1, len(regions)):
        if regions[i-1][indexToSort] + regions[i-1][indexToSort+1] < regions[i][indexToSort] + regions[i][indexToSort+1]:
            return False
    return True

def sortRegions(regions, indexToSort):
    sorted_regions = regions[:]
    
    while not isSorted(sorted_regions, indexToSort):
        for i in range(1, len(sorted_regions)):
            if sorted_regions[i-1][indexToSort] + sorted_regions[i-1][indexToSort+1] < sorted_regions[i][indexToSort] + sorted_regions[i][indexToSort+1]:
                tmp = sorted_regions[i-1]
                sorted_regions.pop(i-1)
                sorted_regions.insert(i-1, sorted_regions[i-1])
                sorted_regions.pop(i)
                sorted_regions.insert(i, tmp)
    
    return sorted_regions


def stats(input, start, end):
    '''
    Compute mean power stats between start and end indices.
    '''
    mean = 0.0
    sd = 0.0
    for i in range(start, end):
        mean += input[i]
    if end - start > 0:
        mean /= end - start

    for i in range(start, end):
        sd += (mean - input[i])**2
    if end - start > 0:
        sd = math.sqrt(sd/(end - start))
    else:
        sd = math.sqrt(sd)
    
    return [mean, sd]

def threshold(regions, input, alpha):
    edges = [] # edge = [col, state]
    # print(len(regions))
    for i in range(1, len(regions)):
        # compare the value of the constant power in each region (array index 4)--> multiscale value
		# if the absolute value of their difference is larger than the threshold (alpha)
        if abs(regions[i-1][4] - regions[i][4]) > alpha:
            # add the higher value to the edges list
            # looking for everything that's above threshold
            if regions[i-1][4] > regions[i][4]:
                edges.append(regions[i-1])
            else:
                edges.append(regions[i])
    
    # now we have the filtered coarse representation. Need to get averages for each adjacent region
    filtered_list = []
    for i in range(1, len(edges)):
        # start with the widest band (leftmost to right most)
        mean_sd = stats(input, edges[i-1][0], edges[i][1])
        
        # set max to these stats
        max = mean_sd
        d = [edges[i-1][0], edges[i][1], max[0], max[1]]

        # compare stats for other intervals
        # (leftmost to leftmost)
        mean_sd = stats(input, edges[i-1][0], edges[i][0])

        # if mean power is greater, this is a new max
        if mean_sd[0] > max[0] and mean_sd[0] < 0:
            max = mean_sd
            d = [edges[i-1][0], edges[i][0], max[0], max[1]]

        # repeat same process for each interval
        # (rightmost to rightmost)
        mean_sd = stats(input, edges[i-1][1], edges[i][1])
        if mean_sd[0] > max[0] and mean_sd[0] < 0:
            max = mean_sd
            d = [edges[i-1][1], edges[i][1], max[0], max[1]]
        
        # (rightmost to leftmost)
        mean_sd = stats(input, edges[i-1][1], edges[i][0])
        if mean_sd[0] > max[0] and mean_sd[0] < 0:
            max = mean_sd
            d = [edges[i-1][1], edges[i][0], max[0], max[1]]

        # add the max to the list
        filtered_list.append(d)

    return filtered_list


def coarseDetection(input, regions, alpha):
    '''
    Detects transmissions using the coarse representation of the multiscale
	transform. Collapses the signal, filters region permutations based on
	threshold, then sorts based on highest mean power in the input signal.
    '''
    edges = [] # edge = [col, state]
    
    # threshold the coarse signal
    filtered = threshold(regions, input, alpha)

    # sort the regions (index 2 is by mean, index 0/1 for frequency bins,
    # 3 for standard deviation, 4 is coarse value)
    filtered = sortRegions(filtered, 2)

    # call every other region a transmitter
    for f in filtered:
        if(not adjacentOrOverlapping(edges, f[0], f[1])):
            edges.append([[f[0], 'r'], [f[1], 'f']])

    return edges

def findTransmittersMultiScale(input, regions, jaccard_threshold, scale, alpha, max_gap_rows):
    # list of Transmitters
    transmitters = []

    # integer : list[edge[]]
    changes = {}

    for r, row in enumerate(input):
        curr_edges = None
        curr_edges = coarseDetection(row, regions[r], alpha)

        # add all of the changes to the map
        changes[r] = curr_edges
        updateTransmitters(changes, transmitters, r, jaccard_threshold, max_gap_rows)

    return transmitters


def multiscale_transform(input, scale1, scale2):
    '''
    Multiscale transform. Takes an input vector and two scales. Transforms
	the signal and builds a coefficient tree. Gets the relevant indices for
	each input level and reconstructs the signal twice, once using each set
	of indices. Finally, the element-wise product is taken between the two
	reconstructions and this is the multiscale transform.
    '''
    t = transform1D(input, 0)

    relevant_values1 = get_values(t, scale1)
    relevant_values2 = get_values(t, scale2)
    r1 = reconstruct1D(relevant_values1)
    r2 = reconstruct1D(relevant_values2)

    if len(r1) != len(r2):
        raise ValueError("Error: Cannot multiply vectors of unequal length.")
    
    # multiply element wise
    return np.multiply(r1, r2)


def getRegionMeans(regions, input):
    '''
    Get mean in input for each region (bins with same values)
    '''
    # for each region
    for region in regions:
        # get start and end column
        start = region[0]
        # print(region)
        end = region[1]
        # calculate mean
        mean = 0.0
        for j in range(start, end+1):
            mean += input[j]

        if end - start != 0:
            mean /= end - start

        # calculate sd
        sd = 0.0
        for k in range (start, end+1):
            sd += (mean-input[k])**2
        if end - start != 0:
            sd = math.sqrt(sd/(end-start))
        else:
            sd = math.sqrt(sd)
        
        # store
        region[2] = mean
        region[3] = sd
    
    return regions

def multiscale_detection_getDefaultRegions(input, scale1, scale2):
    '''
    Get the "default" regions, meaning only the regions of constant power
	without any filtering or thresholding.
    '''
    '''
    Multiscale transform. Takes an input vector and two scales. Transforms
	the signal and builds a coefficient tree. Gets the relevant indices for
	each input level and reconstructs the signal twice, once using each set
	of indices. Finally, the element-wise product is taken between the two
	reconstructions and this is the multiscale transform.
    '''
    transformed = multiscale_transform(input, scale1, scale2)

    # compute the regions
    regions = []
    previous_value = transformed[0]
    start = 0
    end = 0
    # for i, t in enumerate(transformed, start=1):
    for i in range(1, len(transformed)):
        # if the value is remaining constant, increase the end index
        if transformed[i] == previous_value and i != len(transformed)-1:
            end += 1
        else:
            # if the value changed or we reached the end of the signal
            if i == len(transformed) - 1:
                end += 1 # increase end to the final entry
            
            # add the region
            regions.append([start, end, 0.0, 0.0, previous_value])
            end += 1
            start = end
            previous_value = transformed[i]
    
    # calculate the mean/variance of each region
    regions = getRegionMeans(regions, input)
    return regions


# FIND PARAMETER FUNCTIONS

def findSumabsSumsqN_row(input, scale1, scale2):
    '''
    Calculates the sum of absolute values, sum of squares, and the size of one region.
    '''
    sumabs_sumsq_n = [0.0, 0.0, 0.0]
    # get regions with constant power - multiscale transform
    # regions is list of lists
    regions = multiscale_detection_getDefaultRegions(input, scale1, scale2)

    for i in range(1, len(regions)):
        sumabs_sumsq_n[0] += abs(regions[i - 1][4] - regions[i][4])
        sumabs_sumsq_n[1] += (regions[i - 1][4] - regions[i][4]) ** 2
        sumabs_sumsq_n[2] += 1
    
    return sumabs_sumsq_n, regions

def findSumabsSumsqN(input, scale1, scale2):
    '''
    Calculates the sum of absolute values, sum of squares, and size
    '''
    # sumabs_sumsq_n[0] = sum of absolute values
    # sumabs_sumsq_n[1] = sum of squares
    # sumabs_sumsq_n = size
    sumabs_sumsq_n = [0.0, 0.0, 0.0]
    regions = []
    for i, row in enumerate(input): # iterate through rows
        loc_sumabs_sumsq_n, regions_row = findSumabsSumsqN_row(row, scale1, scale2) # compute the sum of absolute values, sum of squares, and size for each row
        regions.append(regions_row)
        # for j, s in enumerate(sumabs_sumsq_n):
        for j in range(len(sumabs_sumsq_n)):
            sumabs_sumsq_n[j] += loc_sumabs_sumsq_n[j] # accumulate results
    return sumabs_sumsq_n, regions

def findAvgAdjDiffCoarse(input, scale1, scale2):
    '''
    Finds the average/standard deviation of adjacent differences in coarse
	signals. This method just uses the "default" regions (those defined
	by the resolution) to construct the coarse signal.
    '''
    mean_stdev = [None, None]
    sumabs_sumsq_n, regions = findSumabsSumsqN(input, scale1, scale2)
    mean_stdev[0] = sumabs_sumsq_n[0]/sumabs_sumsq_n[2]
    mean_stdev[1] = math.sqrt(sumabs_sumsq_n[1]/sumabs_sumsq_n[2] - (mean_stdev[0]**2))
    return mean_stdev, regions


# WAVELET DECOMPOSITION FUNCTIONS

def transform1D(input_row, level):
    '''
    Performs row wise 1-dimensional Haar wavelet transform on a given array.
    '''
    t = copy.deepcopy(input_row)
    avg = 0.0
    diff = 0.0
    tIndex = 0 
    endpoint = 0
    if level > 0:
        endpoint = (int) (len(input_row) // (2**(level)))
    else:
        endpoint = len(input_row)
    
    for i in range(0, endpoint, 2):
        avg = (input_row[i] + input_row[i + 1]) / 2
        diff = avg - input_row[i + 1]
        t[tIndex] = avg
        t[(int)(tIndex + endpoint / 2)] = diff
        tIndex += 1
    if endpoint != 2:
        level += 1
        t = transform1D(t, level)
    return t
        

def get_values(t, scale):
    '''
    Find the values we need in the reconstruction process. Always keeps the first value
    (the whole-signal average), and then finds the detail coefficients for the given 
    scale.
    parameters:
        t:  the complete wavelet transform
        scale:  the scale to reconstruct at
    '''
    output = np.zeros(len(t))
    output[0] = t[0]
    start_index = (int) (2**(scale)) # start index for coefficients to use
    end_index = (int) (start_index + 2**(scale))  # end index for coefficients to use
    for i in range(start_index, end_index):
        output[i] = t[i]
    return output


def reconstruct1D(input):
    '''
    This function iteratively reconstructs a signal that has been transformed
	by the 1-D Haar wavelet.
    '''
    r = np.empty(len(input))
    value = 0.0
    sign = 0
    detailCo = 0.0

    for i in range(len(input)):
        value = input[0]
        
        for k in range(1, (int)((math.log(len(input)) / math.log(2))) + 1):
            sign = (int) ((-1)**(math.floor((i * 2**(k)) / len(input))))
            detailCo = input[(int)(2**(k-1)) 
                             + (int)(math.floor((i * 2**(k-1)) / len(input)))]
            value += sign * detailCo
        
        r[i] = value
    
    return r


# TRANSMITTER FUNCTIONS

def colIntersection(actual, detected):
    col_int = 0.0
    # if the columns don't intersect at all
    if actual.end_col < detected.start_col or detected.end_col < actual.start_col:
        return col_int
    elif actual.start_col >= detected.start_col and actual.end_col <= detected.end_col:
        # the actual transmitter is contained within the detected
        col_int = actual.end_col - actual.start_col
    elif detected.start_col >= actual.start_col and detected.end_col <= actual.end_col:
        # the detected transmitter is contained within the actual
        col_int = detected.end_col - detected.start_col
    elif actual.start_col >= detected.start_col and actual.end_col >= detected.end_col:
        # the actual transmitter starts after the detected, but isn't contained within it
        col_int = detected.end_col - actual.start_col
    elif detected.start_col >= actual.start_col and detected.end_col >= actual.end_col:
        # the detected transmitter starts after the detected, but isn't contained within it
        col_int = actual.end_col - detected.start_col
    elif actual.start_col <= detected.start_col and actual.end_col <= detected.end_col:
        # the actual transmitter ends before the detected, but also starts before it
        col_int = actual.end_col - detected.start_col
    elif detected.start_col <= actual.start_col and detected.end_col <= actual.end_col:
        # the detected transmitter ends before the actual transmitter, but also starts before it
        col_int = detected.end_col - actual.start_col
    
    return col_int

def jaccard_value(t, a):
    '''
    Compute Jaccard similarity between transmitters.
    '''
    jaccard = 0.0
    intersection = colIntersection(t, Transmitter(1, 1, a[0][0], a[1][0]))
    union = (t.end_col - t.start_col) + (a[1][0] - a[0][0])
    jaccard = intersection / (union - intersection)
    return jaccard

def updateTransmitters(changes, t, r, jaccard_threshold, max_gap):
    # if the transmitter list is empty, just add all edges as transmitters
    if not t and changes[r]:
        for e in changes[r]:
            t.append(Transmitter(r, r, e[0][0], e[1][0]))
            t[len(t)-1].active_switch()
    else:
        # otherwise, compare with previous rows
	    # what has changed since the last row?
        # loop through current edges
        for curr in changes[r]:
            # initialize found to false for this edge array
            found = False
            # loop through transmitters, starting with most recent (to catch active transmitters)
            # for i, tx in reversed(list(enumerate(t))):
            for i in range(len(t)-1, -1, -1):
                tx = t[i]
                # if the transmitter + edges match within jaccard threshold
                if jaccard_value(tx, curr) >= jaccard_threshold:
                    found = True # we've matched the edge[] with a transmitter
                    # if the transmitter is currently active
                    if tx.active:
                        tx.set_row_fall(r) # set the latest row fall to the current row
                        tx.found = True
                    else:
                        # if the transmitter has already been deemed inactive, restart it
                        if r - tx.end_row <= max_gap:
                            # if it's been inactive for less than the max gap just restart it
                            tx.active_switch()
                            tx.set_row_fall(r)
                            tx.found = True
                        else:
                            # if it's been inactive for too long, create a new transmitter
                            t.append(Transmitter(r, r, curr[0][0], curr[1][0]))
                            t[len(t)-1].found = True
                            t[len(t)-1].active_switch()
                    break # no need to look at any more transmitters

            # if the edges weren't found in any previous transmitters
            if not found:
                # make a new transmitter with current row as start and end
                t.append(Transmitter(r, r, curr[0][0], curr[1][0]))
                t[len(t)-1].active_switch() # list this transmitter as active
                t[len(t)-1].found = True # set it to found so we don't deactivate it immediately below

        # loop through the transmitters again
        for tx in t:
            # only look at currently active transmitters that were not found
            if tx.active and not tx.found: # transmitter is no longer active
                tx.active = False
            tx.found = False # reset found to false


class Transmitter:
    def __init__(self, start_row, end_row, start_col, end_col, mean=None, sd=None, found=False, active=False, priors=None):
        self.start_row = start_row
        self.end_row = end_row
        self.start_col = start_col
        self.end_col = end_col
        self.mean = mean
        self.sd = sd
        self.found = found
        self.active = active
        self.priors = priors

    def active_switch(self):
        self.active = not self.active

    def set_row_fall(self, end_row):
        self.end_row = end_row
    
    def __str__(self):
        return f"Transmitter(start_row={self.start_row}, start_col={self.start_col}, " \
               f"end_row={self.end_row}, end_col={self.end_col})"


def jaccard_edges(tx1, tx2):
    '''
    Compute Jaccard similarity between transmitters.
    '''
    a = [[tx1.start_col, 'r'], [tx1.end_col, 'f']]
    b = [[tx2.start_col, 'r'], [tx2.end_col, 'f']]

    jaccard = 0.0
    intersection = colIntersection(Transmitter(1, 1, a[0][0], a[1][0]),
                                   Transmitter(1, 1, b[0][0], b[1][0]))
    union = (a[1][0] - a[0][0]) + (b[1][0] - b[0][0])
    # print(a)
    # print(b)
    epsilon = 1e-12
    jaccard = intersection / (union - intersection + epsilon)
    return jaccard


def avgJS(d1, d2):
    if len(d1) > 0:
        js = 0.0
        # need to find the max jaccard for every detected transmitter
        jss = [0.0] * len(d1)

        # for each of the detected transmitters
        for i in range(len(d1)):
            # find the maximum intersection/transmitter ratio
            for tx in d2:
                js = jaccard_edges(d1[i], tx)
                if js > jss[i]:
                    jss[i] = js

        tjs = 0.0

        # total the max jaccaards for each txer
        for i in range(len(jss)):
            tjs += jss[i]
        
        # divide by the total number of detected transmitters
        tjs /= len(d1)
        return tjs
    else:
        return 0.0


def getTxsEdges(edges, row):
    txs = []
    for e in edges:
        txs.append(Transmitter(row, row, e[0][0], e[1][0]))
    return txs


def learnBeta(scale, input, bs, rows):
    # get scale 1
    s1 = math.log2(input.shape[1]) - scale
    # get scale 2
    s2 = math.log2(input.shape[1]) - (scale + 1)

    # create a 2d array to hold alignment data
    # [0] contains alignment, [1] contains relevant beta
    tpsms = [[0.0, 0.0] for _ in range(len(bs))]
    loc_params, regions = findAvgAdjDiffCoarse(input, s1, s2)

    for i in range(len(bs)):
        # collect beta from beta array
        b = bs[i]
        count = 0

        #row by row detection
        ptr1 = 0
        ptr2 = 0
        m = 0
        detected = []
        while True:
            detected_n = []
            while ptr1 < rows and m == 0:
                # current
                # collect transformed array using default regions and threshold
                curr_edges = coarseDetection(input[ptr1], regions[ptr1], loc_params[0] + (loc_params[1]*b))
                # collect transmissions that passed threshold
                detected = getTxsEdges(curr_edges, ptr1)
                if len(detected) > 0:
                    break
                ptr1 += 1
            
            ptr2 = ptr1 + 1
            while ptr2 < rows:
                # next
                # collect transformed array using default regions and threshold
                curr_edges = coarseDetection(input[ptr2], regions[ptr2], loc_params[0] + (loc_params[1]*b))
                # collect transmissions that passed threshold
                detected_n = getTxsEdges(curr_edges, ptr2)
                if len(detected_n) > 0:
                    break
                ptr2 += 1

            count += 1
            # calculate average jaccard of adjacent transmissions
            sm = (avgJS(detected, detected_n) + avgJS(detected_n, detected)) / 2.0
            tpsms[i][0] += sm

            ptr1 = ptr2
            detected = detected_n

            if ptr1 > rows:
                break
            
            m += 1
        
        # divide by the number of detected signals
        tpsms[i][0] /= count
        # reset counter
        count = 0

        tpsms[i][1] = b
    
    return tpsms


def findOptimalParams(spectogram):
    '''
    Finds optimal beta and scale to run airview with.
    '''
    minS=4 # min scale
    maxS=9 # max scale
    startB=2 # start beta
    endB=4 # end beta
    inc = 0.2
    input = copy.deepcopy(spectogram)

    # set array of scales
    scales = [0.0] * (maxS - minS + 1) # make sure each is not same reference or smthing
    for i in range(len(scales)):
        scales[i] = i + minS

    # set array of betas
    bs = [0.0] * math.ceil((endB - startB) / inc)
    bs[0] = startB * 1.0
    for i in range(len(bs)):
        bs[i] = bs[i-1] + inc

    # holds best alignment for comparison
    bestSim = float('-inf')
    # holds result of best alignment and scale
    res = [0.0, 0.0]
    for i in range(len(scales)):
        tpsm = learnBeta(scales[i], input, bs, input.shape[0])

        for j in range(len(tpsm)):
            if tpsm[j][0] > bestSim:
                bestSim = tpsm[j][0]
                res[0] = bs[j]
                res[1] = scales[i]
    res[0] = round(res[0], 1) # round to nearest 10th
    return res


if __name__ == "__main__":
    # Example of how to test your plugin locally
    #set fname to where you are storing your file pairs (data & meta)
    fname = "data/test"
    with open(fname + '.sigmf-meta', 'r') as f:
        meta_data = json.load(f)
    sample_rate = meta_data["global"]["core:sample_rate"]
    center_freq = meta_data["captures"][0]['core:frequency']
    samples = np.fromfile(fname + '.sigmf-data', dtype=np.complex64)
    params = {'sample_rate': sample_rate, 'center_freq': center_freq, 'run_parameter_optimization': 'n'}
    plugin = Plugin(**params)
    annotations = plugin.run(samples)
    print(annotations)
