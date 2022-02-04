import numpy as np

def reverse(array, channels, subdivision=1):
    '''
    Reverses subdivisions of an array of audio data
    array: a numpy array of audio data, numbers not empty
    channels: mono (1) or stereo (2) file
    subdivision: int, amount of subarrays to create default: 1
    every: which subdivions to reverse default: 1
    returns: a reversed version of array by subdivision
    '''
    # check if array.shape divisible by subdivision
    # if not error
    if len(array) % subdivision != 0:
        print('Error: array size not divisible by subdivision.')
    
    # subdivide array
    else:
        # reverse every nth subarray
        rev_array = np.row_stack(np.flip(np.split(array, subdivision), axis=1))

        # mono case for removing extra dimension from np.split on mono arrays
        if channels == '1':
            rev_array = rev_array.reshape(array.size)
        
        # return combined array
        return rev_array