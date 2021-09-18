from Preparation import *
import numpy as np

name_ls = ['../binaries/silence_44100_-infdBFS_Mono.aiff', '../binaries/white_88k_-3dBFS.wav', 
			'../binaries/hdchirp_88k_-3dBFS_lin.wav', '../binaries/sin_44100_100Hz_-3dBFS_1s.wav', 
			'../binaries/hdsweep_1Hz_44000Hz_-3dBFS_30s.wav', '../binaries/silence_44100_-infdBFS_Stereo.aiff', 
			'../binaries/whitenoise_44100_0dBFS_Stereo.aiff', '../binaries/hdchirp_88k_-3dBFS_lin_Stereo.aiff', 
			'../binaries/sin_44100_440Hz_-.8dBFS_Stereo.aiff', '../binaries/Saija Original Mix.aiff']

size_dict = {}

for name in name_ls:
	name, channels, data, subtype, sample_rate = import_array(name)
	size_dict[name] = len(data)

for entry in size_dict:
	print(entry, size_dict[entry])