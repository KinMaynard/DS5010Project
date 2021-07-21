# DS5010Project

Topics to study:

audio statistics
matplotlib visualization

(Soundfile): 
	Import x audio format, make numpy array, export as any audio format

Helpers:
limiter function to avoid clipping (scale values to [-1.0, 1.0])

Main:
	Analysis:
	Spectrogram (Mel Scale, DB Scale not amplitude)
	Tempo
		transient detection
	Key/Note
		fundemental frequency/pitch detection
		Detect all pitches in sample

	Analytics to research:
		Spectral Centroid
		Spectral Rolloff
		Spectral Bandwidth
		Zero-Crossing Rate
		Mel-Frequency Cepstral Coefficients(MFCCs)
		Chroma feature
		Fourier transform

	Manipulation:
	Phase inversion
	Reverse
		Whole array
		slices of array in place
	Time Stetch