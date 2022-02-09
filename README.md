# soundscope

Audio imager & editor Python package
====================================

soundscope is a Python package for imaging & editing audio files. The
package simplifies routine audio editing commonly performed in gui based
software by bringing them to the command line.

Automate routine audio editing and imaging processes that are costly to
perform in a digital audio workstation.

The subpackages exported by soundscope include:

dsp: modules for midside encoding & peak normalization.
io: handles import and export of audio files to numpy arrays and back.
util: editing modules that invert, reverse, split & trim audio arrays.
vis: waveform, spectral & spatial audio array imaging modules.
tests: unit test suite for dsp, io & util subpackages.

soundscope exports the following modules:

import_array: imports audio files as numpy arrays.
export_array: exports numpy arrays as audio files.
midside: converts stereo files to midside encoding and vice-versa.
normalize: peak normalizes audio arrays.
invert: reverses the polarity (inverts the phase) of audio arrays.
reverse: reverses audio arrays by subdivisions.
split: splits stereo audio arrays into 2 mono arrays.
trim: trims leading and trailing 0's from audio arrays.
bins: downsamples audio arrays by averaging bins for imaging resolution.
magnitude: variable scale magnitude/frequency plot of audio array.
spectrogram: spectrogram (frequency/time/amplitude) plot of audio array.
vectorscope: stereo image intensity/position plot of audio array.
visualizer: combined plot of all other imaging modules.
waveform: plot audio array intensity over time.
text_resizer: class that resizes plot text on figure window resizing.

soundscope also includes a test runner for the unit tests & a suite of
tests requiring the user to verify the visual output of the imaging
modules.

Dependancies include numpy, matplotlib, soundfile & inquirer. soundfile
handles the import and export of audio files to and from numpy arrays.
numpy handles the editing of the arrays of audio data. inquirer provides
a ui for the tests requiring user interaction. matplotlib supplies the
interface for the visualization functions.

License MIT 2022 Kin Maynard