import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from soundscope.vis.text_resizer import TextResizer
from soundscope.vis.waveform import waveform
from soundscope.vis.magnitude import magnitude
from soundscope.vis.spectrogram import spectrogram
from soundscope.vis.vectorscope import vectorscope


# Use backend that supports animation, blitting & figure window resizing
mpl.use('Qt5Agg')

# Ignore divide by 0 error in log
np.seterr(divide='ignore')


def visualizer(array, name, channels, sample_rate):
    """
    Plot waveform, magnitude, spectrogram & vectorscope of array.

    array: numpy array of audio data
    name: file name
    channels: mono (1) or stereo (2) file
    sample_rate: sampling rate of audio file

    returns: fasceted subplots of waveform, magnitude, spectrogram &
    vectorscope
    """

    # Initialize figure with dark background and title
    plt.style.use('dark_background')
    fig = plt.figure()

    # Maximize figure window to screen size
    figmanager = plt.get_current_fig_manager()
    figmanager.window.showMaximized()

    # Font
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Helvetica'

    # Title
    title = plt.suptitle('%s VISUALIZATION' % name, color='#F9A438',
                         fontsize=17.5, fontweight=900)

    # Store text objects for later resizing when window resized
    resize_ls = [title]

    # Gridspec to snugly fascet only stereo
    # Spectrogram and waveform plots
    # Initialize for mono case
    gs1, gs2 = None, None
    if channels == '2':
        # Snugly fascet stereo subplots
        fig.subplots_adjust(hspace=0)

        # Outer gridspec, hspace separates waveform & spectrogram plots
        # from magnitude & vectorscope
        outer = gridspec.GridSpec(nrows=2, ncols=1, figure=fig, hspace=0.2,
                                  height_ratios=[2, 1])

        # Nested gridspecs
        gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0])
        gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1])

        # Stereo mag plot with side button
        fig, lrsums, side, lindB, scale, reset_mag, reset_mag_click, \
        resize_ls = magnitude(array, name, channels, sample_rate, fig=fig,
                              sub=True, gridspec=gs2, resize_ls=resize_ls)

        # Enabling mag buttons
        lrsums.on_clicked(side)

    else:
        # Mono mag plot without side button
        fig, lindB, scale, reset_mag, reset_mag_click, resize_ls = magnitude(
            array, name, channels, sample_rate, fig=fig, sub=True,
            gridspec=gs2, resize_ls=resize_ls)

    # Subplots currently multi_spec only shows
    fig, reset_wav, reset_wav_click, resize_ls = waveform(
        array, name, channels, sample_rate, fig=fig, sub=True, gridspec=gs1,
        resize_ls=resize_ls)
    fig, reset_spec, reset_spec_click, resize_ls = spectrogram(
        array, name, channels, sample_rate, fig=fig, sub=True, gridspec=gs1,
        resize_ls=resize_ls)
    fig, resize_ls, polarlissa, choose_plot = vectorscope(
        array, name, channels, sample_rate, fig=fig, sub=True, gridspec=gs2,
        resize_ls=resize_ls)

    # Enabling mag buttons
    lindB.on_clicked(scale)

    # Enabling vectorscope buttons
    polarlissa.on_clicked(choose_plot)

    # Enabling view reset buttons
    reset_wav.on_clicked(reset_wav_click)
    reset_spec.on_clicked(reset_spec_click)
    reset_mag.on_clicked(reset_mag_click)

    # Connect the figure resize events 
    # to the font resizing callback function
    cid = plt.gcf().canvas.mpl_connect("resize_event", TextResizer(resize_ls))

    plt.show()