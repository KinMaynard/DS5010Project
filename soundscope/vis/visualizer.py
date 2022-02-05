"""
Audio processing & visualization library
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from soundscope.vis.text_resizer import TextResizer
from soundscope.vis.waveform import waveform
from soundscope.vis.magnitude import magnitude
from soundscope.vis.spectrogram import spectrogram
from soundscope.vis.vectorscope import vectorscope


# use backend that supports animation, blitting & figure window resizing
mpl.use('Qt5Agg')

# ignore divide by 0 error in log
np.seterr(divide = 'ignore')


def visualizer(array, name, channels, sample_rate):
    """
    array: numpy array of audio data
    name: file name
    channels: mono (1) or stereo (2) file
    sample_rate: sampling rate of audio file
    
    returns: fasceted subplots of waveform, magnitude, 
        spectrogram & vectorscope
    """
    # initialize figure with dark background and title
    plt.style.use('dark_background')
    fig = plt.figure()

    # maximize figure window to screen size
    figmanager = plt.get_current_fig_manager()
    figmanager.window.showMaximized()

    # Font
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    
    # Title
    title = plt.suptitle('%s VISUALIZATION' % name, color='#F9A438', 
                         fontsize=17.5, fontweight=900)

    # store text objects for later resizing when window resized
    resize_ls = [title]

    # gridspec to snugly fascet only stereo 
    # spectrogram and waveform plots
    # initialize for mono case
    gs1, gs2 = None, None
    if channels == '2':
        # snugly fascet stereo subplots
        fig.subplots_adjust(hspace=0)

        # outer gridspec, hspace separates waveform & spectrogram plots 
        # from magnitude & vectorscope
        outer = gridspec.GridSpec(nrows=2, ncols=1, figure=fig, hspace = 0.2, 
                                  height_ratios = [2, 1])

        # nested gridspecs
        gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec = outer[0])
        gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec = outer[1])

        # stereo mag plot with side button
        fig, lrsums, side, lindB, scale, reset_mag, reset_mag_click, \
        resize_ls = magnitude(array, name, channels, sample_rate, fig=fig, 
                              sub=True, gridspec=gs2, resize_ls=resize_ls)
        
        # enabling mag buttons
        lrsums.on_clicked(side)

    else:
        # mono mag plot without side button
        fig, lindB, scale, reset_mag, reset_mag_click, resize_ls = magnitude(
            array, name, channels, sample_rate, fig=fig, sub=True, 
            gridspec=gs2, resize_ls=resize_ls)
    
    # subplots currently multi_spec only shows
    fig, reset_wav, reset_wav_click, resize_ls = waveform(
        array, name, channels, sample_rate, fig=fig, sub=True, gridspec=gs1, 
        resize_ls=resize_ls)
    fig, reset_spec, reset_spec_click, resize_ls = spectrogram(
        array, name, channels, sample_rate, fig=fig, sub=True, gridspec=gs1, 
        resize_ls=resize_ls)
    fig, resize_ls, polarlissa, chooseplot = vectorscope(
        array, name, channels, sample_rate, fig=fig, sub=True, gridspec=gs2, 
        resize_ls=resize_ls)

    # enabling mag buttons
    lindB.on_clicked(scale)

    # enabling vectorscope buttons
    polarlissa.on_clicked(chooseplot)

    # enabling view reset buttons
    reset_wav.on_clicked(reset_wav_click)
    reset_spec.on_clicked(reset_spec_click)
    reset_mag.on_clicked(reset_mag_click)

    # connect the figure resize events 
    # to the font resizing callback function
    cid = plt.gcf().canvas.mpl_connect("resize_event", TextResizer(resize_ls))

    plt.show()