"""
Audio processing & visualization library
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor, Button
import matplotlib.gridspec as gridspec

from soundscope.util.split import split


# use backend that supports animation, blitting & figure window resizing
mpl.use('Qt5Agg')

# ignore divide by 0 error in log
np.seterr(divide = 'ignore')


def spectrogram(array, name, channels, sample_rate, fig=None, sub=False,
                gridspec=None, resize_ls=None):
    """
    Creates a spectrogram given an array of audio data

    array: 1 or 2d numpy array of audio data
    channels: 1 mono or 2 stereo, number of channels in audio array
    name: name of the audio file
    fig: external figure to plot onto if provided, default = None
    sub: boolean, True: plotting as subplot of larger figure,
        False: otherwise, default False
    gridspec: gridspec to plot onto if part of a larger figure otherwise
        None, default None
    resize_ls: list of text objects to be resized on window resize
        events when plotting inside visualizer, default None

    returns a spectrogram with y: frequency decibel scale logarithmic,
        x: time (seconds)
    """
    # Font
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Helvetica'

    # Mono case
    if channels == '1':
        # dark background white text, initilize figure and axes
        plt.style.use('dark_background')

        # figure and axes init in case of subplot or singular
        if fig is None:
            fig, ax = plt.subplots()

        else:
            ax = fig.add_subplot(222)

        # labeling axes & title
        title = '%s SPECTROGRAM' % name
        if sub:
            title = 'SPECTROGRAM'
        xlabel = ax.set_xlabel('TIME (S)', color='#F9A438', fontsize=7)
        ylabel = ax.set_ylabel('FREQUENCY (KHZ)', color='#F9A438',
                               fontsize=7)
        title_mono = ax.set_title(title, color='#F9A438', fontsize=10)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', color='#F9A438', labelsize=6,
                       labelcolor='#F9A438')

        # spine coloring
        spine_ls = ['top', 'bottom', 'left', 'right']
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_color('#F9A438')

        # plot spectrogram (only im is used for colorbar)
        spec, fq, t, im = ax.specgram(array, Fs= sample_rate, cmap='magma',
                                      vmin=-120, vmax=0)

        # make space for colorbar
        if not sub:
            fig.subplots_adjust(right=0.84)

        # colorbar
        if not sub:
            # left, bottom, width, height
            cbar_ax = fig.add_axes([0.85, 0.1125, 0.01, 0.768])
        else:
            cbar_ax = fig.add_axes([0.905, 0.53, 0.003, 0.35])
        fig.colorbar(im, ticks=np.arange(-120, 0 + 5, 5),
                     cax=cbar_ax).set_label('AMPLITUDE (dB)', color='#F9A438',
                                            fontsize=7)
        cbar_ax.tick_params(color='#F9A438', labelsize=5, labelcolor='#F9A438')

        # get colorbar label for resizing
        cbarlabel = cbar_ax.get_yaxis().get_label()

        # limit y axis to human hearing range
        ax.set_ylim([0, 20000])

        # fq in kHz
        ticks = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1000))
        ax.yaxis.set_major_formatter(ticks)

        # state variable dictionary of starting axis limits
        state = {'start_xlim': ax.get_xlim(), 'start_ylim': ax.get_ylim()}

        # zoom reset view button & axes
        if sub:
            # store initial figure dimesions
            fig_width, fig_height = fig.get_size_inches() * fig.dpi

            # reset button axis size based on figure size to look
            # correct on multiple screens
            if fig_height <= 1700:
                reset_button_ax = fig.add_axes([0.878, 0.49, 0.022, 0.015])

            else:
                # left, bottom, width, height
                reset_button_ax = fig.add_axes([0.886, 0.49, 0.0145, 0.01])

            # reset button
            reset_button = Button(reset_button_ax, 'RESET', color='black',
                                  hovercolor='#7E0000')

            # small screen, smaller label
            if fig_height <= 1700:
                reset_button.label.set_size(6)

            # big screen, big label
            else:
                reset_button.label.set_size(7)

            reset_button.label.set_color('#F0191C')
            for spine in spine_ls:
                reset_button_ax.spines[spine].set_color('#F0191C')

            # callback function for zoom reset button
            def reset_button_on_clicked(mouse_event):
                ax.set_xlim(state['start_xlim'])
                ax.set_ylim(state['start_ylim'])
            reset_button.on_clicked(reset_button_on_clicked)

        if resize_ls is not None:
            # store text to be resized
            resize_ls.extend([title_mono, xlabel, ylabel, cbarlabel,
                              reset_button.label])

        # individual figure or as part of larger figure
        if sub:
            return fig, reset_button, reset_button_on_clicked, resize_ls
        else:
            return plt.show()

    # Stereo subplots fasceted
    elif channels == '2':
        # divide array into stereo components
        left, right = split(array, channels)

        # dark background white text, initilize figure and axes
        plt.style.use('dark_background')

        # figure and axes init in case of subplot or singular
        if fig is None:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                           sharey=True)

        else:
            ax1 = fig.add_subplot(gridspec[0, 1])
            ax2 = fig.add_subplot(gridspec[1, 1], sharex=ax1, sharey=ax1)

        # labeling axes & title
        title = '%s SPECTROGRAM' % name
        if sub:
            title = 'SPECTROGRAM'
        xlabel = ax2.set_xlabel('TIME (S)', color='#F9A438', fontsize=7)
        ylabel_L = ax1.set_ylabel('LEFT FREQUENCY (KHZ)', color='#F9A438',
                                  fontsize=7)
        ylabel_R = ax2.set_ylabel('RIGHT FREQUENCY (KHZ)', color='#F9A438',
                                  fontsize=7)
        title_stereo = ax1.set_title(title, color='#F9A438', fontsize=10)
        ax1.minorticks_on()
        ax2.minorticks_on()
        ax1.tick_params(axis='both', which='both', color='#F9A438',
                        labelsize=6, labelcolor='#F9A438')
        ax2.tick_params(axis='both', which='both', color='#F9A438',
                        labelsize=6, labelcolor='#F9A438')

        # x axis on top
        ax1.xaxis.tick_top()

        # spine coloring
        spine_ls = ['top', 'bottom', 'left', 'right']
        for ax, spine in zip([ax1, ax2], spine_ls):
            plt.setp(ax.spines.values(), color='#F9A438')

        # plot spectrograms
        specl, fql, tl, iml = ax1.specgram(left, Fs=sample_rate, cmap='magma',
                                           vmin=-120, vmax=0)
        specr, fqr, tr, imr = ax2.specgram(right, Fs=sample_rate, cmap='magma',
                                           vmin=-120, vmax=0)

        # make space for colorbar & stack plots snug
        if not sub:
            fig.subplots_adjust(right=0.84, hspace=0)

        # colorbar
        if not sub:
            # left, bottom, width, height
            cbar_ax = fig.add_axes([0.845, 0.11, 0.007, 0.77])
        else:
            # left, bottom, width, height
            cbar_ax = fig.add_axes([0.905, 0.414, 0.003, 0.466])
        colorbar = fig.colorbar(iml, ticks=np.arange(-120, 0 + 5, 5),
                                cax=cbar_ax).set_label('AMPLITUDE (dB)',
                                                       color='#F9A438',
                                                       fontsize='x-small')
        cbar_ax.tick_params(color='#F9A438', labelsize=6, labelcolor='#F9A438')
        # get colorbar label for resizing
        cbarlabel = cbar_ax.get_yaxis().get_label()

        # limit y axes to human hearing range
        ax1.set_ylim([0, 20000])
        ax2.set_ylim([0, 20000])

        # fq in kHz
        ticks = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1000))
        ax1.yaxis.set_major_formatter(ticks)
        ax2.yaxis.set_major_formatter(ticks)

        # multicursor
        multi = MultiCursor(fig.canvas, (ax1, ax2), horizOn=True,
                            color='blueviolet', lw=0.5)

        # state variable dictionary for starting axis limits
        state = {'start_xlim1': ax1.get_xlim(), 'start_ylim1': ax1.get_ylim(),
                 'start_xlim2': ax2.get_xlim(), 'start_ylim2': ax2.get_ylim()}

        # zoom reset view button
        if sub:
            # store initial figure dimesions
            fig_width, fig_height = fig.get_size_inches() * fig.dpi

            # reset button axis size based on figure size to look
            # correct on multiple screens
            if fig_height <= 1700:
                reset_button_ax = fig.add_axes([0.878, 0.373, 0.022, 0.015])

            else:
                # axes left, bottom, width, height
                reset_button_ax = fig.add_axes([0.886, 0.373, 0.0145, 0.01])

            # reset button
            reset_button = Button(reset_button_ax, 'RESET', color='black',
                                  hovercolor='#7E0000')

            # small screen, smaller label
            if fig_height <= 1700:
                reset_button.label.set_size(6)

            # big screen, big label
            else:
                reset_button.label.set_size(7)

            reset_button.label.set_color('#F0191C')
            for spine in spine_ls:
                reset_button_ax.spines[spine].set_color('#F0191C')

            # callback function for zoom reset button
            def reset_button_on_clicked(mouse_event):
                ax1.set_xlim(state['start_xlim1'])
                ax2.set_xlim(state['start_xlim2'])
                ax1.set_ylim(state['start_ylim1'])
                ax2.set_ylim(state['start_ylim2'])
            reset_button.on_clicked(reset_button_on_clicked)

        if resize_ls is not None:
            # store text to be resized
            resize_ls.extend([title_stereo, xlabel, ylabel_L, ylabel_R,
                              cbarlabel, reset_button.label])

        # individual figure or as part of larger figure
        if sub:
            return fig, reset_button, reset_button_on_clicked, resize_ls
        else:
            return plt.show()