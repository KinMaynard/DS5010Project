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


def waveform(array, name, channels, sample_rate, fig=None, sub=False,
    gridspec=None, resize_ls=None):
    """
    array: array of audio data
    name: file name
    channels: mono (1) or stereo (2) file
    sample_rate: sampling rate of audio file
    fig: external figure to plot onto if provided, default = None
    sub: boolean, True: plotting as subplot of larger figure,
        False: otherwise, default False
    gridspec: gridspec to plot onto if part of a larger figure
        otherwise None, default None
    resize_ls: list of text objects to be resized on window resize
        events when plotting inside visualizer, default None

    returns: waveform plot of intensity/time either alone
        or as part of provided fig
    """
    # Font
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    mpl.rcParams['agg.path.chunksize'] = 20000

    # mono
    if channels == '1':
        # dark background white text, initilize figure and axes
        plt.style.use('dark_background')

        # initializing figure and axes
        if fig is None:
            fig, ax = plt.subplots()

        # if plotting on external figure only adding subplot
        else:
            ax = fig.add_subplot(221)

        # labeling axes & title
        title = '%s WAVEFORM' % name
        if sub:
            title = 'WAVEFORM'
        title_mono = ax.set_title(title, color='#F9A438', fontsize=10)
        xlabel_mono = ax.set_xlabel('TIME (S)', color='#F9A438', fontsize=7)
        ylabel_mono = ax.set_ylabel('AMPLITUDE', color='#F9A438', fontsize=7)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', color='#F9A438', labelsize=6,
                       labelcolor='#F9A438')

        # spine coloring
        spine_ls = ['top', 'bottom', 'left', 'right']
        for spine in spine_ls:
            ax.spines[spine].set_color('#F9A438')

        # adding gridline on 0 above data
        ax.axhline(0, color='#F9A438', linewidth=0.5, zorder=3)

        # plot signal amplitude/time
        # seconds in file
        time = array.size / sample_rate
        # line = np.stack((np.linspace(0.0, time, array.size), array), axis=-1)
        # col = mpl.collections.LineCollection([line], color='#16F9DA')
        # ax.add_collection(col, autolim=True)
        ax.plot(np.linspace(0.0, time, array.size), array, color='#16F9DA')

        ax.margins(0.001)

        # state variable dictionary of starting axis limits
        state = {'start_xlim': ax.get_xlim(), 'start_ylim': ax.get_ylim()}

        # zoom reset view button & axes
        if sub:
            # store initial figure dimesions
            fig_width, fig_height = fig.get_size_inches() * fig.dpi

            # reset button axis size based on figure size
            # to look correct on multiple screens
            if fig_height <= 1700:
                # left, bottom, width, height
                reset_button_ax = fig.add_axes([0.455, 0.49, 0.022, 0.015])

            else:
                # left, bottom, width, height
                reset_button_ax = fig.add_axes([0.463, 0.49, 0.0145, 0.01])

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
            resize_ls.extend([title_mono, xlabel_mono, ylabel_mono,
                             reset_button.label])

        # individual figure or as part of larger figure
        if sub:
            return fig, reset_button, reset_button_on_clicked, resize_ls
        else:
            return plt.show()

    # stereo
    elif channels == '2':
        # divide array into stereo components
        left, right = split(array, channels)

        # dark background white text, initilize figure and axes
        plt.style.use('dark_background')

        # initializing figure and axes
        if fig is None:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                           sharey=True)

        # if plotting on external figure only adding subplots
        else:
            ax1 = fig.add_subplot(gridspec[0, 0])
            ax2 = fig.add_subplot(gridspec[1, 0], sharex=ax1, sharey=ax1)

        # labeling axes & title
        title = '%s WAVEFORM' % name
        if sub:
            title = 'WAVEFORM'
        title_stereo = ax1.set_title(title, color='#F9A438', fontsize=10)
        xlabel = ax2.set_xlabel('TIME (S)', color='#F9A438', fontsize=7)
        ylabel_L = ax1.set_ylabel('AMPLITUDE LEFT', color='#F9A438',
                                  fontsize=7)
        ylabel_R = ax2.set_ylabel('AMPLITUDE RIGHT', color='#F9A438',
                                  fontsize=7)
        ax1.minorticks_on()
        ax2.minorticks_on()
        ax1.tick_params(axis='both', which='both', color='#F9A438',
                        labelsize=6, labelcolor='#F9A438')
        ax2.tick_params(axis='both', which='both', color='#F9A438',
                        labelsize=6, labelcolor='#F9A438')

        # adding gridline on 0 above data
        ax1.axhline(0, color='#F9A438', linewidth=0.5, zorder=3)
        ax2.axhline(0, color='#F9A438', linewidth=0.5, zorder=3)

        # spine coloring
        spine_ls = ['top', 'bottom', 'left', 'right']
        for ax, spine in zip([ax1, ax2], spine_ls):
            plt.setp(ax.spines.values(), color='#F9A438')

        # snuggly fasceting subplots if plotting to external figure
        if not sub:
            fig.subplots_adjust(hspace=0)

        # x axis on top
        ax1.xaxis.tick_top()

        # plot signal amplitude/time
        # only left size otherwise will be double the amount of time
        time = left.size / sample_rate
        # line1 = np.stack((np.linspace(0.0, time, left.size), left), axis=-1)
        # line2 = np.stack((np.linspace(0.0, time, left.size), right), axis=-1)
        # col1 = mpl.collections.LineCollection([line1], color='#16F9DA')
        # col2 = mpl.collections.LineCollection([line2], color='#16F9DA')
        # ax1.add_collection(col1, autolim=True)
        # ax2.add_collection(col2, autolim=True)
        ax1.plot(np.linspace(0.0, time, left.size), left, color='#16F9DA')
        ax2.plot(np.linspace(0.0, time, right.size), right, color='#16F9DA')

        ax1.margins(0.001)
        ax2.margins(0.001)

        # Multicursor
        multi = MultiCursor(fig.canvas, (ax1, ax2), horizOn=True,
                            color='blueviolet', lw=0.5)

        # state variable dictionary for starting axis limits
        state = {'start_xlim1': ax1.get_xlim(), 'start_ylim1': ax1.get_ylim(),
                 'start_xlim2': ax2.get_xlim(), 'start_ylim2': ax2.get_ylim()}

        # zoom reset view button
        if sub: 
            # store initial figure dimesions
            fig_width, fig_height = fig.get_size_inches() * fig.dpi

            # reset button axis size based on figure size
            # to look correct on multiple screens
            if fig_height <= 1700:
                reset_button_ax = fig.add_axes([0.455, 0.373, 0.022, 0.015])

            else:
                # axes left, bottom, width, height
                reset_button_ax = fig.add_axes([0.463, 0.373, 0.0145, 0.01])

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
                              reset_button.label])

        # individual figure or as part of larger figure
        if sub:
            return fig, reset_button, reset_button_on_clicked, resize_ls
        else:
            return plt.show()