import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class TextResizer():
    """
    Resize plot text by a factor of the scale of a window resize event.

    Whenever the window is resized, the text in the plots is resized
    proportionally.
    Stores the initial figure height and fontsizes, updating the
    fontsizes once the figure is resized, scaled by the new figure
    height divided by the initial height.
    """

    def __init__(self, texts, fig=None, minimal=4):
        """
        texts: list of text objects in figure being resized by this
        class
        fig: matplotlib figure object, the figure being resized
        minimal: minimal fontsize resize threshold
        """

        # Sanity check & for minimal testing examples
        if not fig: 
            fig = plt.gcf()

        # Class attributes
        self.texts = texts
        self.fig = fig
        self.minimal = minimal

        # Create list of fontsizes for all text objects in texts list
        self.fontsizes = [t.get_fontsize() for t in self.texts]

        # Store initial figure windowheight (the window width is unused)
        self.windowwidth, self.windowheight = fig.get_size_inches() * fig.dpi

    def __call__(self, event=None):
        """
        Callback function for figure resize events. 

        Factors the fontsizes of text objects in the texts list by the
        scale of the current figure height from the initial figure
        height.
        """

        # Scale of current figure height by initial figure height
        # Halving height lets size enlarge again
        scale = event.height / (self.windowheight/2)

        # Resizing fontsizes for text objects in texts list
        for i in range(len(self.texts)):
            """
            Factors each fontsize in the texts list by the scale bottom
            bounded by minimal and sets the fontsize of the text object
            in the text list to this new scaled fontsize
            """
            newsize = np.max([int(self.fontsizes[i] * scale), self.minimal])
            self.texts[i].set_fontsize(newsize)