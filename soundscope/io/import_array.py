"""
Audio processing & visualization library

Handles the importing of audio files to numpy arrays via the soundfile dependancy
"""

import soundfile as sf


def import_array(file):
    """
    Import audio file as 64 bit float array
    file: audio file
    returns: a filename, number of channels, data (a 64 bit float numpy array of audio data), 
            the files subtype and sample rate of the file.
    """
    # extracting the filename and subtype from soundfile's info object
    info = str(sf.info(file))

    # includes file path
    name_fp = info[:info.find('\n')]
    name = name_fp[name_fp.rfind('/')+1:]
    channels = info[info.find('channels:') + 10:info.find('channels:') + 11]
    subtype = info[info.find('subtype:') + 9:]
    subtype = subtype[subtype.find('['):]

    # reading the audio file as a soundfile numpy array
    data, sample_rate = sf.read(file)

    return name, channels, data, subtype, sample_rate