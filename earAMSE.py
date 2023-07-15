# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 22:08:23 2022

@author: tanyang
"""

import tensorflow
import tensorflow as tf
import librosa
from tensorflow.keras import backend as K
import numpy as np
import kapre
from kapre.time_frequency import (
    STFT,
    MagnitudeToDecibel, 
    Magnitude
)
from tensorflow.keras import Sequential
import pandas as pd
import xlrd
import h5py
import matplotlib.pyplot as plt

_CH_FIRST_STR = 'channels_first'
_CH_LAST_STR = 'channels_last'
_CH_DEFAULT_STR = 'default'


def filterbank_mel(
    sample_rate, n_freq, n_mels=128, f_min=0.0, f_max=None, htk=False, norm='slaney',trainable=False, num_classes=3
):
    """A wrapper for librosa.filters.mel that additionally does transpose and tensor conversion
    librosa.filters.mel 

    Args:
        sample_rate (`int`): sample rate of the input audio
        n_freq (`int`): number of frequency bins in the input STFT magnitude.
        n_mels (`int`): the number of mel bands 
        f_min (`float`): lowest frequency that is going to be included in the mel filterbank (Hertz)
        f_max (`float`): highest frequency that is going to be included in the mel filterbank (Hertz)
        htk (bool): whether to use `htk` formula or not Hether
        norm: The default, 'slaney', would normalize the the mel weights by the width of the mel band.

    Returns:
        (`Tensor`): mel filterbanks. Shape=`(n_freq, n_mels)`
        (1999,513)pow_frames*fbank.T
    """
    
    filterbank = librosa.filters.mel(
        sr=sample_rate,
        n_fft=(n_freq - 1) * 2,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        htk=htk,
        norm=norm,
    ).astype(K.floatx())
    
    filterbank = filterbank.T
    print('FF shape', filterbank.shape)

    if trainable:
        filterbank_variables = []
        num_l = filterbank.shape[1]
        num_r = filterbank.shape[0]
        filterbank = np.array(filterbank)
        for rr in range(num_r):
            v_temp = []
            for ll in range(num_l):
                if filterbank[rr][ll] == 0:
                    v_t = tf.Variable(filterbank[rr][ll], trainable=False, name=str(rr)+'.'+str(ll))
                else:
                    v_t = tf.Variable(filterbank[rr][ll], trainable=True, name=str(rr)+'.'+str(ll))
                v_temp.append(v_t)
            filterbank_variables.append(v_temp)
        filterbank = filterbank_variables

    print(filterbank)
    print('Trainable mel spectrogram is '+ str(trainable))
    return filterbank


class DCTtrans(tensorflow.keras.layers.Layer):
    """Compute the magnitude of the complex input, resulting in a float tensor

    Args:
        n_filter: integer, the number of filters in the DCT bank. Defaults to 40.
        norm: string, the normalization used for the DCT. Can be "ortho" or None (default).
        
    Example:
        ::
            
            input_shape = (2048, 1)  # mono signal
            model = Sequential()
            model.add(kapre.STFT(n_fft=1024, hop_length=512, input_shape=input_shape))
            mode.add(DCTtrans())
            # now the shape is (batch, n_frame=3, n_freq=513, ch=1) and dtype is float  
    """
    
    def call(self, x):
        """
        Args:
            x (complex `Tensor`): input complex tensor

        Returns:
            (float `Tensor`): magnitude of `x`
        """
        # return tf.signal.dct(inputs, type=self.type, axis=self.axis)
        return tf.signal.dct(x)


class ApplyFilterbank(tensorflow.keras.layers.Layer):
    """
    Apply a filterbank to the input spectrograms.
    Args:
        filterbank (`Tensor`): filterbank tensor in a shape of (n_freq, n_filterbanks)
        data_format (`str`): specifies the data format of batch input/output
        **kwargs: Keyword args for the parent keras layer (e.g., `name`)
    Example:
        ::
            input_shape = (2048, 1)  # mono signal
            n_fft = 1024
            n_hop = n_fft // 2
            kwargs = {
                'sample_rate': 22050,
                'n_freq': n_fft // 2 + 1,
                'n_mels': 128,
                'f_min': 0.0,
                'f_max': 8000,
            }
            model = Sequential()
            model.add(kapre.STFT(n_fft=n_fft, hop_length=n_hop, input_shape=input_shape))
            model.add(Magnitude())
            # (batch, n_frame=3, n_freq=n_fft // 2 + 1, ch=1) and dtype is float
            model.add(ApplyFilterbank(type='mel', filterbank_kwargs=kwargs))
            # (batch, n_frame=3, n_mels=128, ch=1)
    """

    def __init__(
        self, type, filterbank_kwargs, data_format='default', **kwargs,
    ):

        kapre.backend.validate_data_format_str(data_format)

        self.type = type
        self.filterbank_kwargs = filterbank_kwargs

        self.filterbank = _mel_filterbank = filterbank_mel(**filterbank_kwargs)

        if data_format == _CH_DEFAULT_STR:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

        if self.data_format == _CH_FIRST_STR:
            self.freq_axis = 3
        else:
            self.freq_axis = 2
        super(ApplyFilterbank, self).__init__(**kwargs)

    def call(self, x):
        """
        Apply filterbank to `x`.
        Args:
            x (`Tensor`): float tensor in 2D batch shape.
        """

        # x: 2d batch input. (b, t, fr, ch) or (b, ch, t, fr)
        output = tf.tensordot(x, self.filterbank, axes=(self.freq_axis, 0))
        # ch_last -> (b, t, ch, new_fr). ch_first -> (b, ch, t, new_fr)
        if self.data_format == _CH_LAST_STR:
            output = tf.transpose(output, (0, 1, 3, 2))
        return output

    def get_config(self):
        config = super(ApplyFilterbank, self).get_config()
        config.update(
            {
                'type': self.type,
                'filterbank_kwargs': self.filterbank_kwargs,
                'data_format': self.data_format,
            }
        )
        return config


def get_melspectrogram_layer(
    input_shape=None,
    n_fft=2048,
    win_length=None,
    hop_length=None,
    window_name=None,
    pad_begin=False,
    pad_end=False,
    sample_rate=22050,
    n_mels=128,
    mel_f_min=0.0,
    mel_f_max=None,
    mel_htk=False,
    mel_norm='slaney',
    return_decibel=False,
    db_amin=1e-5,
    db_ref_value=1.0,
    db_dynamic_range=80.0,
    input_data_format='default',
    output_data_format='default',
    trainable = True,
    name='melspectrogram',
    num_classes=3,
):
    """A function that returns a melspectrogram layer, which is a `keras.Sequential` model consists of
    `STFT`, `Magnitude`, `ApplyFilterbank(_mel_filterbank)`, and optionally `MagnitudeToDecibel`.
    
    Args:
        input_shape (None or tuple of integers): input shape of the model. Necessary only if this melspectrogram layer is
            is the first layer of your model (see `keras.model.Sequential()` for more details)
        n_fft (int): number of FFT points in `STFT`
        win_length (int): window length of `STFT`
        hop_length (int): hop length of `STFT`
        window_name (str or None): *Name* of `tf.signal` function that returns a 1D tensor window that is used in analysis.
            Defaults to `hann_window` which uses `tf.signal.hann_window`.
            Window availability depends on Tensorflow version. More details are at `kapre.backend.get_window()`.
        pad_begin (bool): Whether to pad with zeros along time axis (length: win_length - hop_length). Defaults to `False`.
        pad_end (bool): whether to pad the input signal at the end in `STFT`.
        sample_rate (int): sample rate of the input audio
        n_mels (int): number of mel bins in the mel filterbank
        mel_f_min (float): lowest frequency of the mel filterbank
        mel_f_max (float): highest frequency of the mel filterbank
        mel_htk (bool): whether to follow the htk mel filterbank fomula or not
        mel_norm ('slaney' or int): normalization policy of the mel filterbank triangles
        return_decibel (bool): whether to apply decibel scaling at the end
        db_amin (float): noise floor of decibel scaling input. See `MagnitudeToDecibel` for more details.
        db_ref_value (float): reference value of decibel scaling. See `MagnitudeToDecibel` for more details.
        db_dynamic_range (float): dynamic range of the decibel scaling result.
        input_data_format (str): the audio data format of input waveform batch.
        input_data_format(str)：
            `'channels_last'` if it's `(batch, time, channels)`
            `'channels_first'` if it's `(batch, channels, time)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())
        output_data_format (str): the data format of output melspectrogram.
            `'channels_last'` if you want `(batch, time, frequency, channels)`
            `'channels_first'` if you want `(batch, channels, time, frequency)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())
        name (str): name of the returned layer
    Note:
        Melspectrogram is originally developed for speech applications and has been *very* widely used for audio signal
        analysis including music information retrieval. As its mel-axis is a non-linear compression of (linear)
        frequency axis, a melspectrogram can be an efficient choice as an input of a machine learning model.
        We recommend to set `return_decibel=True`.
        **References**:
        `Automatic tagging using deep convolutional neural networks <https://arxiv.org/abs/1606.00298>`_,
        `Deep content-based music recommendation <http://papers.nips.cc/paper/5004-deep-content-based-music-recommen>`_,
        `CNN Architectures for Large-Scale Audio Classification <https://arxiv.org/abs/1609.09430>`_,
        `Multi-label vs. combined single-label sound event detection with deep neural networks <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.711.74&rep=rep1&type=pdf>`_,
        `Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification <https://arxiv.org/pdf/1608.04363.pdf>`_,
        and way too many speech applications.
    Example:
        ::
            input_shape = (2, 2048)  # stereo signal, audio is channels_first
            melgram = get_melspectrogram_layer(input_shape=input_shape, n_fft=1024, return_decibel=True,
                n_mels=96, input_data_format='channels_first', output_data_format='channels_last')
            model = Sequential()
            model.add(melgram)
            # now the shape is (batch, n_frame=3, n_mels=96, n_ch=2) because output_data_format is 'channels_last'
            # and the dtype is float
    """
    kapre.backend.validate_data_format_str(input_data_format)
    kapre.backend.validate_data_format_str(output_data_format)

    stft_kwargs = {}
    if input_shape is not None:
        stft_kwargs['input_shape'] = input_shape

    waveform_to_stft = STFT( # 短时傅里叶变换
        **stft_kwargs,
        n_fft = n_fft,
        win_length = win_length,
        hop_length = hop_length,
        window_name = window_name,
        pad_begin = pad_begin,
        pad_end = pad_end,
        input_data_format = input_data_format,
        output_data_format = output_data_format,
    )

    stft_to_stftm = Magnitude() # 转为为功率谱

    kwargs = {
        'sample_rate': sample_rate,
        'n_freq': n_fft // 2 + 1,
        'n_mels': n_mels,
        'f_min': mel_f_min,
        'f_max': mel_f_max,
        'htk': mel_htk,
        'trainable': trainable,
        'norm': mel_norm,
        'num_classes':num_classes,
    }
    stftm_to_melgram = ApplyFilterbank(
        type='mel', filterbank_kwargs=kwargs, data_format=output_data_format
    )

    mag_to_decibel = MagnitudeToDecibel(
            ref_value = db_ref_value, amin=db_amin, dynamic_range=db_dynamic_range
        )
    
    melspectrogram_to_mfcc = DCTtrans()
     
    if name == 'mel':
        layers = [waveform_to_stft, stft_to_stftm, stftm_to_melgram]
    elif name == 'log':
        layers = [waveform_to_stft, stft_to_stftm, stftm_to_melgram, mag_to_decibel]
    elif name == 'mfcc':
        layers = [waveform_to_stft, stft_to_stftm, stftm_to_melgram, mag_to_decibel, melspectrogram_to_mfcc]
    
    return Sequential(layers, name=name)
    

def show_AMs(model_file, MFBs_file, mel_type, n_mels, n_fft):
    """
    Extracts Mel filter coefficients from a model file and stores them in an Excel file.

    Args:
        model_file (str): Path to the model file.
        MFBs_file (str): Path to the Excel file to store the Mel filter coefficients.
        mel_type (str): Type of Mel filter.
        n_mels (int): Number of Mel filters.
        n_fft (int): FFT size.

    Returns:
        None

    Comments:
        - The model file should be in HDF5 format.
        - Requires the 'h5py', 'numpy', and 'pandas' libraries to be imported.
        - Requires the 'openpyxl' library to be installed for saving the Excel file.
        
    Example usage:
        ::
            model_file = 'model.h5'
            MFBs_file = 'mel_filters.xlsx'
            mel_type = 'mel_layer'
            n_mels = 40
            n_fft = 2048
            
            # Call the show_AMs function to extract and save the Mel filter coefficients
            show_AMs(model_file, MFBs_file, mel_type, n_mels, n_fft)
    """
    
    # Open the model file
    model_data = h5py.File(model_file, 'r')

    mel_filters = []

    # Extract Mel filter coefficients
    for rr in range(int(n_fft / 2) + 1):
        for ll in range(n_mels):
            filter_key = f'model_weights/{mel_type}/{rr}.{ll}:0'
            filter_coeffs = model_data[filter_key][()]
            mel_filters.append(filter_coeffs)

    mel_filters = np.array(mel_filters)
    mel_filters_reshaped = mel_filters.reshape((int(n_fft / 2) + 1, n_mels))

    # Create a DataFrame object to store the reshaped Mel filter coefficients
    Logmelspec = pd.DataFrame(mel_filters_reshaped)

    # Save the DataFrame to the Excel file
    Logmelspec.to_excel(MFBs_file)
    
    
def AMFBs_plot(AMFBs_path, n_mels, n_fft, sample_rate, AMFBs_fig_path):
    """
    Plot the Audio Mel Filter Bank coefficients.

    Args:
        AMFBs_path (str): Path to the input file of Audio Mel Filter Bank coefficients.
        n_mels (int): Number of Mel filters.
        n_fft (int): Window size of FFT (Fast Fourier Transform).
        sample_rate: Sample rate of the input audio.
        AMFBs_fig_path (str): Path to save the output figure of Audio Mel Filter Bank coefficients.

    Returns:
        None
        
    Example usage:
        ::
            # Input parameters
            AMFBs_path = "path/to/AMFBs_file.xls"
            n_mels = 20
            n_fft = 2048
            AMFBs_fig_path = "path/to/output_figure.eps"
            
            # Plot the Audio Mel Filter Bank coefficients
            AMFBs_plot(AMFBs_path, n_mels, n_fft, AMFBs_fig_path)
    """
    
    # Import required libraries

    
    # Open the workbook
    exl = xlrd.open_workbook(AMFBs_path)
    
    # Initialize lists to store x and y values
    x_value = []
    y_value = []
    
    # Read data from sheet 1
    sheet1 = exl.sheet_by_index(0)
    
    # Get number of rows and columns
    n_rows = sheet1.nrows
    n_cols = sheet1.ncols
    
    # Calculate x values
    for i in range((int(n_fft / 2) + 1)):
        x_value.append(i * (sample_rate / 2) / (int(n_fft / 2) + 1))
    
    # Extract y values
    for i in range(0, n_mels + 1):
        y_value.append(sheet1.col_values(i)[1:int(n_fft / 2) + 2])
    
    # Set major locators for x and y axes
    x_interval = sample_rate / 4
    x_major_locator = plt.MultipleLocator(4000)
    y_major_locator = plt.MultipleLocator(0.01)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    
    # Set x and y limits
    plt.xlim(0, sample_rate / 2)
    plt.ylim(-0.010, 0.020)
    
    # Plot multiple curves
    for i in range(1, n_mels + 1):
        plt.plot(x_value, y_value[i], linestyle='-')
    
    # Set tick parameters and labels
    plt.tick_params(labelsize=48)
    plt.xlabel('Frequency (Hz)', fontsize = 50, fontdict = {'family': 'Times New Roman'})
    plt.ylabel('Amplitude', fontsize = 50, fontdict = {'family': 'Times New Roman'})
    x1_label = ax.get_xticklabels()
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels()
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
    
    # Adjust spacing and save the figure
    plt.tight_layout()
    plt.savefig(AMFBs_fig_path, format='eps', dpi=1000)

