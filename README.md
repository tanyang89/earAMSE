# earAMSE
earAMSE is the first open-source toolkit for a trainable Mel spectrogram, whose full name is open-source trainable Mel filter banks transformed feature extractor.

# Installation

When you use it, download the brunch named "maseter" and use to install python third-party libraries via setup.py.

1. Download the installer named "master" brunch and unzip it.

2. Go to the "setup.py" directory and right-click here to open cmd.

3. Enter the command: python setup.py build.

4. Enter the command: Enter the command: python setup.py build.

The earAMSE library is now installed in your environment.

# Requirements

The minimal requirements to install earAMSE are listed below.

1. Python 3.8.0
2. Keras 2.10.0
3. kapre 0.3.4

# Usage

The adaptive Mel-related spectrogram layer can be used in the following three ways:

1. Sequential API
   
input_shape = (2, 2048) 

mel_layer = get_melspectrogram_layer(name='mel',n_fft=n_fft,sample_rate=sample_rate,n_mels=n_mels,
                                          win_length=win_length,hop_length=hop_length,
                                          return_decibel=return_decibel, input_data_format=input_data_format,
                                          trainable = trainable, num_classes=outputclasses)
                                          
model = Sequential()

model.add(mel_layer)

2. Functional API (We recommend to use the adaptive Mel layer with functional API.)
   
inputs = Input(shape=input_shape)

x = self.get_melspectrogram_layer(name='mel',n_fft=n_fft,sample_rate=sample_rate,n_mels=n_mels,
                                          win_length=win_length,hop_length=hop_length,
                                          return_decibel=return_decibel, input_data_format=input_data_format,
                                          trainable = trainable, num_classes=outputclasses) (inputs)

3. Subclassing API

After the network is trained completely, the figures of the adaptive Mel fiter banks dedicated to each network can be shown in the following way:

First, a siginicant prerequisite is to save the model (.h5 file).

Second, in order to extracts Mel filter coefficients from a model file and stores them in an Excel file, call the "show_AMs" function.

Third, call the "AMFBs_plot" function to plot and save the Mel filter banks.

Detailed example usage are described in the code.

# Citing
