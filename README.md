# Audio-Classification-Using-CNN
[![Downloads](https://pepy.tech/badge/audio-classification-features)](https://pepy.tech/project/audio-classification-features)

It is made to extract the features from any audio dataset. User's have to provide location of the dataset folder and this library will produce x and y npy files. 

# Installation
```sh
$ pip install audio_classification_features
```

## Usage
### Making Training Dataset
```py
from audio_classification_features import audio_features as af
af().extractor('dataset')
```

These npy files are loaded with numpy using following commands:
```py
import numpy
x=np.load('x.npy')
y=np.load('y.npy')
```

### Training Custom Built Model
```py
#input_shape generated by this package is of shape (9,13,1)
input_shape=(9,13,1)
```

### Making Predictions
```py 
from audio_classification_features import audio_features as af

#filename example: audio_test.wav
#num_classes same as above 
#model

af().make_prediction(filename,num_classes,model)
```

# Dataset Structure
>folder name of audio will be used as label.

* dataset
    * audio class 1
        * audiofile.wav
    * audio class 2
        * audiofile.wav

# LICENSE
```
GNU General Public License v3.0
```
