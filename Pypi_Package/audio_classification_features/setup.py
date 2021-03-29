from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='audio_classification_features',
      version='2.0',
      description='Complete Package for Audio Classification',
      packages=['audio_classification_features'],
      author="Sumit Paul",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/Sumit189/Audio-Classification-Using-CNN",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent",
        ],
      install_requires=[
          'pandas',
          'numpy',
          'tqdm',
          'scipy',
          'librosa',
          'keras',
          'python_speech_features'
      ],
      python_requires='>=3.6',
      author_email="sumit.18.paul@gmail.com",
      keywords=['ML','audio-classification','audio-features'],
      zip_safe=False)