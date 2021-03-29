import pandas as pd
import os
import shutil
import numpy as np
import pickle
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
import librosa
from keras.utils import to_categorical

class audio_features:
  def extractor(self,audio_dataset):
    self.make_labels(audio_dataset)

  def envelope(self,y,rate, threshold):
    mask=[]
    y = pd.Series(y).apply(np.abs)
    y_mean=y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
      if mean>threshold:
        mask.append(True)
      else:
        mask.append(False)
    return mask

  def make_labels(self,loc):
    f_name=[]
    label=[]
    path=loc
    for r,d,f in os.walk(path):
      for file in f:
        file_path=r+'/'+file
        ext_label=os.path.basename(os.path.dirname(file_path))
        f_name.append(file)
        label.append(ext_label)
    df= pd.DataFrame(list(zip(f_name,label)), 
               columns =['fname','label']) 
    df = df.sample(frac = 1)
    df.to_csv('generated_dataset.csv',index=False)
    print("Class : Label")
    classes_got=[]
    classes_got= [f_got for f_got in sorted(os.listdir(path))]
    for i in range(len(classes_got)):
      print("{} : {}".format(classes_got[i],i))
    print("\nBuilding Features...")
    self.downsample(loc,'generated_dataset.csv')

  def downsample(self,loc,data_csv):
    df=pd.read_csv(data_csv)
    path=os.path.join(os.getcwd(),'cleaned_dataset')
    try:
      os.mkdir(path)
      #Downsampling
      if len(os.listdir('cleaned_dataset')) ==0:
        for label,f in zip(df.label,df.fname):
          signal, rate=librosa.load(loc+'/'+str(label)+'/'+f,sr=16000)
          mask=self.envelope(signal,rate,0.0005)
          wavfile.write(filename='cleaned_dataset/'+f,rate=rate,data=signal[mask])
        x,y=self.build_rand_feat(loc,data_csv)
        np.save('x.npy', x) 
        np.save('y.npy', y) 
    except OSError as error: 
      print(error)
      print()
      choice=input("Would You Like To Delete Cleaned_Dataset Folder And Try Again? Y/N")
      if choice=='Y' or choice=='y':
        try:
          os.remove('x.npy')
        except OSError:
          pass
        try:
          os.remove('y.npy')
        except OSError:
          pass
        shutil.rmtree('cleaned_dataset')
        self.downsample(loc,data_csv)
      else:
        print("Using Previous Cleaned Dataset")
        x,y=self.build_rand_feat(loc,data_csv)
        np.save('x.npy', x) 
        np.save('y.npy', y) 
    
  def build_rand_feat(self,loc,data_csv):
    try:
      os.remove('minmax_temp.txt')
    except OSError:
      pass
    df=pd.read_csv(data_csv)
    df.set_index('fname',inplace=True)
    for f,l in zip(df.index,df.label):
      rate, signal =  wavfile.read(loc+'/'+str(l)+'/'+f)
      df.at[f, 'length'] = signal.shape[0]/rate
    classes= list(np.unique(df.label))
    class_dist = df.groupby(['label'])['length'].mean()

    n_samples=2*int(df['length'].sum()/0.1)
    prob_dist=class_dist/class_dist.sum()
    choices=np.random.choice(class_dist.index,p = prob_dist)

    X=[]
    y=[]
    num_classes=len(np.unique(df.label))
    
    _min,_max=float('inf'),-float('inf')
    rate=16000
    step=int(rate/10)
    nfeat=13
    nfilt=26
    nfft=512
    min_=0
    max_=0
    df=df.reset_index()

    for _ in tqdm(range(n_samples)):
      rand_class=np.random.choice(class_dist.index,p=prob_dist)
      file=np.random.choice(df[df.label==rand_class].index)
      rate,wav=wavfile.read('cleaned_dataset/'+df.fname[file])

      label=df.at[file,'label']
      rand_index=np.random.randint(0,wav.shape[0]-step)
      sample=wav[rand_index:rand_index+step]
      X_sample = mfcc(sample,rate,numcep=nfeat,nfilt=nfilt,nfft=nfft)
      _min=min(np.amin(X_sample), _min)
      _max=max(np.amax(X_sample), _max)
      X.append(X_sample)
      y.append(classes.index(label))

    min_=_min
    max_=_max

    X, y = np.array(X), np.array(y)
    X=(X-_min)/(_max-_min)
    X=X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
    y=to_categorical(y,num_classes)
    minmax_dict = {'max': max_, 'min': min_}
    minmax_file = open('minmax_temp.txt', 'wb')
    pickle.dump(minmax_dict, minmax_file)
    minmax_file.close()
    try:
      os.remove('generated_dataset.csv')
    except OSError:
      pass
    return X,y

  #function to make predictions
  def prediction(self,audioloc,model):
    y_pred=[]
    minmax_frm_file = open('minmax_temp.txt', 'rb')
    min_max_dict = pickle.load(minmax_frm_file)
    
    print("Extracting audio features and making prediction")
    rate,wav = wavfile.read(audioloc)
    for i in range(0,wav.shape[0]-1600, 1600):
      sample = wav[i:i+1600]
      x = mfcc(sample, rate, numcep=13, nfilt=26, nfft=512)
      x = (x - min_max_dict['min'])/(min_max_dict['max'] - min_max_dict['min'])
      x = x.reshape(1, x.shape[0], x.shape[1],1 )
      y_hat = model.predict(x)
      y_pred.append(y_hat)
    return y_pred

  def make_prediction(self,filename,num_classes,model):
    signal, rate = librosa.load(filename, sr=16000)
    mask = self.envelope(signal, rate, 0.0005)
    os.remove(filename)
    wavfile.write(filename=filename, rate=rate, data=signal[mask])

    y_pred=self.prediction(filename,model)
    flat_pred = []
    for i in y_pred:
        for j in i:
            flat_pred.extend(j)
    y_pred = np.argmax(y_pred)
    y_pred=(y_pred)%num_classes
    return y_pred