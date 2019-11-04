#!/usr/bin/env python
# coding: utf-8

# In[105]:


from os import listdir
import os
import shutil
import random
from numpy import asarray
from numpy import save


# In[76]:


dataset_home = 'dogs-vs-cats/'
sub_dirs = ["test1/","train/"]
label_dirs=["dogs/","cats/"]
for subdir in sub_dirs:
    for labeldir in label_dirs:
        newdir = dataset_home+subdir+labeldir
        os.mkdir(newdir)


# In[80]:


random.seed(1)
val_ratio=0.25

src_directory = 'dogs-vs-cats/train/train'

for file in listdir(src_directory):
    src = src_directory+'/'+file
    dst_dir = 'train/'
    if random.random() < val_ratio:
        dst_dir = 'test1/'
    if file.startswith('cat'):
        dst = dataset_home + dst_dir + 'cats/'  + file
        shutil.copyfile(src, dst)
    elif file.startswith('dog'):
        dst = dataset_home + dst_dir + 'dogs/'  + file
        shutil.copyfile(src, dst)


# In[106]:


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def define_model():
    model =Sequential()
    model.add(Conv2D(32,(3,3),activation="relu",kernel_initializer="he_uniform",padding="same",input_shape=(200,200,3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:





# In[107]:



# plot diagnostic learning curves
def summarize_diagnostics(history):
   # plot los
   pyplot.subplot(211)
   pyplot.title('Cross Entropy Loss')
   pyplot.plot(history.history['loss'], color='blue', label='train')
   pyplot.plot(history.history['val_loss'], color='orange', label='test')
   # plot accuracy
   pyplot.subplot(212)
   pyplot.title('Classification Accuracy')
   pyplot.plot(history.history['accuracy'], color='blue', label='train')
   pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
   # save plot to file
   filename = sys.argv[0].split('/')[-1]
   pyplot.savefig(filename + '_plot.png')
   pyplot.close()


# In[ ]:





# In[114]:


def run_test_harness():
    model = define_model()
    # create data generator
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    # prepare iterators
    train_it = datagen.flow_from_directory('dogs-vs-cats/train/',
                                           class_mode='binary', batch_size=64, target_size=(200, 200))
    test_it = datagen.flow_from_directory('dogs-vs-cats/test1/',
                                           class_mode='binary', batch_size=64, target_size=(200, 200))
    # fit model
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                                  validation_data=test_it, validation_steps=len(test_it), epochs=5, verbose=0)
    # evaluate model
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)
 


# In[ ]:


run_test_harness()


# In[ ]:





# In[ ]:




