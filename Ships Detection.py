#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#завантажуємо закодовані маски
url = 'https://raw.githubusercontent.com/Ram81/Airbus-Ship-Detection/master/train_ship_segmentations_v2.csv'
df = pd.read_csv(url,index_col=0)


# In[ ]:


import opendatasets as od
od.download("https://www.kaggle.com/mikaelstrauhs/airbus-ship-detection-train-set-30")


# In[ ]:


#завантажуємо тренувлаьні фото, складаємо список


# In[2]:


import os

input_dir="airbus-ship-detection-train-set-30/test_v3/test_v2/Images"
#input_dir = "images/"
img_size = (128, 128)
num_classes = 2
batch_sizez = 1000

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)

input_img_paths.sort()
print("Number of samples:", len(input_img_paths))
inp=[]
list(set(input_img_paths))
for input_path in zip(input_img_paths[:10]):
    print(input_path)
for ff in range(0,len(input_img_paths)):
        inp.append(input_img_paths[ff].replace("airbus-ship-detection-train-set-30/test_v3/test_v2/Images\\", ""))


# In[3]:


inp[0]


# In[29]:


from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
import cv2
from PIL import ImageOps

# Display input image #7
display(Image(filename=input_img_paths[3]))


# In[7]:


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T 


# In[ ]:


#вище розкодовуємо лейбел-фото


# In[25]:


import numpy as np
import matplotlib.pyplot as plt
#ac=rle_decode(df['EncodedPixels'][3])
#plt.imshow(ac)


# In[10]:


df['EncodedPixels'] = df['EncodedPixels'].fillna(0) #замінюмо нан на 0
#df=df[:57767]
df.head(8)


# In[ ]:





# In[11]:


df.rename(columns={ df.columns[0]: "EncodedPixels" }, inplace = True)
list(df)


# In[12]:


df = df.reset_index(drop=False)#вибираємо лейбели лише до наявних фото у тренувалному наборі
df.drop_duplicates(subset=['ImageId'], inplace=True)

df['prapor']=df['ImageId'].isin(inp)


# In[13]:


df.sort_values(by=['ImageId'])
df.head(9)


# In[14]:


df=df[df['prapor']==True]


# In[15]:


len(df)


# In[16]:


df.reset_index(drop=True, inplace=True)
df.drop(columns=['prapor'], inplace=True)


# In[17]:


df.shape


# In[19]:


from tensorflow.keras import layers
from tensorflow import keras

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### здійснюємо згортку та пулінг чергово
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # residual

    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual]) 
        previous_block_activation = x  

    ### РУХАЄМОСЬ У ЗВОРОТНЬОМУ НАПРЯМКУ

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual]) 
        previous_block_activation = x  

    # класифікація чи піксель до корабля чи ні
    outputs = layers.Conv2D(num_classes-1, 2, activation="sigmoid", padding="same")(x)

    model = keras.Model(inputs, outputs)
    return model


keras.backend.clear_session()

model = get_model(img_size, num_classes)

model.summary()


# In[428]:


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


# In[21]:


import random

#вибір входу та виходу для тренування
i = random.randint(0,5000)
def getx(odyn, dva):
  xx = np.zeros((dva-odyn,) + (128,128) + (3,), dtype="float32")
  count2=0
  for j in range (odyn,dva):
          img =PIL.ImageOps.autocontrast(load_img(input_img_paths[j], target_size=(128,128,3)))
          img=np.array(img)
          img2 = (img - np.min(img))/np.ptp(img) #нормалізація іксів
          xx[count2] = np.array(img2)
          count2+=1
  return xx
def gety(odyn,dva):
  y = np.zeros((dva-odyn,) + (128,128) + (1,), dtype="float32")
  count=0
  for j in range (odyn,dva):
            if df['EncodedPixels'][j]==0:
                img=np.zeros((128,128), dtype=int)
            else:
                img = rle_decode(df['EncodedPixels'][j])
                img=cv2.resize(img, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
            
            img=np.array(img)
            y[count] = img.reshape(128,128,1)

            count+=1
  return y


# In[359]:


plt.imshow(y[7])


# In[483]:


from keras import backend as K
model.compile(optimizer="rmsprop", loss="binary_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
]

# Тренуємо, зьеррігаємо кращу модель
epochs = 15
model.fit(x=getx(0,5000), y=gety(0,5000),epochs=epochs, validation_split=0.2,callbacks=callbacks)


# In[439]:


plt.imshow(gety(0,100)[7])


# In[35]:


model=keras.models.load_model("oxford_segmentation.h5") #здійснюємо передбачення
xx=getx(0,100)
xx5=xx[7].reshape(1,128,128,3)
img=np.array(model.predict(xx5))
print(img.shape)
nump=np.zeros((128,128))
img=img.reshape(128,128,)
plt.imshow(img)


# In[ ]:




