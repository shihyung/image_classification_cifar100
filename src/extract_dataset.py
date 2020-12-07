import torch
import torchvision
import pandas as pd
import numpy as np
import PIL
import tqdm

'''
classes=['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
         'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 
         'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
         'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 
         'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 
         'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 
         'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 
         'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
         'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 
         'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
'''

ds_train=torchvision.datasets.CIFAR100(root='./', train= True, download = True)
image_id=['']*len(ds_train.data)
label=[999]*len(ds_train.data)
lb_class=['']*len(ds_train.data)
for idx in tqdm.tqdm(range(len(ds_train.data)),total=len(ds_train.data)):
    fname=str(idx).zfill(6)+'.png'
    img=PIL.Image.fromarray(ds_train.data[idx])
    img.save(f'train_images/{fname}')
    tgt=ds_train.targets[idx]

    image_id[idx]=fname
    label[idx]=tgt
    lb_class[idx]=ds_train.classes[tgt]
df=pd.DataFrame({'image_id':image_id,'label':label,'lb_class':lb_class})
df.to_csv('./train.csv',index=False)

ds_test=torchvision.datasets.CIFAR100(root='./', train= False, download = True)
image_id=['']*len(ds_test.data)
label=[999]*len(ds_test.data)
lb_class=['']*len(ds_test.data)
for idx in tqdm.tqdm(range(len(ds_test.data)),total=len(ds_test.data)):
    fname=str(idx).zfill(6)+'.png'
    img=PIL.Image.fromarray(ds_test.data[idx])
    img.save(f'test_images/{fname}')
    tgt=ds_test.targets[idx]

    image_id[idx]=fname
    label[idx]=tgt
    lb_class[idx]=ds_test.classes[tgt]
df=pd.DataFrame({'image_id':image_id,'label':label,'lb_class':lb_class})
df.to_csv('./test.csv',index=False)
