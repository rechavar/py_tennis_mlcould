#preprocesamiento y entrenamiento
from azureml.core import Run
import pandas as pd
import typing as t
from ast import literal_eval
import albumentations as a
import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
import re
from functools import partial
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from model import get_unet_model
import requests


LABEL = {'lat' : 1, 'ext' : 2, 'int_a' : 3, 'int_b' : 4}

class LogToAzure(Callback):
    '''Keras Callback for realtime logging to Azure'''
    def __init__(self, run):
        super(LogToAzure, self).__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        # Log all log data to Azure
        for k, v in logs.items():
            self.run.log(k, v)

def _prepare_dataset(df: pd.DataFrame, full_path) -> pd.DataFrame:

    fn_img_path = partial(_get_file_name, full_path)

    cleaning_fn = _chain(
        [
            _drop_useless,
            _string_to_list,
            fn_img_path
        ]
    )
    df = cleaning_fn(df)
    return df

def _chain(functions: t.List[t.Callable[[pd.DataFrame], pd.DataFrame]]):
    def helper(df):
        for fn in functions:
            df = fn(df)
        return df

    return helper

def _drop_useless(df):
   return df.drop(['id', 'annotator', 'annotation_id'], axis = 1)

def _string_to_list(df: pd.DataFrame):  
    new_dict = df.label.apply(literal_eval)
    df['label'] = pd.Series(new_dict)
    return df

def _preprocess_image(img_size, img_path):
    #Resize
    size =(img_size, img_size)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(image, size, interpolation= cv2.INTER_AREA)

    return img_resized

def _regex_search(img_name):
    pattern = re.compile(r"/(\d+)")
    matches = pattern.finditer(img_name)

    for match in matches:
        file_name = match.group() + ".png"

    return file_name

def _get_file_name(full_path, dataset: pd.DataFrame):
    dataset['image'] = full_path + dataset.image.apply(_regex_search)
    return dataset

def build_sources(data_dir, image_size = 512,mode = 'train', gray = False):
    #Debe retornar un dataframe con la estrustura[path_img, img, mask]
    datasets_names = os.listdir(data_dir)
    clean_ann = pd.DataFrame()
    for dataset in datasets_names:
        print(dataset)
        full_dataset_path = os.path.join(data_dir, dataset)
        ann_list = [ file for file in os.listdir(
            full_dataset_path) if file.endswith('csv')]
            
        for ann in ann_list:
            ann_df = pd.read_csv(os.path.join(full_dataset_path, ann), sep=';')
            clean_aux = _prepare_dataset(ann_df, full_dataset_path)
            fn_pre = partial(_preprocess_image, image_size)
            clean_aux['img'] = clean_aux.image.apply(fn_pre)
            clean_aux['mask'] = list(_create_masks(clean_aux, image_size).values())
        clean_ann = clean_ann.append(clean_aux)

    if gray:
        clean_ann['img'] = clean_ann.img.apply(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY))
    return clean_ann

def im_show_three(dataset: pd.DataFrame, title = True):
    fig, ax = plt.subplots(3,2, figsize=(100,100))
    rnd_sample_df = dataset.sample(n=3, random_state=np.random.RandomState())

    for df_sample, row_figure in zip(rnd_sample_df.iterrows(), ax):
        if title:
            row_figure[0].set_title(df_sample[1]['image'], size=100)
        row_figure[0].imshow(df_sample[1].img)
        row_figure[1].imshow(df_sample[1]['mask'])

    fig.tight_layout()
    plt.show()


def _get_transformations():

    transformations = [
        a.Compose([
            a.HorizontalFlip(p = 0.5),
            a.RandomBrightnessContrast(p = 0.3)
        ]),

        a.Compose([
            a.RandomBrightnessContrast(),    
            a.RandomGamma(p=1),    
        ]),
    ]

    return transformations

def _augment_image(img, mask, save = False):
    #img -> image route
    #mask -> mask route. spected a numpy array
    img_t = []
    mask_t = []
    for transform in _get_transformations():
        transformed = transform(image = img, mask = mask)
        img_t.append(transformed['image']) 
        mask_t.append(transformed['mask'])

    return img_t, mask_t

def augment_dataset(dataframe: pd.DataFrame):

    output = pd.DataFrame()
    for row in dataframe.iterrows():
        output = output.append({'img' : row[1]['img'],
                        'mask' : row[1]['mask']}, ignore_index=True)
        aug_img, aug_mask = _augment_image(row[1].img, row[1]['mask'])

        for img, mask in zip(aug_img, aug_mask):
            output = output.append({'img' : img,
                            'mask' : mask}, ignore_index=True)

    return output

def _create_masks(df: pd.DataFrame, img_size):
    output_df = {}
    for row in df.iterrows():
        img = Image.new('L', (img_size, img_size), 0)
        factor = img_size/100
        mask = []
        literal_label = row[1].label
        for label in literal_label:

            if label['polygonlabels'][0] not in LABEL.keys():
                print(f'check annotations on file: {row[1].image}')
                continue

            good_points = [((x1*factor),(x2*factor)) for x1,x2 in label['points']]
            ImageDraw.Draw(img).polygon(good_points,
                                        outline=LABEL[label['polygonlabels'][0]],
                                        fill=LABEL[label['polygonlabels'][0]])
            mask = np.array(img)
        
        output_df[row[1].image] = mask

    return output_df

def one_hot_mask(img):
  n_classes = 5
  one_hot = np.zeros((img.shape[0], img.shape[1], n_classes))
  for i, unique_value in enumerate(np.unique(img)):
    one_hot[:, :, i][img == unique_value] = 1

  return one_hot

def download_h5(file_name = './outputs/model-dsbowl2018-1.h5', 
                url = 'https://s3.amazonaws.com/rlx/model-dsbowl2018-1.h5'):

    if not os.path.isdir('outputs'):
        os.mkdir('outputs')

    if os.path.exists(file_name):
        print('File already on dir, skipping download')
        return

    r = requests.get(url, allow_redirects=True)

    open(file_name, 'wb').write(r.content)

def main(dataset):

    DATASET_PATH = dataset

    img_size = 512

    ann = build_sources(DATASET_PATH, img_size, gray=True)

    augmented_df = augment_dataset(ann)
    augmented_df['mask'] = augmented_df['mask'].apply(one_hot_mask)
    augmented_df['img'] = augmented_df['img'].apply(list)
    augmented_df['mask'] = augmented_df['mask'].apply(list)

    X_train, X_test, y_train, y_test = train_test_split(augmented_df['img'], augmented_df['mask'].values, test_size=0.33, random_state=42)

    unet = get_unet_model((img_size,img_size,1),5)

    unet.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy())

    X_train = np.stack(X_train)
    y_train = np.stack(y_train)

    download_h5()

    if os.path.isfile('./outputs/model-dsbowl2018-1.h5'):
        print('Descarga exitosa')
    
    else:
        print('el archivo h5 no se descargo')

    earlystopper = EarlyStopping(patience=5, verbose=1)

    run = Run.get_submitted_run()

    log_to_azure = LogToAzure(run)

    checkpointer = ModelCheckpoint('outputs/model-dsbowl2018-1.h5', verbose=1, save_best_only=True)

    results = unet.fit(X_train, y_train, validation_split=0.1, batch_size=8, epochs=30, 
                   callbacks=[earlystopper, log_to_azure, checkpointer])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasetpath", default=' azureml-blobstore-f03e9ff4-a008-44e0-a0a2-67c3090aaa67/datasets/pytennis/')

    args = parser.parse_args()

    main(args.datasetpath)



