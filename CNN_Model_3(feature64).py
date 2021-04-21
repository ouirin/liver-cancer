import os
import cv2
import glob
import ntpath
import random
import warnings
import numpy as np
import seaborn as sns

from imblearn import over_sampling
from keras.utils import np_utils 
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler,ModelCheckpoint
from keras.metrics import categorical_accuracy, categorical_crossentropy
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, AveragePooling2D, Concatenate, GlobalMaxPooling2D,concatenate
from keras.layers.advanced_activations import ReLU

USE_DROPOUT = False
LEARN_RATE = 0.001
Height = 128
Weight = 128
Channel= 3

if __name__ == "__main__":
    
    if True:
            
        train(model_name="classifier_CNN", load_weights_path=None)      

def train(model_name, load_weights_path):
    
    batch_size = 8
    
    #获得训练和测试集合，以：路径、class label的形式保存
    train_files, holdout_files = get_train_holdout_files()
    
    #训练数据集
    train_gen = data_generator(batch_size, train_files, train_set=True)
    
    #测试数据集
    holdout_gen = data_generator(batch_size, holdout_files, train_set=False)
   
    #动态设置学习率
    learnrate_scheduler = LearningRateScheduler(step_decay)
    
    #获取model
    model = get_net(load_weight_path=load_weights_path)
    
    model.fit_generator(generator=train_gen, samples_per_epoch=len(train_files), nb_epoch=10, verbose=1, validation_data=holdout_gen, nb_val_samples=len(holdout_files), class_weight="auto", callbacks=[learnrate_scheduler])
 
    model.save("workdir/model_" + model_name + "_"  + "_end.hd5")

def step_decay(epoch):
    res = 0.001
    if epoch > 100:
        res = 0.0001
    print("learnrate: ", res, " epoch: ", epoch)
    return res

def get_net(input_shape=(Height, Weight, Channel), load_weight_path=None) -> Model:  #期待返回类型为model
    
    inputs = Input(shape=input_shape, name="input")
    x = inputs
    
    ##################################################################################################################
    x_ident_1 = x
    x_ident_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid')(x_ident_1)
    # 1st layer group
    x = Convolution2D(16, 3, 3, activation=None, border_mode='same', name='conv1a', subsample=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Convolution2D(16, 3, 3, activation=None, border_mode='same', name='conv1b', subsample=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid', name='pool1')(x)
    x = Concatenate(axis=3)([x,x_ident_1])
    
    ##################################################################################################################
    x_ident_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid')(x_ident_1)
    x_ident_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid')(x)
    # 2nd layer group
    x = Convolution2D(32, 3, 3, activation=None, border_mode='same', name='conv2a', subsample=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Convolution2D(32, 3, 3, activation=None, border_mode='same', name='conv2b', subsample=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid', name='pool2')(x)
    x = Concatenate(axis=3)([x,x_ident_1,x_ident_2])

    ##################################################################################################################
    x_ident_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid')(x_ident_1)
    x_ident_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid')(x_ident_2)
    x_ident_3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid')(x)
    # 3rd layer group
    x = Convolution2D(64, 3, 3, activation=None, border_mode='same', name='conv3a', subsample=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Convolution2D(64, 3, 3, activation=None, border_mode='same', name='conv3b', subsample=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid', name='pool3')(x)
    x = Concatenate(axis=3)([x,x_ident_1,x_ident_2,x_ident_3])
     
    ##################################################################################################################
    x_ident_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid')(x_ident_1)
    x_ident_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid')(x_ident_2)
    x_ident_3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid')(x_ident_3)
    x_ident_4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid')(x)
    # 4th layer group
    x = Convolution2D(128, 3, 3, activation=None, border_mode='same', name='conv4a', subsample=(1, 1),)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Convolution2D(128, 3, 3, activation=None, border_mode='same', name='conv4b', subsample=(1, 1),)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid', name='pool4')(x)
    x = Concatenate(axis=3)([x,x_ident_1,x_ident_2,x_ident_3,x_ident_4])
    
    x = GlobalMaxPooling2D()(x)
    x = BatchNormalization(name="final_features_344")(x)
    
    ##################################################################################################################
    if USE_DROPOUT:
        x = Dropout(p=0.3)(x)
        
    x = Dense(64, activation='relu',name="final_features_64")(x)
    out_class = Dense(5, activation='softmax', name='out_class')(x)

    model = Model(input=inputs, output=out_class)
    
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)

    #编译模型
    model.compile(optimizer=SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True), loss={ "out_class": "categorical_crossentropy" }, metrics={"out_class": [categorical_accuracy, categorical_crossentropy] } )
    model.summary(line_length=120)

    return model

def data_generator(batch_size, record_list, train_set):
    
    while True:
        
        batch_index = 0
        image_list = []
        label_list = []

        if train_set:
            random.shuffle(record_list)

        #逐一遍历所有数据
        for index, record_item in enumerate(record_list):

            sample_path = record_item[0]
            sample_label = record_item[1]

            #转换成多分类标签
            sample_label = np_utils.to_categorical(sample_label,5)  

            #读取图片、修改尺寸、标准化
            sample_image = cv2.imread(sample_path)
            sample_image = (sample_image - np.average(sample_image)) / np.std(sample_image)
            sample_image = sample_image.reshape(1, sample_image.shape[0], sample_image.shape[1], 3)

            #数据增强
            if train_set:   
                if random.randint(0, 100) > 50:
                    sample_image = np.fliplr(sample_image)
                if random.randint(0, 100) > 50:
                    sample_image = np.flipud(sample_image)
                if random.randint(0, 100) > 50:
                    sample_image = sample_image[:,::-1]
                if random.randint(0, 100) > 50:
                    sample_image = sample_image[::-1, :]

            #添加数据
            image_list.append(sample_image)
            label_list.append(sample_label)
            batch_index += 1

            if batch_index >= batch_size:
                x = np.vstack(image_list)
                y = np.vstack(label_list)
                yield x, y
                image_list = []
                label_list = []
                batch_index = 0

def get_train_holdout_files(current_iteration = 1):
    
    print("Get train/holdout files.")
        
    src_dir = "D:/jupyter-notebook/LiverCancer/Data_Description/" + str(current_iteration) +"/"

    #分割训练数据和测试数据  
    train_samples = pd.read_csv(src_dir + "train.csv")["file_path"].tolist()
    holdout_samples = pd.read_csv(src_dir + "holdout.csv")["file_path"].tolist()
    print("Train Count: ", len(train_samples), ", Holdout Count: ", len(holdout_samples))

    #建立描述集合
    train_rep = []
    holdout_rep = []
    sets = [[train_rep, train_samples], [holdout_rep, holdout_samples]]

    for set_item in sets:

        rep = set_item[0]
        samples = set_item[1]

        for index, sample_path in enumerate(samples):

            if "grade0" in sample_path:
                sample_label = 0
            elif "grade1" in sample_path:
                sample_label = 1
            elif "grade2" in sample_path:
                sample_label = 2
            elif "grade3" in sample_path:
                sample_label = 3
            elif "grade4" in sample_path:
                sample_label = 4

            rep.append([sample_path, sample_label])

    print("Train Count: ", len(train_rep), ", Holdout Count: ", len(holdout_rep))

    return train_rep, holdout_rep