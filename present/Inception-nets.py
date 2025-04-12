import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.backend import concatenate
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation, Dropout
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from pickle import dump
import present as pre
import pandas as pd


bs = 12000
wdir = './Inception_nets/'


def cyclic_lr(num_epochs, high_lr, low_lr):
    res = lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)
    return res


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return res

def convert_to_binary(arr):
  X = np.zeros((4 * 64, len(arr[0])),dtype=np.uint8);
  for i in range(4 * 64):
    index = i // 64;
    offset = 64 - (i % 64) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);



def make_resnet(num_blocks=16, num_filters=32, num_outputs=1, d1=256, d2=256, word_size=16, ks=3, depth=5,
                reg_param=0.0001, final_activation='sigmoid'):
    inp = Input(shape=(num_blocks * word_size,))
    rs = Reshape((num_blocks, word_size))(inp)
    perm = Permute((2, 1))(rs)

    conv01 = Conv1D(num_filters, kernel_size=1, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv02 = Conv1D(num_filters, kernel_size=3, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv03 = Conv1D(num_filters, kernel_size=5, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv04 = Conv1D(num_filters, kernel_size=7, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    c2 = concatenate([conv01, conv02, conv03, conv04], axis=-1)
    conv0 = BatchNormalization()(c2)
    conv0 = Activation('relu')(conv0)
    shortcut = conv0

    for i in range(depth):
        conv1 = Conv1D(num_filters*4, kernel_size=ks, padding='same',
                       kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters*4, kernel_size=ks,
                       padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
        ks += 2
    # add prediction head
    # 展开，全连接层
    flat1 = Flatten()(shortcut)
    dense0 = Dropout(0.2)(flat1)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(dense0)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation,
                kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return (model)

def train_present_distinguisher(num_epochs, num_rounds=7, depth=1, folder='./'):
    net = make_resnet(depth=depth, reg_param=10 ** -5)
    # 使用混合优化器
    optimizer = Adam(learning_rate=0.001)
    net.compile(optimizer=optimizer, loss='mse', metrics=['acc'])
    
    Numm1 = 0;
	while(True):
        X1, Y1, C1 = pre.make_train_data(10 ** 7, num_rounds)  
        X2, Y2, C2 = pre.make_train_data(10 ** 7, num_rounds)		
		if(Numm1 < 10 ** 7):
			Numm1 += C1 + C2;
			X = np.concatenate((X1, X2), axis=0)
			Y = np.concatenate((Y1, Y2), axis=0)
		else:
			break;

	Numm2 = 0;
	while(True):
        X_eval1, Y_eval1, C_eval1 = pre.make_train_data(10 ** 7, num_rounds)  
        X_eval2, Y_eval2, C_eval2 = pre.make_train_data(10 ** 7, num_rounds)		
		if(Numm2 < 10 ** 7):
			Numm2 += C_eval1 + C_eval2;
			X_eval = np.concatenate((X_eval1, X_eval2), axis=0)
			Y_eval = np.concatenate((Y_eval1, Y_eval2), axis=0)
		else:
			break;

    check = make_checkpoint(wdir + 'best' + str(num_rounds) + 'depth' + str(depth) + '.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10)  # Early stopping

    h = net.fit(X, Y, epochs=num_epochs, batch_size=bs, validation_data=(X_eval, Y_eval),
                callbacks=[lr, check])

    np.save(wdir + 'h' + str(num_rounds) + 'r_depth' + str(depth) + '_val_acc.npy', h.history['val_acc'])
    np.save(wdir + 'h' + str(num_rounds) + 'r_depth' + str(depth) + '_val_loss.npy', h.history['val_loss'])
    np.save(wdir + 'h' + str(num_rounds) + 'r_depth' + str(depth) + '_train_acc.npy', h.history['acc'])  # 保存训练集精度
    np.save(wdir + 'h' + str(num_rounds) + 'r_depth' + str(depth) + '_train_loss.npy', h.history['loss'])  # 保存训练集损失
    dump(h.history, open(wdir + 'hist' + str(num_rounds) + 'r_depth' + str(depth) + '.p', 'wb'))

    print("Best validation accuracy: ", np.max(h.history['val_acc']))
    net.save(folder + str(num_rounds) + '_distinguisher_Inceptions_nets_depth_' + str(depth) + '.h5')

    return net, h

# net5, h5 = train_speck_distinguisher(50, num_rounds=5, depth=2)
# net6, h6 = train_speck_distinguisher(50, num_rounds=6, depth=2)
# 训练第7轮speck分辨器
# net7, h7 = train_present_distinguisher(50, num_rounds=6, depth=2)
net5, h5 = train_present_distinguisher(50, num_rounds=5, depth=2)