import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation, Dropout
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from pickle import dump
from keras.models import load_model, model_from_json
import present as pre 

bs = 12000
wdir = './change_present_nets/'


def augment_data(X, Y):
    augmented_X, augmented_Y = [], []
    for x, y in zip(X, Y):
        augmented_X.append(x)
        augmented_Y.append(y)
        # 添加噪声
        noise = np.random.normal(0, 0.01, x.shape)
        augmented_X.append(x + noise)
        augmented_Y.append(y)
    return np.array(augmented_X), np.array(augmented_Y)

def cyclic_lr(num_epochs, high_lr, low_lr):
    res = lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)
    return res


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return res


def make_resnet(num_blocks=4, num_filters=32, num_outputs=1, d1=128, d2=128, word_size=16, ks=3, depth=5,
                reg_param=0.0001, final_activation='sigmoid'):
    inp = Input(shape=(num_blocks * word_size * 2,))
    rs = Reshape((2 * num_blocks, word_size))(inp)
    perm = Permute((2, 1))(rs)
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)

    shortcut = conv0
    for i in range(depth):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = Dropout(0.5)(conv1)  # Dropout layer
        conv2 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])

    flat1 = Flatten()(shortcut)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense1 = Dropout(0.5)(dense1)  # Dropout layer
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    dense2 = Dropout(0.5)(dense2)  # Dropout layer
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)

    model = Model(inputs=inp, outputs=out)
    return model


def train_present_distinguisher(num_epochs, num_rounds=7, depth=1, folder='./'):
    net = load_model("7_distinguisher_Inceptions_nets_depth_2.h5")
    net_json = net.to_json()

    net_first = model_from_json(net_json)

    # net = make_resnet(depth=depth, reg_param=10 ** -5)
    # 使用混合优化器
    optimizer = Adam(learning_rate=0.001)
    net_first.compile(optimizer=optimizer, loss='mse', metrics=['acc'])
    net_first.load_weights("7_distinguisher_Inceptions_nets_depth_2.h5")

	X, Y , C = pre.make_train_data(10 ** 7, 8-3, diff=0x0000020200000202, datai=1)
	Numm1 = 0;
	while(True):
        X1, Y1, C1 = pre.make_train_data(10 ** 8, num_rounds-3, diff=0x0000020200000202)  
        X2, Y2, C2 = pre.make_train_data(10 ** 8, num_rounds-3, diff=0x0000020200000202)		
		if(Numm1 < 10 ** 7):
			Numm1 += C1 + C2;
			X = np.concatenate((X1, X2), axis=0)
			Y = np.concatenate((Y1, Y2), axis=0)
		else:
			break;

	Numm2 = 0;
	while(True):
        X_eval1, Y_eval1, C_eval1 = pre.make_train_data(10 ** 8, num_rounds-3, diff=0x0000020200000202) 
        X_eval2, Y_eval2, C_eval2 = pre.make_train_data(10 ** 8, num_rounds-3, diff=0x0000020200000202)	
		if(Numm2 < 10 ** 7):
			Numm2 += C_eval1 + C_eval2;
			X_eval = np.concatenate((X_eval1, X_eval2), axis=0)
			Y_eval = np.concatenate((Y_eval1, Y_eval2), axis=0)
		else:
			break;
    

    check = make_checkpoint(wdir + 'best' + str(num_rounds) + 'depth' + '.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
  

    net_first.fit(X, Y, epochs=num_epochs, batch_size=bs, validation_data=(X_eval, Y_eval),
                callbacks=[lr, check])

    # np.save(wdir + 'h' + str(num_rounds) + 'r_depth' + str(depth) + '_val_acc.npy', h.history['val_acc'])
    # np.save(wdir + 'h' + str(num_rounds) + 'r_depth' + str(depth) + '_val_loss.npy', h.history['val_loss'])
    # np.save(wdir + 'h' + str(num_rounds) + 'r_depth' + str(depth) + '_train_acc.npy', h.history['acc'])  # 保存训练集精度
    # np.save(wdir + 'h' + str(num_rounds) + 'r_depth' + str(depth) + '_train_loss.npy', h.history['loss'])  # 保存训练集损失
    # dump(h.history, open(wdir + 'hist' + str(num_rounds) + 'r_depth' + str(depth) + '.p', 'wb'))


    net_first.save("net_first.h5")


def second_stage(num_rounds=8):

    Numm3 = 0;
	while(True):
        X1, Y1, C1 = pre.make_train_data(10 ** 8, num_rounds)  
        X2, Y2, C2 = pre.make_train_data(10 ** 8, num_rounds)		
		if(Numm3 < 10 ** 7):
			Numm3 += C1 + C2;
			X = np.concatenate((X1, X2), axis=0)
			Y = np.concatenate((Y1, Y2), axis=0)
		else:
			break;

	Numm4 = 0;
	while(True):
        X_eval1, Y_eval1, C_eval1 = pre.make_train_data(10 ** 8, num_rounds)  
        X_eval2, Y_eval2, C_eval2 = pre.make_train_data(10 ** 8, num_rounds)		
		if(Numm4 < 10 ** 7):
			Numm4 += C_eval1 + C_eval2;
			X_eval = np.concatenate((X_eval1, X_eval2), axis=0)
			Y_eval = np.concatenate((Y_eval1, Y_eval2), axis=0)
		else:
			break;

    net = load_model("net_first.h5")
    net_json = net.to_json()

    net_second = model_from_json(net_json)
    net_second.compile(optimizer=Adam(learning_rate=10 ** -4), loss='mse', metrics=['acc'])

    net_second.load_weights("net_first.h5")

    check = make_checkpoint(wdir + 'best' + str(num_rounds) + '.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    net_second.fit(X, Y, epochs=50, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr, check])

    net_second.save("net_second.h5")


def stage_train(num_rounds=8):
    Numm5 = 0;
	while(True):
        X1, Y1, C1 = pre.make_train_data(10 ** 8, num_rounds)  
        X2, Y2, C2 = pre.make_train_data(10 ** 8, num_rounds)		
		if(Numm5 < 10 ** 7):
			Numm5 += C1 + C2;
			X = np.concatenate((X1, X2), axis=0)
			Y = np.concatenate((Y1, Y2), axis=0)
		else:
			break;

	Numm6 = 0;
	while(True):
        X_eval1, Y_eval1, C_eval1 = pre.make_train_data(10 ** 8, num_rounds)  
        X_eval2, Y_eval2, C_eval2 = pre.make_train_data(10 ** 8, num_rounds)		
		if(Numm6 < 10 ** 7):
			Numm6 += C_eval1 + C_eval2;
			X_eval = np.concatenate((X_eval1, X_eval2), axis=0)
			Y_eval = np.concatenate((Y_eval1, Y_eval2), axis=0)
		else:
			break;

    net = load_model("net_second.h5")
    net_json = net.to_json()

    net_third = model_from_json(net_json)
    net_third.compile(optimizer=Adam(learning_rate=10 ** -5), loss='mse', metrics=['acc'])
 
    net_third.load_weights("net_second.h5")

    check = make_checkpoint(wdir + 'best' + str(num_rounds) + '.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    net_third.fit(X, Y, epochs=50, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr, check])

    net_third.save(wdir + "model_" + str(num_rounds) + "r_depth2_"  + ".h5")


if __name__ == "__main__":

    train_present_distinguisher(50, num_rounds=8)
    second_stage(num_rounds=8)
    stage_train(num_rounds=8)