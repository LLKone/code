import present as pre
import numpy as np
import pandas as pd
from keras.models import load_model



path = './present_change_inputdiff/'

def convert_to_binary(arr):
  X = np.zeros((4 * 64, len(arr[0])),dtype=np.uint8);
  for i in range(4 * 64):
    index = i // 64;
    offset = 64 - (i % 64) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);

def read_csv():#读取训练数据
    file_path = '6round_test_head.csv'
    df = pd.read_csv(file_path)  # 使用openpyxl引擎读取Excel文件
    A = np.array(df['mergedarray0'].tolist() , dtype='uint64')
    B = np.array(df['mergedarray1'].tolist() , dtype='uint64')
    C = np.array(df['mergedarray2'].tolist() , dtype='uint64')
    D = np.array(df['mergedarray3'].tolist() , dtype='uint64')
    X = convert_to_binary([A, B, C, D])
    Y = np.array(df['Y0'].tolist() , dtype='uint16')
    return (X,Y)





net6 = load_model('6_distinguisher_Inceptions_nets_depth_2.h5')

def evaluate(net,X,Y):
    
    Z = net.predict(X,batch_size=10000).flatten();
    Zbin = (Z > 0.5);
    
    diff = Y - Z; mse = np.mean(diff*diff);
    
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]==1) / n1;
    
    tnr = np.sum(Zbin[Y==0] == 0) / n0;
    
    mreal = np.median(Z[Y==1]);
    
    high_random = np.sum(Z[Y==0] > mreal) / n0;
    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr, "MSE:", mse);
    print("Percentage of random pairs with score higher than median of real pairs:", 100*high_random);

Numm1 = 0;
	while(True):
        X1, Y1, C1 = pre.make_train_data(10 ** 7, 6)  
        X2, Y2, C2 = pre.make_train_data(10 ** 7, 6)		
		if(Numm1 < 10 ** 7):
			Numm1 += C1 + C2;
			X6 = np.concatenate((X1, X2), axis=0)
			Y6 = np.concatenate((Y1, Y2), axis=0)
		else:
			break;

	



print('Testing neural distinguishers against 6 to 9 blocks in the ordinary real vs random setting');

print('6 rounds:');
evaluate(net6, X6, Y6);



