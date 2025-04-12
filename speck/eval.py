import speck as sp
import numpy as np
from keras.models import model_from_json
from keras.models import load_model

wdir = './change_nets/'

# net6 = load_model("6_distinguisher_Inceptions_nets_change_depth_2.h5")
# net7 = load_model('7_distinguisher_Inceptions_nets_change_depth_2.h5')
wdir = './change_nets/'
# net8 = load_model(wdir + "model_8r_depth2_.h5")
net4 = load_model('4_distinguisher_Inceptions_nets_change_depth_2.h5')
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


X1, Y1 = sp.make_train_data(10 ** 7, 4)  
X2, Y2 = sp.make_train_data(10 ** 7, 4)  
X4 = np.concatenate((X1, X2), axis=0)
Y4 = np.concatenate((Y1, Y2), axis=0)

print('Testing neural distinguishers against 6 to 9 blocks in the ordinary real vs random setting');
print('4 rounds:');
evaluate(net4, X4, Y4);
# print('5 rounds:');
# evaluate(net5, X5, Y5);
# print('6 rounds:');
# evaluate(net6, X6, Y6);
# print('7 rounds:');
# evaluate(net7, X7, Y7);
# print('8 rounds:');
# evaluate(net8, X8, Y8);



