import numpy as np
from os import urandom
import pandas as pd

def WORD_SIZE():
    return(16);

def ALPHA():
    return(7);

def BETA():
    return(2);

MASK_VAL = 2 ** WORD_SIZE() - 1;

def shuffle_together(l):
    state = np.random.get_state();
    for x in l:
        np.random.set_state(state);
        np.random.shuffle(x);

def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)));

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL));

def enc_one_round(p, k):
    c0, c1 = p[0], p[1];
    c0 = ror(c0, ALPHA());
    c0 = (c0 + c1) & MASK_VAL;
    c0 = c0 ^ k;
    c1 = rol(c1, BETA());
    c1 = c1 ^ c0;
    return(c0,c1);

def dec_one_round(c,k):
    c0, c1 = c[0], c[1];
    c1 = c1 ^ c0;
    c1 = ror(c1, BETA());
    c0 = c0 ^ k;
    c0 = (c0 - c1) & MASK_VAL;
    c0 = rol(c0, ALPHA());
    return(c0, c1);

def expand_key(k, t):
    ks = [0 for i in range(t)];
    ks[0] = k[len(k)-1];
    l = list(reversed(k[:len(k)-1]));
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i);
    return(ks);

def encrypt(p, ks):
    x, y = p[0], p[1];
    for k in ks:
        x,y = enc_one_round((x,y), k);
    return(x, y);

def decrypt(c, ks):
    x, y = c[0], c[1];
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k);
    return(x,y);

def check_testvector():
  key = (0x1918,0x1110,0x0908,0x0100)
  pt = (0x6574, 0x694c)
  ks = expand_key(key, 22)
  ct = encrypt(pt, ks)
  if (ct == (0xa868, 0x42f2)):
    print("Testvector verified.")
    return(True);
  else:
    print("Testvector not verified.")
    return(False);

# def check_testvector():
#     key = (0x1918, 0x1110, 0x0908, 0x0100)
#     pt = (0x6574, 0x694c)
#     ks = expand_key(key, 22)
#     print(ks)
#     ct = encrypt(pt, ks)
#     print(ct)
#     ct0 = dec_one_round(ct, ks[21])
#     print(ks[21])
#     print(ct0)
#     k1 = expand_key(key, 21)
#     print(k1)
#     c1 = encrypt(key, k1)
#     print(c1)
#     if (ct == (0xa868, 0x42f2)):
#         print("Testvector verified.")
#         return (True);
#     else:
#         print("Testvector not verified.")
#         return (False);
#
# check_testvector()
#convert_to_binary takes as input an array of ciphertext pairs
#where the first row of the array contains the lefthand side of the ciphertexts,
#the second row contains the righthand side of the ciphertexts,
#the third row contains the lefthand side of the second ciphertexts,
#and so on
#it returns an array of bit vectors containing the same data
def convert_to_binary(arr):
  X = np.zeros((8 * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
  for i in range(8 * WORD_SIZE()):
    index = i // WORD_SIZE();
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);

#takes a text file that contains encrypted block0, block1, true diff prob, real or random
#data samples are line separated, the above items whitespace-separated
#returns train data, ground truth, optimal ddt prediction
def readcsv(datei):
    data = np.genfromtxt(datei, delimiter=' ', converters={x: lambda s: int(s,16) for x in range(2)});
    X0 = [data[i][0] for i in range(len(data))];
    X1 = [data[i][1] for i in range(len(data))];
    Y = [data[i][3] for i in range(len(data))];
    Z = [data[i][2] for i in range(len(data))];
    ct0a = [X0[i] >> 16 for i in range(len(data))];
    ct1a = [X0[i] & MASK_VAL for i in range(len(data))];
    ct0b = [X1[i] >> 16 for i in range(len(data))];
    ct1b = [X1[i] & MASK_VAL for i in range(len(data))];
    ct0a = np.array(ct0a, dtype=np.uint16); ct1a = np.array(ct1a,dtype=np.uint16);
    ct0b = np.array(ct0b, dtype=np.uint16); ct1b = np.array(ct1b, dtype=np.uint16);
    
    #X = [[X0[i] >> 16, X0[i] & 0xffff, X1[i] >> 16, X1[i] & 0xffff] for i in range(len(data))];
    X = convert_to_binary([ct0a, ct1a, ct0b, ct1b]); 
    Y = np.array(Y, dtype=np.uint8); Z = np.array(Z);
    return(X,Y,Z);

#baseline training data generator
def make_train_data(n, nr, diff=(0x0040,0)):
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
    plain2l = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain2r = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain3l = plain2l ^ diff[0]; plain3r = plain2r ^ diff[1]
    keys_list = expand_key(keys, nr);
    
    x = nr // 2
    kr = []
    for i in range(x):
      kr.append(keys_list[i])

    ct0l, ct0r = encrypt((plain0l, plain0r), kr)
    ct1l, ct1r = encrypt((plain1l, plain1r), kr)
    ct2l, ct2r = encrypt((plain2l, plain2r), kr)
    ct3l, ct3r = encrypt((plain3l, plain3r), kr)

    joined_elements0l = []
    joined_elements0r = []
    joined_elements1l = []
    joined_elements1r = []
    joined_elements2l = []
    joined_elements2r = []
    joined_elements3l = []
    joined_elements3r = []
    ks = []
    
    for i in range(n):
      if ((ct0l[i] ^ ct1l[i] == ct2l[i] ^ ct3l[i]) and (ct0r[i] ^ ct1r[i] == ct2r[i] ^ ct3r[i])):

          joined_elements0l.append(ct0l[i])
          joined_elements0r.append(ct0r[i])
          joined_elements1l.append(ct1l[i])
          joined_elements1r.append(ct1r[i])
          joined_elements2l.append(ct2l[i])
          joined_elements2r.append(ct2r[i])
          joined_elements3l.append(ct3l[i])
          joined_elements3r.append(ct3r[i])
          k = []
          for j in range(nr):
              k.append(keys_list[j][i])
          ks.append(k)

    joined_elements0l = np.array(joined_elements0l, dtype=np.uint16)
    joined_elements0r = np.array(joined_elements0r, dtype=np.uint16)
    joined_elements1l = np.array(joined_elements1l, dtype=np.uint16)
    joined_elements1r = np.array(joined_elements1r, dtype=np.uint16)
    joined_elements2l = np.array(joined_elements2l, dtype=np.uint16)
    joined_elements2r = np.array(joined_elements2r, dtype=np.uint16)
    joined_elements3l = np.array(joined_elements3l, dtype=np.uint16)
    joined_elements3r = np.array(joined_elements3r, dtype=np.uint16)
    kw = []
    for i in range(nr):
        kq = []
        for j in range(len(ks)):
            kq.append(ks[j][i])
        kw.append(kq)
    
    kt = []
    for i in range(x, nr):
        kt.append(kw[i])
    kt = np.array(kt, dtype=np.uint16)
    # print(kt)
    ck0l, ck0r = encrypt((joined_elements0l, joined_elements0r), kt)
    ck1l, ck1r = encrypt((joined_elements1l, joined_elements1r), kt)
    ck2l, ck2r = encrypt((joined_elements2l, joined_elements2r), kt)
    ck3l, ck3r = encrypt((joined_elements3l, joined_elements3r), kt)

    c0l = ck0l.tolist()
    c0r = ck0r.tolist()
    c1l = ck1l.tolist()
    c1r = ck1r.tolist()
    c2l = ck2l.tolist()
    c2r = ck2r.tolist()
    c3l = ck3l.tolist()
    c3r = ck3r.tolist()
    

    tn = len(ck0l)
    Y = np.random.randint(1, 2, 2 * tn, dtype=np.uint8)
    Y[:tn] = 0
    # np.random.shuffle(Y)
    num_rand_samples = np.sum(Y == 0)
    # print(num_rand_samples)
    rp0l = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    rp0r = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    rp1l = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    rp1r = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    rp2l = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    rp2r = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    rp3l = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    rp3r = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    ky = []
    for i in range(0, x):
        ky.append(kw[i])
    ky = np.array(ky, dtype=np.uint16)
    rc0l, rc0r = encrypt((rp0l, rp0r), ky)
    rc1l, rc1r = encrypt((rp1l, rp1r), ky)
    rc2l, rc2r = encrypt((rp2l, rp2r), ky)
    rc3l, rc3r = encrypt((rp3l, rp3r), ky)

    rk0l, rk0r = encrypt((rc0l, rc0r), kt)
    rk1l, rk1r = encrypt((rc1l, rc1r), kt)
    rk2l, rk2r = encrypt((rc2l, rc2r), kt)
    rk3l, rk3r = encrypt((rc3l, rc3r), kt)

    r0l = rk0l.tolist()
    r0r = rk0r.tolist()
    r1l = rk1l.tolist()
    r1r = rk1r.tolist()
    r2l = rk2l.tolist()
    r2r = rk2r.tolist()
    r3l = rk3l.tolist()
    r3r = rk3r.tolist()


    merged0l = r0l + c0l
    merged0r = r0r + c0r
    merged1l = r1l + c1l
    merged1r = r1r + c1r
    merged2l = r2l + c2l
    merged2r = r2r + c2r
    merged3l = r3l + c3l
    merged3r = r3r + c3r

    merged0l = np.array(merged0l, dtype=np.uint16)
    merged0r = np.array(merged0r, dtype=np.uint16)
    merged1l = np.array(merged1l, dtype=np.uint16)
    merged1r = np.array(merged1r, dtype=np.uint16)
    merged2l = np.array(merged2l, dtype=np.uint16)
    merged2r = np.array(merged2r, dtype=np.uint16)
    merged3l = np.array(merged3l, dtype=np.uint16)
    merged3r = np.array(merged3r, dtype=np.uint16)
    
    combined = np.array([merged0l, merged0r, merged1l, merged1r, merged2l, merged2r, merged3l, merged3r, Y])
    np.random.shuffle(combined.T)
    merged0l, merged0r, merged1l, merged1r, merged2l, merged2r, merged3l, merged3r, Y0 = combined[0], combined[1], combined[2], combined[3], combined[4], combined[5], combined[6], combined[7], combined[8]
    
    X = convert_to_binary([merged0l, merged0r, merged1l, merged1r, merged2l, merged2r, merged3l, merged3r])
    return (X, Y0)

