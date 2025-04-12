import random

import numpy as np
from os import urandom

def addRoundKey( state , key_round ):
	return state ^ key_round 

mask = 0b1111
sBox = [12,5,6,11,9,0,10,13,3,14,15,8,4,7,1,2]

def sBoxLayer(state):
	string_state = bin(state)[2:].zfill(64)
	newstring = ""
	for i in range(0, len(string_state),4):
		now_str = string_state[i:i+4]
		c = int(now_str,2)
		newstring += bin(sBox[c])[2:].zfill(4)
	newstate = int(newstring,2)
	return newstate


pLayerTable = [ 0,16,32,48, 1,17,33,49, 2,18,34,50, 3,19,35,51,
			    4,20,36,52, 5,21,37,53, 6,22,38,54, 7,23,39,55,
			    8,24,40,56, 9,25,41,57,10,26,42,58,11,27,43,59,
			   12,28,44,60,13,29,45,61,14,30,46,62,15,31,47,63]

def pLayer(state):
	string_state = bin(state)[2:].zfill(64)
	char_state = list(string_state)
	for i in range(0,len(string_state)):
		char_state[pLayerTable[i]] = string_state[i]
	string_state = "".join(char_state)
	new_state = int(string_state, 2)
	return new_state

def string_sbox(nibble):
	return bin(sBox[int(nibble, 2)])[2:].zfill(4)
def string_xor_counter(almost_nibble , counter):
	return bin(int(almost_nibble, 2) ^ counter)[2:].zfill(5)

#TODO do better without string only bitwise operations
def generateRoundKeys( key, round ):
	K = []
	string_key = key
	K.append(int(string_key[:64], 2))
	for i in range(1, round+1):
		string_key = string_key[61:] + string_key[:61]
		string_key = string_sbox ( string_key[:4]) + string_key[4:]
		string_key = string_key[:60] + string_xor_counter(string_key[60:65], i) + string_key[65:]
		K.append( int(string_key[:64], 2))
	return K

def expand_key(k1, k2, round):
	keys_list = []
	for i in range(len(k1)):
		k11 = '{:064b}'.format(k1[i])
		k12 = '{:016b}'.format(k2[i])
		key = k11 + k12
		rk_list = generateRoundKeys(key, round)
		keys_list.append(rk_list)
	return keys_list



def present_cipher(state, key, round):
	state = int(state)
	for i in range(0, round):
		state = state ^ key[i]
		state = sBoxLayer(state)
		state = pLayer(state)
	state = state ^ key[round]
	return state

def encrypt_present(plaintexts, keys_list, round):
	ciphers = []
	i = 0
	for state in plaintexts:
		ciphertext = present_cipher(state, keys_list[i], round)
		i += 1
		ciphers.append(ciphertext)
	return ciphers

def convert_to_binary(arr):
  X = np.zeros((4 * 64, len(arr[0])),dtype=np.uint8);
  for i in range(4 * 64):
    index = i // 64;
    offset = 64 - (i % 64) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);

def make_train_data(n, nr, diff=0x0000000000d00000):

	k1 = np.frombuffer(urandom(8*n),dtype=np.uint64)
	k2 = np.frombuffer(urandom(2*n),dtype=np.uint16)#随即产生n个字节的字符串，可以作为随机加密key使用~
	pt0 = np.frombuffer(urandom(8*n),dtype=np.uint64)
	pt1 = pt0 ^ diff
	pt2 = np.frombuffer(urandom(8*n),dtype=np.uint64)
	pt3 = pt2 ^ diff
	keys_list = expand_key(k1, k2, nr)
	
	x = nr // 2
	kr = []
	for i in range(n):
		kr.append(keys_list[i][:(x + 1)])
	# print(kr)
	ct0 = encrypt_present(pt0, kr, x)
	ct1 = encrypt_present(pt1, kr, x)
	ct2 = encrypt_present(pt2, kr, x)
	ct3 = encrypt_present(pt3, kr, x)
	
	joined_elements0 = []
	joined_elements1 = []
	joined_elements2 = []
	joined_elements3 = []
	ks = []
	
	for i in range(n):
		if(ct0[i] ^ ct1[i] == ct2[i] ^ ct3[i]):
			
			joined_elements0.append(ct0[i])
			joined_elements1.append(ct1[i])
			joined_elements2.append(ct2[i])
			joined_elements3.append(ct3[i])
			ks.append(keys_list[i])
			

			
	c0 = []
	c1 = []
	c2 = []
	c3 = []
	for i in range(len(joined_elements0)):
		cq0 = joined_elements0[i] ^ ks[i][x]
		c0.append(cq0)
	for i in range(len(joined_elements1)):
		cq1 = joined_elements1[i] ^ ks[i][x]
		c1.append(cq1)
	for i in range(len(joined_elements2)):
		cq2 = joined_elements2[i] ^ ks[i][x]
		c2.append(cq2)
	for i in range(len(joined_elements3)):
		cq3 = joined_elements3[i] ^ ks[i][x]
		c3.append(cq3)
	
	kt = []
	for i in range(len(ks)):
		kt.append(ks[i][x:])
	ck0 = encrypt_present(c0, kt, nr-x)
	ck1 = encrypt_present(c1, kt, nr-x)
	ck2 = encrypt_present(c2, kt, nr-x)
	ck3 = encrypt_present(c3, kt, nr-x)
	
	Y = np.random.randint(1, 2, 2*len(ck0), dtype=np.uint8)
	Y[:len(ck0)] = 0
	# np.random.shuffle(Y)
	num_rand_samples = np.sum(Y == 0)
	# print(num_rand_samples)
	rp0 = np.frombuffer(urandom(8 * num_rand_samples), dtype=np.uint64)
	rp1 = np.frombuffer(urandom(8 * num_rand_samples), dtype=np.uint64)
	rp2 = np.frombuffer(urandom(8 * num_rand_samples), dtype=np.uint64)
	rp3 = np.frombuffer(urandom(8 * num_rand_samples), dtype=np.uint64)
	kk = []
	for i in range(len(ks)):
		kk.append(ks[i][:(x + 1)])
	rc0 = encrypt_present(rp0, kk, x)
	rc1 = encrypt_present(rp1, kk, x)
	rc2 = encrypt_present(rp2, kk, x)
	rc3 = encrypt_present(rp3, kk, x)
	r0 = []
	r1 = []
	r2 = []
	r3 = []
	for i in range(len(rc0)):
		cr0 = rc0[i] ^ ks[i][x]
		r0.append(cr0)
	for i in range(len(rc1)):
		cr1 = rc1[i] ^ ks[i][x]
		r1.append(cr1)
	for i in range(len(rc2)):
		cr2 = rc2[i] ^ ks[i][x]
		r2.append(cr2)
	for i in range(len(rc3)):
		cr3 = rc3[i] ^ ks[i][x]
		r3.append(cr3)

	rk0 = encrypt_present(r0, kt, nr - x)
	rk1 = encrypt_present(r1, kt, nr - x)
	rk2 = encrypt_present(r2, kt, nr - x)
	rk3 = encrypt_present(r3, kt, nr - x)

	
	merged0 = rk0 + ck0
	merged1 = rk1 + ck1
	merged2 = rk2 + ck2
	merged3 = rk3 + ck3
	
	merged0 = np.array(merged0, dtype=np.uint64)
	
	merged1 = np.array(merged1, dtype=np.uint64)
	merged2 = np.array(merged2, dtype=np.uint64)
	merged3 = np.array(merged3, dtype=np.uint64)
	
	combined = np.array([merged0, merged1, merged2, merged3, Y])
	
	np.random.shuffle(combined.T)
	mergedarray0, mergedarray1, mergedarray2, mergedarray3,Y0 = combined[0], combined[1], combined[2], combined[3],combined[4]
	Num1 = len(mergedarray0)
	X = convert_to_binary([mergedarray0, mergedarray1, mergedarray2, mergedarray3])
	return (X,Y0,Num1)






