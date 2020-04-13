import numpy as np 

def Calculate(user_input):
	string_=user_input.split('/')
	try:
		ret_value=int(string_[0])/int(string_[1])
	except:
		return int(string_[0])
	else:
		return ret_value


def sigmoid(x,deriv=False):
	if(deriv==True):
		return x*(1-x)

	return 1/(1+np.exp(-x))

X=np.array([[1,2,1,2,1,2,1],
			[1,2,1,2,1,1,0],
			[1,2,1,1,1,2,0],
			[1,1,1,2,1,2,0],
			[1,2,-1,2,1,1,0],
			[1,2,1,1,-1,2,0],
			[1,1,-1,2,1,2,0],
			[-1,2,1,2,1,1,0],
			[-1,2,1,1,1,2,0],
			[1,1,1,2,-1,2,0],
			[1,2,1,2,1,2,0],
			[1,2,1,2,1,2,0],
			[1,2,1,2,1,2,0],
			[1,2,1,2,-1,2,1],
			[1,2,-1,2,1,2,1],
			[-1,2,1,2,1,2,1],
			[1,2,1,2,-1,2,-1],
			[1,2,-1,2,1,2,-1],
			[-1,2,1,2,1,2,-1],
			[1,2,1,2,0,0,1],
			[1,2,0,0,1,2,1],
			[0,0,1,2,1,2,1],
			[1,2,-1,2,0,0,1],
			[1,2,0,0,-1,2,1],
			[0,0,1,2,-1,2,1],
			[-1,2,1,2,0,0,1],
			[-1,2,0,0,1,2,1],
			[0,0,-1,2,1,2,1]])

y=np.array([[1,0,0,0,0,0,0,0],
			[0,1,0,0,0,0,0,0],
			[0,1,0,0,0,0,0,0],
			[0,1,0,0,0,0,0,0],
			[0,0,1,0,0,0,0,0],
			[0,0,1,0,0,0,0,0],
			[0,0,1,0,0,0,0,0],
			[0,0,1,0,0,0,0,0],
			[0,0,1,0,0,0,0,0],
			[0,0,1,0,0,0,0,0],
			[0,0,0,1,0,0,0,0],
			[0,0,0,1,0,0,0,0],
			[0,0,0,1,0,0,0,0],
			[0,0,0,0,1,0,0,0],
			[0,0,0,0,1,0,0,0],
			[0,0,0,0,1,0,0,0],
			[0,0,0,0,0,1,0,0],
			[0,0,0,0,0,1,0,0],
			[0,0,0,0,0,1,0,0],
			[0,0,0,0,0,0,1,0],
			[0,0,0,0,0,0,1,0],
			[0,0,0,0,0,0,1,0],
			[0,0,0,0,0,0,0,1],
			[0,0,0,0,0,0,0,1],
			[0,0,0,0,0,0,0,1],
			[0,0,0,0,0,0,0,1],
			[0,0,0,0,0,0,0,1],
			[0,0,0,0,0,0,0,1],
			])

np.random.seed(420)
syn0=2*np.random.random((7,8)) - 1
syn1=2*np.random.random((8,8))- 1
syn2=2*np.random.random((8,8)) - 1

#Training

for j in range(100000):
	l0 = X
	l1 = sigmoid(np.dot(l0,syn0))
	l2 = sigmoid(np.dot(l1,syn1))
	l3 = sigmoid(np.dot(l2,syn2))

	#back prop

	l3_error = y -l3
	if(j%10000)==0:
		print("Error: "+str(np.mean(np.abs(l3_error))))

	l3_delta = l3_error*sigmoid(l3,deriv=True)

	l2_error = l3_delta.dot(syn2.T)

	l2_delta = l2_error*sigmoid(l2,deriv=True)

	l1_error = l2_delta.dot(syn1.T)

	l1_delta = l1_error*sigmoid(l1,deriv=True)

	syn2+= l2.T.dot(l3_delta)
	syn1+= l1.T.dot(l2_delta)
	syn0+= l0.T.dot(l1_delta)

Variables=['a','x power','b','y power','c','z power','constant']
Var_=[]
for i in range(len(Variables)):
	Lock=False
	while(Lock==False):
		user_input=input("Give a value for "+Variables[i]+":\n")
		for number in user_input:
			try:
				number=int(number)
			except:
				Lock=False
			else:
				Lock=True

	if(Variables[i]=='a' or Variables[i]=='b'or Variables[i]=='c' or Variables[i]=='constant'):
		if(Calculate(user_input)>0):
			user_input=1;
			Var_.append(user_input)
		elif(Calculate(user_input)<0):
			user_input=-1;
			Var_.append(user_input)
		elif(Calculate(user_input)==0):
			user_input=0;
			Var_.append(user_input)
	else:
		Var_.append(int(user_input))
	

def predict(U_array):
	print(U_array)
	pred_0 = np.array([[U_array]])
	pred_1 = sigmoid(np.dot(pred_0,syn0))
	pred_2 = sigmoid(np.dot(pred_1,syn1))
	pred_3 = sigmoid(np.dot(pred_2,syn2))
	predict= np.round(pred_3)
	
	case_1=np.array([[[1,0,0,0,0,0,0,0]]])
	case_2=np.array([[[0,1,0,0,0,0,0,0]]])
	case_3=np.array([[[0,0,1,0,0,0,0,0]]])
	case_4=np.array([[[0,0,0,1,0,0,0,0]]])
	case_5=np.array([[[0,0,0,0,1,0,0,0]]])
	case_6=np.array([[[0,0,0,0,0,1,0,0]]])
	case_7=np.array([[[1,0,0,0,0,0,1,0]]])
	case_8=np.array([[[1,0,0,0,0,0,0,1]]])
	case_911=np.array([1,9,1,9,1,9,1])

	if(np.array_equal(predict,case_1)):
		print("Elleipsoeides")
	elif(np.array_equal(predict,case_2)):
		print("Elleiptiko Paravoloeides")
	elif(np.array_equal(predict,case_3)):
		print("Ypervoliko Paravoloeides")
	elif(np.array_equal(predict,case_4)):
		print("Elleiptikos Kwnos")
	elif(np.array_equal(predict,case_5)):
		print("Monoxwno Ypervoloeides")
	elif(np.array_equal(predict,case_6)):
		print("Dixwna Ypervoloeides")
	elif(np.array_equal(predict,case_7)):
		print("Elleiptikos Kulindros")
	elif(np.array_equal(predict,case_8)):
		print("Ypervolikos Kulindros")
	if(np.array_equal(U_array,case_911)):
		print("Bush did 9/11")
	


predict(Var_)