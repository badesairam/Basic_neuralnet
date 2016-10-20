import numpy as np
import math
# #need to load data
# #set regularization parameter
       
import csv
data_list = []
data_input = []
data_output = []

probs = []

#creating inp and op
with open('Train.csv') as data_file:
	for data_line in data_file:
		data_list.append(data_line.strip().split(','))	
for i in range(len(data_list)):
	tmp = []
	for j in range(len(data_list[i])-1):
		tmp.append(data_list[i][j])
	tmp.append(1)
	tmp = np.array(tmp)
	tmp = tmp.astype(np.float)
	data_input.append(tmp)
	data_output.append(data_list[i][len(data_list[i])-1])

num_traindata = len(data_list)
inp_dim = len(data_input[0])
out_dim = 1
reg_lambda = 0.01
epsilon = 1

def train_model(num_hidden_nodes,num_rounds):
	np.random.seed(0)
	W_array = []
	W1 = np.zeros((inp_dim,num_hidden_nodes)) / np.sqrt(inp_dim)
	W2 = np.random.randn(num_hidden_nodes, out_dim) / np.sqrt(num_hidden_nodes)
	for i in xrange(0, num_rounds):
		data_inp = np.array(data_input)
		S1 = data_inp.dot(W1)
		Z1 = 1 / (1+np.exp(-S1))
		Z_aft = 1-Z1
		F1 = Z1*Z_aft
		S2 = Z1.dot(W2)
		out = 1 / (1+np.exp(-S2))
		out = np.round(out)

		# Backpropagation
		data_outp = np.array(data_output)
		data_outp1 = data_outp.astype(np.float)
		data_outp2 = data_outp1.reshape(len(data_outp1),1)
		probs = out
		delta2 = out - data_outp2
		delt = delta2.dot(W2.T)
		delta1 = F1*(delt)
		dW2 = (Z1.T).dot(delta2)
		dW1 = np.dot(data_inp.T, delta1)
		dW2 += reg_lambda*W2
		dW1 += reg_lambda*W1
		W1 += -epsilon*dW1
		W2 += -epsilon*dW2
	W_array.append(W1)
	W_array.append(W2)
	return W_array

def return_output(num_hidden_nodes,rounds):
	W_array = train_model(num_hidden_nodes,rounds)
	test_data = []
	with open('TestX.csv') as data_file:
		for data_line in data_file:
			test_data.append(data_line.strip().split(','))
	csvwriter = csv.writer(open("130050079.csv", "wb"))
	csvwriter.writerow(["Id","Label"])
	for row in range(len(test_data)):
		test_data[row].append(1)
		data_in = np.array(test_data[row])
		data_in = data_in.astype(np.float)
		S1 = data_in.dot(W_array[0])
		Z1 = 1 / (1+np.exp(-S1))
		S2 = Z1.dot(W_array[1])
		out = 1 / (1+np.exp(-S2))
		if out[0]>=1:
			output = 1
		else:
			output = 0
		csvwriter.writerow([row,output])

return_output(25,20000)
