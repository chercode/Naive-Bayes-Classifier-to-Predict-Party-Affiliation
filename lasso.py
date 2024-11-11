import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV


# load data
dpa = pd.read_csv('./data/house-votes-84.complete.csv')
dpa['Class'] = dpa['Class'].map({'republican': 0, 'democrat': 1})
for i in range(16):
	index = 'A'+ str(i+1)
	dpa[index] = dpa[index].map({'y': 1, 'n': 0})
#dpa.info()

pay = dpa.Class
paX = dpa.drop('Class', axis = 1)



'''
  10-cv with house-votes-84.complete.csv using LASSO
  - train_subset: train the classifier on a smaller subset of the training
    data
  -subset_size: the size of subset when train_subset is true
  NOTE you do *not* need to modify this function
  '''
def lasso_evaluate(train_subset=False, subset_size = 0):
	sample_size = pay.shape[0]
	tot_incorrect=0
	tot_test=0
	tot_train_incorrect=0
	tot_train=0
	step = int( sample_size/ 10 + 1)
	for holdout_round, i in enumerate(range(0, sample_size, step)):
		#print("CV round: %s." % (holdout_round + 1))
		if(i==0):
			X_train = paX.iloc[i+step:sample_size]
			y_train = pay.iloc[i+step:sample_size]
		else:
			X_train =paX.iloc[0:i]
			X_train = X_train.append(paX.iloc[i+step:sample_size], ignore_index=True)
			y_train = pay.iloc[0:i]
			y_train = y_train.append(pay.iloc[i+step:sample_size], ignore_index=True)
		X_test = paX.iloc[i: i+step]
		y_test = pay.iloc[i: i+step]
		if(train_subset):
			X_train = X_train.iloc[0:subset_size]
			y_train = y_train.iloc[0:subset_size]
		#print(" Samples={} test = {}".format(y_train.shape[0],y_test.shape[0]))
		# train the classifiers
		lasso = Lasso(alpha = 0.001)
		lasso.fit(X_train, y_train)
		lasso_predit = lasso.predict(X_test)           # Use this model to predict the test data
		lasso_result = [1 if x>0.5 else 0 for x in lasso_predit]
		error = 0
		for (index, num) in enumerate(lasso_result):
			if(y_test.values.tolist()[index] != num):
				error+=1
		tot_train_incorrect+= error
		tot_train += len(lasso_result)
		#print('Error rate {}'.format(1.0*error/len(lasso_result)))
		lasso_predit = lasso.predict(X_train)           # Use this model to get the training error
		lasso_result = [1 if x>0.5 else 0 for x in lasso_predit]
		error = 0
		for (index, num) in enumerate(lasso_result):
			if(y_train.values.tolist()[index] != num):
				error+=1
		#print('Train Error rate {}'.format(1.0*error/len(lasso_result)))
		tot_incorrect += error
		tot_test += len(lasso_result)
	#print('10CV Error rate {}'.format(1.0*tot_incorrect/tot_test))
	#print('10CV train Error rate {}'.format(1.0*tot_train_incorrect/tot_train))

	return 1.0*tot_incorrect/tot_test, 1.0*tot_train_incorrect/tot_train

'''
 For Q5
 Run LASSO, using the commonly-used strategy of replacing unknown values with 0 (“no)
 We have modified the csv file for this purpose.
  NOTE you do *not* need to modify this function
  '''

def lasso_evaluate_incomplete_entry():
	# get incomplete data
	dpc = pd.read_csv('./data/house-votes-84.incomplete.csv')
	for i in range(16):
		index = 'A'+ str(i+1)
		dpc[index] = dpc[index].map({'y': 1, 'n': 0})
	lasso = Lasso(alpha = 0.001)
	lasso.fit(paX, pay)
	lasso_predit = lasso.predict(dpc)
	print(lasso_predit)

def plot(subset_sizes, train_errors, test_errors):

    plt.plot(subset_sizes, test_errors, label = "LASSO Test Error")
    plt.plot(subset_sizes, train_errors, label = "LASSO Train Error")
    plt.title("Classifier Error Rates")
    plt.xlabel("Sample Size")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

def main():
	'''
	TODO modify or use the following code to evaluate your implemented
	classifiers
	Suggestions on how to use the starter code for Q2, Q3, and Q5:

	#For Q2
	'''
	error_rate, unused = lasso_evaluate()
	print('10CV Error rate {}'.format(error_rate))


	#For Q3
	train_error = np.zeros(15)
	test_error = np.zeros(15)
	subset_sizes = list(np.arange(5,10)) + [(i+1)*10 for i in range(10)]
	for i,s in enumerate(subset_sizes):
		x, y =lasso_evaluate(train_subset=True, subset_size=s)
		train_error[i] = x
		test_error[i] = y
	print(train_error)
	print(test_error)
	plot(subset_sizes, train_error, test_error)





	#Q4
	#TODO
	#You may find lasso.coef_ useful
	# load synthetic data
	dpa_synthetic = pd.read_csv('./data/synthetic.csv')
	dpa_synthetic['Class'] = dpa_synthetic['Class'].map({'republican': 0, 'democrat': 1})
	for i in range(16):
		index = 'A' + str(i + 1)
		dpa_synthetic[index] = dpa_synthetic[index].map({'y': 1, 'n': 0})

	nonpartisan_fraction = np.zeros(10)
	subset_size = [i * 400 + 400 for i in range(10)]
	for i in range(10):
		X_train = dpa_synthetic.drop('Class', axis=1).iloc[:subset_size[i]]
		y_train = dpa_synthetic.Class.iloc[:subset_size[i]]
		lasso = Lasso(alpha=0.001)
		lasso.fit(X_train, y_train)
		nonpartisan_count = 0
		for index in range(4, 16):
			if lasso.coef_[index] == 0:
				nonpartisan_count += 1
		nonpartisan_fraction[i] = nonpartisan_count/12
		print(f"nonpartisan fraction {nonpartisan_fraction[i]:.4f} on {(i+1)*400} examples")

	plt.plot(subset_size, nonpartisan_fraction, label="LASSO nonpartisan_fraction")
	plt.legend()
	plt.show()
	

	#Q5
	print('LASSO  P(C= 1|A_observed) ')
	lasso_evaluate_incomplete_entry()

if __name__ == "__main__":
    main()
