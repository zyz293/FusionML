import numpy as np 
import pickle
import csv
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def cross_validation_score(X_train, y_train, X_test, y_test):
	pca = PCA(0.95)
	pca.fit(X_train)
	X_train = pca.transform(X_train)
	X_test = pca.transform(X_test)
	clf = GradientBoostingRegressor()
	clf.fit(X_train, y_train)
	prediction = clf.predict(X_test)
	true_list = np.asarray(y_test)
	prediction = np.asarray(prediction)
	return np.mean(np.absolute(prediction.flatten()-true_list.flatten())/true_list.flatten())*100, pca.n_components_

def generate_missing(total_data_number, missing_data_rate, missing_feature_number):
	missing_data_idx = np.random.choice(range(total_data_number), size=int(total_data_number*missing_data_rate), replace=False)
	missing_feature_idx = []
	for _ in range(int(total_data_number*missing_data_rate)):
		missing_feature_idx.append(np.random.choice(range(5), size=missing_feature_number, replace=False))
	return missing_data_idx, missing_feature_idx

def generate_missing_v2(total_data_number, missing_data_rate, missing_feature_number_list):
	missing_data_idx = np.random.choice(range(total_data_number), size=int(total_data_number*missing_data_rate), replace=False)
	missing_feature_idx = []
	for _ in range(int(total_data_number*missing_data_rate)):
		missing_feature_number = np.random.choice(missing_feature_number_list)
		missing_feature_idx.append(np.random.choice(range(5), size=missing_feature_number, replace=False))
	return missing_data_idx, missing_feature_idx

def fusion(X, Y, missing_data_idx, missing_feature_idx):
	data = []
	label = []
	for i, dd in enumerate(X):
		temp = np.zeros([1,	dd.shape[1]])
		count = 0.
		if i in missing_data_idx:
			idx = missing_data_idx.index(i)
			for j, d in enumerate(dd):
				if j not in missing_feature_idx[idx]:
					count += 1
					temp += d
		else:
			for j, d in enumerate(dd):
				count += 1
				temp += d
		temp /= count
		temp = temp.flatten()
		data.append(temp)
		label.append(Y[i])
	return np.asarray(data), np.asarray(label)

def fusion_test(X, Y):
	data = []
	label = []
	for i, dd in enumerate(X):
		temp = np.zeros([1,	dd.shape[1]])
		count = 0.
		for j, d in enumerate(dd):
			count += 1
			temp += d
		temp /= count
		temp = temp.flatten()
		data.append(temp)
		label.append(Y[i])
	return np.asarray(data), np.asarray(label)

def remove_missing(X, Y, missing_data_idx, missing_feature_idx):
	data = []
	label = []
	for i in range(len(X)):
		if i not in missing_data_idx:
			data.append(X[i].flatten())
			label.append(Y[i])
	return np.asarray(data), np.asarray(label)

def remove_missing_test(X, Y):
	data = []
	label = []
	for i in range(len(X)):
		data.append(X[i].flatten())
		label.append(Y[i])
	return np.asarray(data), np.asarray(label)

# for sphere function dataset
num_repeat = 10
with open('data.pkl', 'rb') as f:
	data = pickle.load(f)
feature = data['feature']
label = data['label']
test_feature = feature[-200:]
test_label = label[-200:]
feature = feature[:-200]
label = label[:-200]
print (feature.shape, test_feature.shape)

missing_data_rate_list = np.asarray(range(1, 100))/100.
missing_feature_number_list = [1, 2, 3, 4]

feature_flatten = feature.reshape((len(feature), -1))
test_feature_flatten = test_feature.reshape((len(test_feature), -1))
ss = StandardScaler()
ss.fit(feature_flatten)
feature_flatten = ss.transform(feature_flatten)
test_feature_flatten = ss.transform(test_feature_flatten)
print (feature_flatten.shape, test_feature_flatten.shape)
logging = []
res, num = cross_validation_score(feature_flatten, label, test_feature_flatten, test_label)
a = ['nofusion', 0, 0, res, num, len(feature_flatten)]
print (a)
logging.append(a)
logging.append(['**', '**', '**', '**', '**', '**'])

for missing_data_rate in missing_data_rate_list:
	for missing_feature_number in missing_feature_number_list:
		temp = 0.
		for _ in range(num_repeat):
			missing_data_idx, missing_feature_idx = generate_missing(len(feature), missing_data_rate, missing_feature_number)
			fused_data, fused_label = fusion(feature, label, list(missing_data_idx), missing_feature_idx)
			test_fused_data, test_fused_label = fusion_test(test_feature, test_label)
			ss = StandardScaler()
			ss.fit(fused_data)
			fused_data = ss.transform(fused_data)
			test_fused_data = ss.transform(test_fused_data)
			res, num = cross_validation_score(fused_data, fused_label, test_fused_data, test_fused_label)
			temp += res
		res = temp/num_repeat
		a = ['fusion', missing_data_rate, missing_feature_number, res, num, len(fused_data)]
		print (a)
		logging.append(a)

		temp = 0.
		for _ in range(num_repeat):
			remain_data, remain_label = remove_missing(feature, label, missing_data_idx, missing_feature_idx)
			test_remain_data, test_remain_label = remove_missing_test(test_feature, test_label)
			ss = StandardScaler()
			ss.fit(remain_data)
			remain_data = ss.transform(remain_data)
			test_remain_data = ss.transform(test_remain_data)
			res, num = cross_validation_score(remain_data, remain_label, test_remain_data, test_remain_label)
			temp += res
		res = temp/num_repeat
		a = ['no fusion', missing_data_rate, missing_feature_number, res, num, len(remain_data)]
		print (a)
		logging.append(a)

logging.append(['**', '**', '**', '**', '**', '**'])

for missing_data_rate in missing_data_rate_list:
	temp = 0.
	for _ in range(num_repeat):
		missing_data_idx, missing_feature_idx = generate_missing_v2(len(feature), missing_data_rate, missing_feature_number_list)
		fused_data, fused_label = fusion(feature, label, list(missing_data_idx), missing_feature_idx)
		test_fused_data, test_fused_label = fusion_test(test_feature, test_label)
		ss = StandardScaler()
		ss.fit(fused_data)
		fused_data = ss.transform(fused_data)
		test_fused_data = ss.transform(test_fused_data)
		res, num = cross_validation_score(fused_data, fused_label, test_fused_data, test_fused_label)
		temp += res
	res = temp/num_repeat
	a = ['fusion', missing_data_rate, missing_feature_number, res, num, len(fused_data)]
	print (a)
	logging.append(a)

	temp = 0.
	for _ in range(num_repeat):
		remain_data, remain_label = remove_missing(feature, label, missing_data_idx, missing_feature_idx)
		test_remain_data, test_remain_label = remove_missing_test(test_feature, test_label)
		ss = StandardScaler()
		ss.fit(remain_data)
		remain_data = ss.transform(remain_data)
		test_remain_data = ss.transform(test_remain_data)
		res, num = cross_validation_score(remain_data, remain_label, test_remain_data, test_remain_label)
		temp += res
	res = temp/num_repeat
	a = ['no fusion', missing_data_rate, missing_feature_number, res, num, len(remain_data)]
	print (a)
	logging.append(a)	

with open('results.csv', 'w') as f:
	writer = csv.writer(f, delimiter=',')
	writer.writerow(['if fusion', 'missing_data_rate', 'missing_feature_number', 'MAE', '#_pca', '#_data'])
	for i in range(len(logging)):
		writer.writerow([logging[i][0], logging[i][1], logging[i][2], logging[i][3], logging[i][4], logging[i][5]])


