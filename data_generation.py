import numpy as np 
import pickle
from keras.applications.vgg16 import VGG16
from keras.layers import Input
import numpy as np 
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

# generate data
dd = np.random.uniform(-5.12, 5.12, size=(1200, 32*32))
label = []
data = []
for d in dd:
	visited = set()
	for idx in range(5):
		temp = d
		while tuple(temp) in visited:
			sign = np.random.choice([-1,1], size=len(d))
			temp = d*sign
		visited.add(tuple(temp))
		s = 0
		for i, n in enumerate(temp):
			s += n**2
		if idx == 0:
			label.append(s)
		scaler = MinMaxScaler((0, 255))
		temp = temp.reshape((-1,1))
		temp = scaler.fit_transform(temp)
		data.append(temp)
data = np.asarray(data)
label = np.asarray(label)
print (data.shape, np.min(data), np.max(data))
print (label.shape, np.min(label), np.max(label))
print (len(np.unique(label)))

# extract feature with vgg16
feature = data.reshape((6000, 32, 32))
feature = np.expand_dims(feature, axis=3)
feature = np.repeat(feature, 3, axis=3)
print (feature.shape)

layer_name = 'block5_pool'
input_tensor = Input(shape=(32,32,3))
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
model.summary()

feature = model.predict(feature)
feature = np.mean(feature, axis=(1,2))
print (feature.shape)

final_feature = []
for i in range(0, len(feature), 5):
	final_feature.append(feature[i:i+5])
final_feature = np.asarray(final_feature)
print (final_feature.shape)

dic = {'data': data, 'feature':final_feature, 'label': label}
with open('data.pkl','wb') as f:
	pickle.dump(dic, f)


