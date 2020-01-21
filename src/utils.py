import pandas as pd
import cv2
from os import chdir, pardir
from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf


GENRES = ['Acción', 'Aventura', 'Animada', 'Biográfica', 'Comedia', 'Crimen', 
			'Documental', 'Drama', 'Familiar', 'Fantasía', 'Histórica', 
			'Terror', 'Música', 'Musical', 'Misterio', 'N/A', 
			'Informativa', 'Reality-TV', 'Romántica', 'Ciencia Ficción', 'Corto', 
			'Deportiva', 'Thriller', 'Bélica', 'Western']

def process_prediction(results):
	final = list(zip(results, range(len(results))))
	final.sort(reverse=True)
	final = [(GENRES[y], round(x, 3)) for x, y in final][:3]
	return list(filter(lambda x:x[1] > 0, final))

def predict2(model, path, shape):
	img = image.load_img(path, target_size=(shape[0],shape[1],3))
	img = image.img_to_array(img)
	img /= 255
	results = model.predict(img.reshape(1,shape[0],shape[1],3))[0]            
	results = process_prediction(results)
	results, _ = zip(*results)
	vec = [0 for _ in range(25)]
	for i in range(len(GENRES)):
		if GENRES[i] in results:
			vec[i] = 1

	return vec

def predict_list(model, shape, path_x, path_y):
	df = pd.read_csv(path_y).values

	# ("name", [genres])
	df = [(x[0], x[1:]) for x in df]

	y_train = []
	y_truth = []
	count = 1
	for x, y in df:
		y_train.append(predict2(model, path_x + '/' + x + '.jpg', shape))
		y_truth.append(y)
		print("Processed " + str(count) + " image.")
		count += 1

	return y_train, y_truth

def metrics(model, shape, path_x, path_y):
	y1, y2 = predict_list(model, shape, path_x, path_y)
	
	fp = tf.keras.metrics.FalsePositives()
	fp.update_state(y1,y2)
	
	fn = tf.keras.metrics.FalseNegatives()
	fn.update_state(y1,y2)
	
	tp = tf.keras.metrics.TruePositives()
	tp.update_state(y1,y2)
	
	tn = tf.keras.metrics.TrueNegatives()
	tn.update_state(y1,y2)

	p = tf.keras.metrics.Precision()
	p.update_state(y1,y2)
	
	r = tf.keras.metrics.Recall()
	r.update_state(y1,y2)

	return  {
		'false_positive':int(fp.result().numpy()),
		'false_negative':int(fn.result().numpy()),
		'true_positive':int(tp.result().numpy()),
		'true_negative':int(tn.result().numpy()),
		'precision':float(p.result().numpy()),
		'recall':float(r.result().numpy())
	}
