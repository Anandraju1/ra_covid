from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2

app = Flask(__name__)

dic = {0: 'invalid', 1: 'negative',2:'positive'}

model = load_model('model2.h5')

model.make_predict_function()

def predict_label(img_path):
	resized = image.load_img(img_path, target_size=(100,100))
	resized = image.img_to_array(resized)/255.0
	new_image = np.zeros(resized.shape, resized.dtype)
	alpha = 1.0  # Simple contrast control
	beta = 0  # Simple brightness control
	for y in range(resized.shape[0]):
		for x in range(resized.shape[1]):
			for c in range(resized.shape[2]):
				resized[y, x, c] = np.clip(alpha * resized[y, x, c] + beta, 0, 255)
	i = resized.reshape(1, 100,100,3)
	p = model.predict(i)
	p1 = model.predict_classes(i)
	print(p[0][0],max(p[0]))
	g = max(p[0])
	if g>0.6:
		return dic[p1[0]]
	else:
		return 'Invalid'

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "hi"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
