from flask import Flask, request, render_template
from keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

classes = ['monkey', 'boar', 'crow']
image_size = 50
UPLOAD_FOLDER = 'static/uploads'
model = load_model('./animal_cnn_aug.h5')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # 画像を読み込み、処理
            image = Image.open(filepath)
            image = image.convert('RGB')
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X = np.array([data])

            result = model.predict([X])[0]
            predicted = result.argmax()
            percent = int(result[predicted] * 100)
            result_text = f"予測結果: {classes[predicted]} {percent}%"

            return render_template('result.html', result=result_text, image_path='/' + filepath)

    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
