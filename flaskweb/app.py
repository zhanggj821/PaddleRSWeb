import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import paddlers as pdrs

UPLOAD_FOLDER = 'static/Uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf','jpg'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):#通过将文件名分段的方式查询文件格式是否在允许上传格式范围之内
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/up_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        sensebefor = request.files['sensingbefor']
        senseafter = request.files['sensingafter']
        # if sensebefor and allowed_file(sensebefor.filename):
        #     sensebefor.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(sensebefor.filename)))
        # if senseafter and allowed_file(senseafter.filename):
        #     sensebefor.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(sensebefor.filename)))
        # senseafter.save(secure_filename(senseafter.filename))
        predictor = pdrs.deploy.Predictor('./inference_model')
        result = predictor.predict(img_file=(sensebefor, senseafter))
        # predictor.predict()
        # return result
        return render_template('index.html', data = result.shape)

if __name__ == '__main__':
	app.run(debug=True)
