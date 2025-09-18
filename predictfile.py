import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = './uploads'

#uploadできるファイルの拡張子を宣言
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#ファイルのアップロード可否判定関数
def allowed_file(filename):
    #.rsplit('.', 1) は、右から1回だけ.（ドット）で分割する
    #結果はリストに格納。例)['photo', 'PNG']
    #index 1 を指定することで拡張子部分のみ取り出すことが出来る。
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            #ファイルをuploadするページに戻す
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            #ファイルをuploadするページに戻す
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename = filename))
    return '''
<!doctype html>
<html>
<head>
<meta charset = "UTF-8">
<title>ファイルをアップロードして判定しよう</title>
</head>
<body>
<h1>ファイルをアップロードして判定しよう</h1>
<form method=post enctype=multipart/form-data>
    <input type=file name=file>
    <input type=submit value=アップロード>
</form>
</body>
</html>
'''
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)