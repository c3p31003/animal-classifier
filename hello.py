from flask import Flask

#Flaskクラスのインスタンス、__name__は、ほとんどの場合に適した便利なショートカット
#Pythonは、自動で __name__ に __main__ または モジュール名 を代入してくれる
#__name__ に入る値が"__main__":「このファイルがメインとして実行された」
#__name__ に入る値が"モジュール名"（例：hello）:「他のファイルから呼び出された」
app = Flask(__name__)
# print(__name__)

#ルートurl　/　の後ろになにも続かない場合のみhello_world関数が呼び出される。
@app.route("/")
def hello_world():
    return '<p>Hello, World!<p>'