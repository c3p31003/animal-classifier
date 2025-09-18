from PIL import Image
import os,glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.model_selection import KFold

classes = ['monkey', 'boar', 'crow']
num_classes = len(classes)

#画像のサイズを縦横ともに50ピクセルにする
image_size = 50


#画像の読み込み

#画像データを格納
X = []
#ラベルデータを格納
Y = []

for index, classlabel in enumerate(classes):
    #カレントディレクトリの後ろにクラス名を追加
    photos_dir = './' + classlabel
    files = glob.glob(photos_dir + '/*.jpg')
    
    #一覧を取得した後、その中のデータを順番に取り出していく。
    for i,file in enumerate(files):
        if i >= 200:
            break
        image = Image.open(file)
        #imageをRGBの三色に変換
        image = image.convert("RGB")
        #imageのサイズをそろえる
        image = image.resize((image_size, image_size))
        #imageのデータを数字の配列にして、変数に格納
        data = np.asarray(image)
        #appendで配列の最後尾に追加
        X.append(data)
        Y.append(index)
        
#リスト型から、TensorFlowが扱いやすいnumpy配列にする
X = np.array(X)
Y = np.array(Y)


x_train,x_test,y_train,y_test = model_selection.train_test_split(X, Y)
xy = (x_train,x_test,y_train,y_test)
np.savez('./animal_data', x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)



