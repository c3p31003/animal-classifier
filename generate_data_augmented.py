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
num_testdata = 100

#画像の読み込み

#画像データを格納
X_train = []
X_test = []
#ラベルデータを格納
Y_train = []
Y_test = []

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
        
        if i< num_testdata:
            #appendで配列の最後尾に追加
            X_test.append(data)
            Y_test.append(index)
        else:
            X_train.append(data)
            Y_train.append(index)
            
            #angleは、-20~20まで5度刻みで回転をしていく。
            for angle in range(-20,20,5):
                #反転
                img_r = image.rotate(angle)
                data = np.asarray(img_r) 
                X_train.append(data)
                Y_train.append(index)
                
                #反転
                img_trans = image.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)
#リスト型から、TensorFlowが扱いやすいnumpy配列にする
x_train = np.array(X_train)
x_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)



# x_train,x_test,y_train,y_test = model_selection.train_test_split(X, Y)
xy = (x_train,x_test,y_train,y_test)
np.savez('./animal_aug', x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)



