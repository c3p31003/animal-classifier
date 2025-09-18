#Sequentialはニューラルネットワークのモデルを定義する際に使用する
from keras.models import Sequential
#畳み込みやpoolingなどの処理をするための関数を読み込む
from keras.layers import Conv2D, MaxPooling2D
#活性化関数とDropout処理を行う関数、データを一次元に変換するためのFlatten、レイヤーとして使用するDenseを読み込む
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np
from keras import optimizers

classes = ['monkey', 'boar', 'crow']
num_classes = len(classes)

#画像のサイズを縦横ともに50ピクセルにする
image_size = 50

#データを読み込んでトレーニングを行うメインの関数を定義する
def main():
    # .npzファイルの正しい読み込み方法
    data = np.load("./animal_aug.npz", allow_pickle=True)
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"データ形状確認:")
    print(f"x_train: {x_train.shape}, type: {type(x_train)}")
    print(f"x_test: {x_test.shape}, type: {type(x_test)}")
    print(f"y_train: {y_train.shape}, type: {type(y_train)}")
    print(f"y_test: {y_test.shape}, type: {type(y_test)}")
    
    #データの正規化をする 0~255の値を最大値で割って、0~1に収束する
    x_train = x_train.astype("float32") / 255.0  # 256ではなく255、float32を推奨
    x_test = x_test.astype("float32") / 255.0
    
    #one-hot-vector()正解は1,他は0を返す
    #monkey [1,0,0]
    #boar [0,1,0]
    #crow[0,0,1]
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    
    model = model_train(x_train, y_train)
    #モデルの評価を行う
    model_eval(model, x_test, y_test)

# model_train関数をmain関数の外に移動（インデントの修正）
def model_train(X, y):
    #モデルを作成
    model = Sequential()
    
    #addでニューラルネットワークの層を足す
    # input_shapeの修正：X.shape[1:]を使用
    model.add(Conv2D(32, (3,3), padding='same', input_shape=X.shape[1:]))
    # 'lelu'は存在しない → 'relu'に修正
    model.add(Activation('relu'))
    
    #2層目の畳み込み
    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    #25%を捨てる
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    #最後に全結合する
    
    #データを一列に並べるFlatten処理を行う
    model.add(Flatten())
    #そのデータをDenseで全結合処理をする
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))  # 3ではなくnum_classesを使用
    model.add(Activation('softmax'))
    
    #トレーニング時の最適化アルゴリズムを使用して最適化の処理を行う 
    opt = optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    
    #loss:損失関数　正解と推定値との誤差
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    #トレーニング処理
    #epoch数を増やすと精度が上がるが時間がかかる
    # nb_epochは古い書き方 → epochsに修正
    model.fit(X, y, batch_size=32, epochs=100, validation_split=0.1)
    
    #モデルの保存
    model.save('./animal_cnn_aug.h5')
    print("モデルを保存しました: ./animal_cnn.h5")
    
    return model

def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose=1)
    print('Test Loss:', scores[0])
    print('Test Accuracy:', scores[1])
    
if __name__ == "__main__":
    main()