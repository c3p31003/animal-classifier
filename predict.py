from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np
from keras import optimizers
from PIL import Image
import sys

classes = ['monkey', 'boar', 'crow']
num_classes = len(classes)

#画像のサイズを縦横ともに50ピクセルにする
image_size = 50

def build_model():
        #モデルを作成
    model = Sequential()
    
    #addでニューラルネットワークの層を足す
    # input_shapeの修正：X.shape[1:]を使用
    model.add(Conv2D(32, (3,3), padding='same', input_shape=(50,50,3)))
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
    

    
    #モデルの保存
    model = load_model('./animal_cnn_aug.h5')
    print("モデルを保存しました: ./animal_cnn.h5")
    
    return model


def main():
    image = Image.open(sys.argv[1])
    image = image.convert('RGB') 
    image = image.resize((image_size, image_size))
    data = np.asarray(image)
    X = []
    X.append(data)
    X = np.array([data])
    X = X.astype('float32') / 255.0 
    model = build_model()
    
    result = model.predict([X])[0]  
    predicted = result.argmax()
    parcent = int(result[predicted] * 100)
    print("予測結果:" + classes[predicted] + str(parcent) + "%")
    

if __name__ == "__main__":
    main()
    
