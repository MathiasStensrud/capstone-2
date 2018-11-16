from sklearn.ensemble import RandomForestClassifier
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input
from sklearn.model_selection import train_test_split

def fill():
    x=[]
    y=[]
    gen=ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True).flow_from_directory('data/asl_alphabet_test',[64,64],batch_size=100)
    for i in range(len(gen)):
        for e in range(len(gen)):
            x.append(gen[i][0][e])
            y.append(gen[i])
    return np.array(x),np.array(y)


if __name__=='__main__':
    forest=RandomForestClassifier(n_estimators=100)
    x, y=fill()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    forest.fit(x_train,y_train)
    print(forest.score(x_test,y_test))
