from keras.models import Sequential
from keras.layers import Dense, Flatten,Dropout
from keras.datasets import cifar10
from keras.utils import to_categorical

(X_train,y_train),(X_test,y_test)=cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#Step 3: Build MLP Model
model = Sequential()
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

#Step 4: Compile & Train
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=2,validation_split=0.1)

#Step 5: Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")