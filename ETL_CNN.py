# Japanese Optical Character Recognition Convolutional Neural Network
import tensorflow as tf
import numpy as np

import math
import matplotlib.pyplot as plt

label_names = ["あ","い","う","え","お","か","が","き","ぎ","く","ぐ","け","げ","こ","ご","さ",
"ざ","し","じ","す","ず","せ","ぜ","そ","ぞ","た","だ","ち","ぢ","つ","づ","て","で","と","ど",
"な","に","ぬ","ね","の","は","ば","ぱ","ひ","び","ぴ","ふ","ぶ","ぷ","へ","べ","ぺ","ほ","ぼ",
"ぽ","ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","ん"]

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(label_names
        [predicted_label],
        100*np.max(predictions_array),
        label_names
        [true_label]),
        color=color,
        fontname="MS Gothic") ## just added

#
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(71))
    plt.yticks([])
    thisplot = plt.bar(range(71), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#
## Evalutation
def eval_plot(model, test_dataset):
    test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
    print('\nTest loss:', test_loss, '\nTest accuracy:', test_acc)

    probability_model = tf.keras.Sequential([model, 
                                           tf.keras.layers.Softmax()])
    for test_images, test_labels in test_dataset.take(1):
        test_images = test_images.numpy()
        test_labels = test_labels.numpy()
        predictions = model.predict(test_images)

    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()
    return test_loss, test_acc
#






#-------------------
#
#   DATA HANDLING
#
#-------------------
train_images = np.load("hiragana_train_images.npz")['arr_0']
train_labels = np.load("hiragana_train_labels.npz")['arr_0']
test_images = np.load("hiragana_test_images.npz")['arr_0']
test_labels = np.load("hiragana_test_labels.npz")['arr_0']

if(train_images.shape[0] != train_labels.shape[0]):
    print("!!!WARNING!!! Training data size doesn't match Training labels size")
    input();
if(test_images.shape[0] != test_labels.shape[0]):
    print("!!!WARNING!!! Training data size doesn't match Training labels size")
    input();

train_images = train_images.reshape(train_images.shape[0], 48, 48, 1)
test_images = test_images.reshape(test_images.shape[0], 48, 48, 1)
shape = (48,48,1)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# batch data
BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(train_images.shape[0]).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


#-------------------
#
#   NET SETUP
#
#-------------------
"""
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=shape),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dense(71, activation="softmax")
])
model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])



#-------------------
#
#   TRAINING
#
#-------------------
num_epochs = 10
for i in range (0, num_epochs):
    print("\nEpoch: ", i)
    model.fit(train_dataset, epochs=1, steps_per_epoch=math.ceil(train_images.shape[0]/BATCH_SIZE), verbose=2)
    #eval(model, test_dataset)
    test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
    print('Test loss:', test_loss, '\nTest accuracy:', test_acc)
"""
#model = tf.keras.models.load_model("models/ETL8/e10")
#model = tf.keras.models.load_model("models/IRL/128_64_32_d512_e30")
#model = tf.keras.models.load_model("models/IRL/3x64_e30")
model = tf.keras.models.load_model("models/IRL/e10")
model.fit(train_dataset, epochs=1, steps_per_epoch=math.ceil(train_images.shape[0]/BATCH_SIZE), verbose=2)
eval_plot(model, test_dataset)
"""

#model.save("models/ETL8/e" + str(num_epochs))
"""