import os
from PIL import Image
import numpy as np
import random
import tensorflow as tf
import math
import matplotlib.pyplot as plt

label_hira = ["あ","い","う","え","お","か","が","き","ぎ","く","ぐ","け","げ","こ","ご","さ",
"ざ","し","じ","す","ず","せ","ぜ","そ","ぞ","た","だ","ち","ぢ","つ","づ","て","で","と","ど",
"な","に","ぬ","ね","の","は","ば","ぱ","ひ","び","ぴ","ふ","ぶ","ぷ","へ","べ","ぺ","ほ","ぼ",
"ぽ","ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","ん"]
label_roma = ["a","i","u","e","o","ka","ga","ki","gi","ku","gu","ke","ge","ko","go","sa",
"za","shi","ji","su","zu","se","ze","so","zo","ta","da","chi","dji","tsu","tzu","te","de","to","do",
"na","ni","nu","ne","no","ha","ba","pa","hi","bi","pi","fu","bu","pu","he","be","pe","ho","bo",
"po","ma","mi","mu","me","mo","ya","yu","yo","ra","ri","ru","re","ro","wa","wo","n"]
#
def draw(img, label_id):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.xlabel(label_roma[label_id])
    plt.show()
#
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

    plt.xlabel("{} {:2.0f}% ({})".format(label_roma
        [predicted_label],
        100*np.max(predictions_array),
        label_roma
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
def load_rand(size):
    char_ids = []
    imgs = np.zeros([size, 48, 48]) #image size
    lbls = np.zeros(size)
    for i in range(0, size):
        valid = True
        while valid:
            x = random.randint(0, 70)
            path = "IRL-data/" + label_roma[x] + "/"
            img_names = os.listdir(path);
            if len(img_names) == 0 and x not in char_ids:        
                valid = False
            else:
                valid = True
                char_ids.append(x)
                print("Seleted " + label_roma[x])
                img_id = random.randint(0, len(img_names)-1)
                f_name = img_names[img_id]
                img = Image.open(path + f_name)
                imgs[i] = np.array(img)
                lbls[i] = x
    return imgs, lbls
#
def load_even():
    size = 10000
    # Find the smallest class set
    for x in range(0, 71):
        path = "IRL-data/" + label_roma[x] + "/"
        count = len(os.listdir(path))
        if count < size:
            size = count
        if(count == 0):
            print("!!!WARNING!!! No data for label ", label_roma[x])
            input();

    
    imgs = np.zeros([71 * size, 48, 48], dtype=np.float32) #image size

    arr = np.arange(71)
    lbls = np.repeat(arr, size)

    print("size ", size)
    index = 0
    for x in range(0, 71):
        path = "IRL-data/" + label_roma[x] + "/"
        img_names = os.listdir(path);
        for sample in range(0, size):
            img_ids = []
            valid = False
            while(not valid):
                img_id = random.randint(0, len(img_names)-1)
                if img_id in img_ids:        
                    valid = False
                else:
                    valid = True
                    img_ids.append(img_id)
                    f_name = img_names[img_id]
                    img = Image.open(path + f_name)
                    #index = (x * 71) + sample
                    print(index,":\t", label_roma[x], "-", img_id)
                    imgs[index] = np.array(img)
                    index+=1
                
    imgs = imgs/np.max(imgs) # normalise images
    return imgs, lbls
#
def load_all():
    size = 0
    # Find the smallest class set
    for x in range(0, 71):
        path = "IRL-data/" + label_roma[x] + "/"
        size += len(os.listdir(path))

    
    imgs = np.zeros([size, 48, 48], dtype=np.float32) #image size
    lbls = np.zeros(size, dtype=np.uint8)

    print("size ", size)
    index = 0
    for x in range(0, 71):
        path = "IRL-data/" + label_roma[x] + "/"
        img_names = os.listdir(path);
        for f_name in img_names:
            img = Image.open(path + f_name)
            #index = (x * 71) + sample
            imgs[index] = np.array(img)
            lbls[index] = x
            index+=1
                
    imgs = imgs/np.max(imgs) # normalise images
    return imgs, lbls
#

#------
#
# DATA HANDLING
#
#------
BATCH_SIZE = 32
images, labels = load_all()
size = images.shape[0]
images = images.reshape(size, 48, 48, 1)
print("imgs shape", images.shape)
print("lbls shape", labels.shape)
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(size)
dataset = dataset.batch(BATCH_SIZE)

#------
#
# MODEL
#
#------
model = tf.keras.models.load_model("models/ETL8/e10")
eval_plot(model, test_dataset)