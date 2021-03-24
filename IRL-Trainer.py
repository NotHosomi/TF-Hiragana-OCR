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

#------
#
# LOSS/ACC PLOTTING
#
#------
def plot_hist(hist):
    # loss
    plt.plot(hist[0], 'b-', )
    plt.plot(hist[2], 'r-', )
    plt.axis([0, hist[0].size-1, 0, 0.5])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"])
    plt.show()
    # accuracy
    plt.plot(hist[1], 'b-', )
    plt.plot(hist[3], 'r-', )
    plt.axis([0, hist[0].size-1, 0.5, 1])
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"])
    plt.show()
    
#

#------
#
# DATA LOADING
#
#------
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
split = int(size * 0.2)
train_dataset = dataset.skip(split)
test_dataset = dataset.take(split) 
train_dataset = train_dataset.repeat().shuffle(size).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE).shuffle(BATCH_SIZE)



#-------------------
#
#   NET SETUP
#
#-------------------
def build_model(c1, c2 = 0, c3 = 0, c4 = 0, c5 = 0, drop = 0.5, dense = 1024):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(c1, (3,3), activation='relu', input_shape=(48, 48, 1)))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    if(c2 != 0):
        model.add(tf.keras.layers.Conv2D(c2, (3,3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2,2))
    if(c3 != 0):
        model.add(tf.keras.layers.Conv2D(c3, (3,3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2,2))
    if(c4 != 0):
        model.add(tf.keras.layers.Conv2D(c4, (3,3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2,2))
    if(c5 != 0):
        model.add(tf.keras.layers.Conv2D(c5, (3,3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Flatten())
    if(drop != 0):
        model.add(tf.keras.layers.Dropout(drop))
    model.add(tf.keras.layers.Dense(dense, activation='relu'))
    model.add(tf.keras.layers.Dense(71, activation="softmax"))
    model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
    return model

def run_keyinfo(model, num_epochs, name):
    #print("\n\n---------\n  NEW MODEL: ", name, "_e", num_epochs, "\n---------\n")
    print("\n", name, "e" + str(num_epochs))
    best_acc = -1
    best_acc_e = -1
    best_los = 10
    best_los_e = -1
    acc_at_5 = 0
    acc_at_10 = 0
    acc_at_20 = 0
    acc_at_30 = 0
    acc_at_40 = 0
    acc_at_50 = 0
    landmark95 = 0
    landmark99 = 0
    for i in range (0, num_epochs):
        #print("\nEpoch: ", i)
        model.fit(train_dataset, epochs=1, steps_per_epoch=math.ceil(size/BATCH_SIZE), verbose=2)
        test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
        #print('Test loss:', test_loss, '\nTest accuracy:', test_acc)
        if(best_acc < test_acc):
            best_acc = test_acc
            best_acc_e = i
            if(landmark95 == 0 and best_acc > 0.95):
                landmark95 == best_acc
            if(landmark99 == 0 and best_acc > 0.99):
                landmark99 == best_acc
        if(best_los > test_loss):
            best_los = test_loss
            best_los_e = i 
        if(i == 4):
            acc_at_5 = test_acc
        elif(i == 9):
            acc_at_10 = test_acc
        elif(i == 19):
            acc_at_20 = test_acc
        elif(i == 29):
            acc_at_30 = test_acc
        elif(i == 39):
            acc_at_40 = test_acc
        elif(i == 49):
            acc_at_50 = test_acc
    print("\nBest acc:  ", best_acc, "@", best_acc_e)
    print("Best loss: ", best_los, "@", best_los_e)
    print("waypoints: ", acc_at_5, acc_at_10, acc_at_20, acc_at_30, acc_at_40, acc_at_50)
    print("landmarks: ", landmark95, landmark99)
    model.save("models/IRL/" + name + "_e" + str(num_epochs))


def run_fullinfo(model, num_epochs, name):
    print("\n\n---------\n  NEW MODEL: ", name, "_e", num_epochs, "\n---------\n")
    # tr loss, tr acc, te loss, te acc
    hist = np.zeros([4, num_epochs])
    for i in range (0, num_epochs):
        #print("\nEpoch: ", i)
        # TRAIN
        fit_hist = model.fit(train_dataset, epochs=1, steps_per_epoch=math.ceil(size/BATCH_SIZE), verbose=2)
        hist[0][i] = fit_hist.history.get("loss")[-1]
        hist[1][i] = fit_hist.history.get("accuracy")[-1]
        # EVAL
        test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
        hist[2][i] = test_loss
        hist[3][i] = test_acc
        #print('Test loss:', test_loss, '\nTest accuracy:', test_acc)
    model.save("models/IRL/" + name + "_e" + str(num_epochs))
    print(hist)
    return hist
    


#-------------------
#
#   TRAINING
#
#-------------------
#e = 2
#m = build_model(64, 64, 64)
#run_fullinfo(m, e, "test")

e = 50

m = build_model(64, 64, 64)
hist_a = run_fullinfo(m, e, "3x64(2)")

m = build_model(128, 64, 32)
hist_b = run_fullinfo(m, e, "128_64_32(3)")

m = build_model(128, 64, 32, dense = 512)
hist_c = run_fullinfo(m, e, "128_64_32_d512(2)")

m = build_model(128, 64, 32, dense = 768)
hist_d = run_fullinfo(m, e, "128_64_32_d768")

m = build_model(128, 128, 128)
hist_e = run_fullinfo(m, e, "3x128")

m = build_model(128, 128, 128, dense = 512)
hist_f = run_fullinfo(m, e, "3x128_d512")

plot_hist(hist_a)
plot_hist(hist_b)
plot_hist(hist_c)
plot_hist(hist_d)
plot_hist(hist_e)
plot_hist(hist_f)

"""
m = build_model(64, 64, 64)
run(m, e, "3x64")

m = build_model(128, 64, 32)
run(m, e, "128_64_32(3)")

m = build_model(128, 64, 32, dense = 512)
run(m, e, "128_64_32_d512(2)")

m = build_model(32, 64, 128)
run(m, e, "32_64_128")

m = build_model(32, 64, 128, dense = 512)
run(m, e, "32_64_128_d512")

m = build_model(64, 64, 64, 64)
run(m, e, "4x64")

m = build_model(64, 64, 64, 64, 64)
run(m, e, "5x64")

m = build_model(32, 32, 32, 32, 32)
run(m, e, "5x32")

m = build_model(64, 64, 64, 64, 64, dense = 512)
run(m, e, "5x64_d512")

m = build_model(128, 128, 128)
run(m, e, "3x128")

m = build_model(128, 128, 128, dense = 512)
run(m, e, "3x128_d512")

m = build_model(128, 128, 128, 128)
run(m, e, "4x128")

m = build_model(128, 128, 128, 128, dense = 512)
run(m, e, "4x128_d512")

m = build_model(128, 128, 128, 128, 128)
run(m, e, "5x128")

m = build_model(128, 128, 128, 128, 128, dense = 512)
run(m, e, "5x128_d512")

m = build_model(32, 64, 128, 128, 128)
run(m, e, "32_64_3x128")

m = build_model(32, 64, 128, 128, 128, dense = 512)
run(m, e, "32_64_3x128_d512")

m = build_model(32, 64, 64, 128)
run(m, e, "32_2x64_128")

m = build_model(32, 64, 64, 128, dense = 512)
run(m, e, "32_2x64_128_d512")

m = build_model(32, 64, 64, 128, 128)
run(m, e, "32_2x64_2x128")

m = build_model(32, 64, 64, 128, 128, dense = 512)
run(m, e, "32_2x64_2x128_d512")

m = build_model(32, 32, 64, 64, 128)
run(m, e, "2x32_2x64_128")

m = build_model(32, 32, 64, 64, 128, dense = 512)
run(m, e, "2x32_2x64_128_d512")
"""