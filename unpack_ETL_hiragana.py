# HIRAGANA UNPACKER COURTESY OF 
# https://github.com/Nippon2019/Handwritten-Japanese-Recognition/blob/master/Hiragana/read_hira.py
# https://github.com/Nippon2019/Handwritten-Japanese-Recognition/blob/master/Hiragana/modify_hira.py

import struct
from PIL import Image
import numpy as np
import skimage.transform
from sklearn.model_selection import train_test_split

def read_record_ETL8G(f):
    s = f.read(8199)
    r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
    iF = Image.frombytes('F', (128, 127), r[14], 'bit', 4)
    iL = iF.convert('L')
    return r + (iL,)

def read_hira():
    hiragana = np.zeros([71, 160, 127, 128], dtype=np.uint8)
    for i in range(1, 33):
        filename = 'ETL8G/ETL8G_{:02d}'.format(i)
        with open(filename, 'rb') as f:
            for dataset in range(5):
                char = 0
                for j in range(956):
                    r = read_record_ETL8G(f)
                    if b'.HIRA' in r[2] or b'.WO.' in r[2]:
                        if not b'KAI' in r[2] and not b'HEI' in r[2]:
                            hiragana[char, (i - 1) * 5 + dataset] = np.array(r[-1])
                            char += 1
    np.savez_compressed("hiragana.npz", hiragana)
    
def split_hira():
    hira = np.load("hiragana.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32)

    hira = hira/np.max(hira)
        
    ## 71 characters, 160 writers, transform image to 48*48
    train_images = np.zeros([71 * 160, 48, 48], dtype=np.float32)

    for i in range(71 * 160):
        train_images[i] = skimage.transform.resize(hira[i], (48, 48))

    arr = np.arange(71)
    train_labels = np.repeat(arr, 160) # create labels

    # split to train and test ## 71 characters, 160 writers, transform image to 48*48
    train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2)

    np.savez_compressed("hiragana_train_images.npz", train_images)
    np.savez_compressed("hiragana_train_labels.npz", train_labels)
    np.savez_compressed("hiragana_test_images.npz", test_images)
    np.savez_compressed("hiragana_test_labels.npz", test_labels)

#read_hira()
#split_hira()