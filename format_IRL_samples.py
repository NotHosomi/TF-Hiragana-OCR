# UNUSED!!!
# See resize_images.py

import os

label_hira = ["あ","い","う","え","お","か","が","き","ぎ","く","ぐ","け","げ","こ","ご","さ",
"ざ","し","じ","す","ず","せ","ぜ","そ","ぞ","た","だ","ち","ぢ","つ","づ","て","で","と","ど",
"な","に","ぬ","ね","の","は","ば","ぱ","ひ","び","ぴ","ふ","ぶ","ぷ","へ","べ","ぺ","ほ","ぼ",
"ぽ","ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","ん"]
label_roma = ["a","i","u","e","o","ka","ga","ki","gi","ku","gu","ke","ge","ko","go","sa",
"za","shi","ji","su","zu","se","ze","so","zo","ta","da","chi","dji","tsu","tzu","te","de","to","do",
"na","ni","nu","ne","no","ha","ba","pa","hi","bi","pi","fu","bu","pu","he","be","pe","ho","bo",
"po","ma","mi","mu","me","mo","ya","yu","yo","ra","ri","ru","re","ro","wa","wo","n"]
label_hira2 =
["あ","い","う","え","お",
"か","き","く","け","こ",
"が","ぎ","ぐ","げ","ご",
"さ","し","す","せ","そ",
"ざ","じ","ず","ぜ","ぞ",
"た","ち","つ","て","と",
"だ","ぢ","づ","で","ど",
"な","に","ぬ","ね","の",
"は","ひ","ふ","へ","ほ",
"ば","び","ぶ","べ","ぼ",
"ぱ","ぴ","ぷ","ぺ","ぽ",
"ま","み","む","め","も",
"や","ゆ","よ",
"ら","り","る","れ","ろ",
"わ","を","ん"]
label_roma2 =
["a","i","u","e","o",       #0     #0   #0
"ka","ki","ku","ke","ko",   #5     #1   #1
"sa","shi","su","se","so",  #15    #2   #2
"ta","chi","tsu","te","to", #20    #3   #3
"ha","hi","fu","he","ho",   #25    #4   #5
"ga","gi","gu","ge","go",   #30 D1 #5
"za","ji","zu","ze","zo",   #35 D1 #6
"da","dji","tzu","de","do", #40 D1 #7
"ba","bi","bu","be","bo",   #45 D1 #8   
"pa","pi","pu","pe","po",   #50 D2 #9  
"na","ni","nu","ne","no",   #35    #10  #4
"ma","mi","mu","me","mo",   #55    #11  #6
"ya","yu","yo",             #60+   #12  #7
"ra","ri","ru","re","ro",
"wa","wo","n"]
label_roma_s =
["a","i","u","e","o",       #0
"ka","ki","ku","ke","ko",   #5
"sa","shi","su","se","so",  #10
"ta","chi","tsu","te","to", #15
"na","ni","nu","ne","no",   #20
"ha","hi","fu","he","ho",   #25
"ma","mi","mu","me","mo",   #30
"ya","yu","yo",             #35+
"ra","ri","ru","re","ro",
"wa","wo","n",
"dia1","dia2"]              #46, 47


def load_images():
    DIR = "IRL-Data"
    for i in range(0, 70):
        int size = len(listdir(DIR + label_roma2[i]))
        imgs = np.zeros([size, 48, 48])
        label = np.zeros(len(label_roma_s))

        char_id = i;
        if i >= 30 and i < 50 :
            char_id -= 25
            label[46] = 1
        elif i >= 50 and i < 55 :
            char_id -= 25
            label[47] = 1
        label[char_id] = 1