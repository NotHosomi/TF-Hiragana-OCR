import os
from PIL import Image, ImageOps

label_roma = ["a","i","u","e","o","ka","ga","ki","gi","ku","gu","ke","ge","ko","go","sa",
"za","shi","ji","su","zu","se","ze","so","zo","ta","da","chi","dji","tsu","tzu","te","de","to","do",
"na","ni","nu","ne","no","ha","ba","pa","hi","bi","pi","fu","bu","pu","he","be","pe","ho","bo",
"po","ma","mi","mu","me","mo","ya","yu","yo","ra","ri","ru","re","ro","wa","wo","n"]

# note: no "ze", "dji", "pu" in data
count = 0
for i in range(0, 71):
    path = "IRL-data-raw/" + label_roma[i] + "/"
    new_path = "IRL-data/" + label_roma[i] + "/"
    f_list = os.listdir(path)
    print(label_roma[i], ": ", len(f_list))
    for f_name in f_list:
        image = Image.open(path + f_name)
        new_img = image.resize((48, 48), Image.ANTIALIAS)
        new_img = ImageOps.grayscale(new_img)
        new_img.save(new_path + f_name, "PNG")
        count+=1
print("Done: ", count, " images")