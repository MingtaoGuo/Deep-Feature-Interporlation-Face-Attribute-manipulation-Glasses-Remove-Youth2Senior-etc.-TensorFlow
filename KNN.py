import numpy as np
import scipy.io as sio
from PIL import Image
import os
from config import *


attributes = "person	imagenum	Male	Asian	White	Black	Baby	Child	Youth	Middle Aged	Senior	Black Hair	Blond Hair	Brown Hair	Bald" \
       "	No Eyewear	Eyeglasses	Sunglasses	Mustache	Smiling	Frowning	Chubby	Blurry	Harsh Lighting	Flash	Soft Lighting	Outdoor	Curly Hair	Wavy Hair" \
       "	Straight Hair	Receding Hairline	Bangs	Sideburns	Fully Visible Forehead	Partially Visible Forehead	Obstructed Forehead	Bushy Eyebrows	Arched Eyebrows	Narrow Eyes	Eyes Open" \
       "	Big Nose	Pointy Nose	Big Lips	Mouth Closed	Mouth Slightly Open	Mouth Wide Open	Teeth Not Visible	No Beard	Goatee	Round Jaw	Double Chin	Wearing Hat	Oval Face" \
       "	Square Face	Round Face	Color Photo	Posed Photo	Attractive Man	Attractive Woman	Indian	Gray Hair	Bags Under Eyes	Heavy Makeup	Rosy Cheeks	Shiny Skin	Pale Skin" \
       "	5 o' Clock Shadow	Strong Nose-Mouth Lines	Wearing Lipstick	Flushed Face	High Cheekbones	Brown Eyes	Wearing Earrings	Wearing Necktie	Wearing Necklace"



class person:
    def __init__(self, name, dist, imgnum):
        self.name = name
        self.num = imgnum
        self.dist = dist


def load_label(path="./lfw_label.mat"):
    return sio.loadmat(path)["lfwlabel1"]

def l1_distance(attr0, attr1):
    return np.mean(np.abs(attr0 - attr1))

def KNN(center, data, k=100):
    #col0: person, col1: image_num, col2-col75: attribute value
    dist_all = []
    #initialize
    dist0 = l1_distance(center[2:], data[0, 2:])
    p = person(data[0, 0], dist0, data[0, 1])
    dist_all.append(p)
    #KNN match
    for i in range(1, np.size(data, 0)):
        dist = l1_distance(center[2:], data[i, 2:])
        for j in range(dist_all.__len__()):
            if dist < dist_all[j].dist:
                p = person(data[i, 0], dist, data[i, 1])
                dist_all.insert(j, p)
                break
            if j == dist_all.__len__()-1:
                p = person(data[i, 0], dist, data[i, 1])
                dist_all.append(p)

    return dist_all[:k]

def get_corresponding_attr_dataset(data, attribute):
    attr_num = attributes.split("\t").__len__()
    attrs = dict(zip(attributes.split("\t"), range(attr_num)))#attrs: "person":0, "imagenum":1, "Male":2, ...
    neg = 0
    pos = 0
    for i in range(np.size(data, 0)):
        if data[i, attrs[attribute]] < 0:
            neg += 1
        else:
            pos += 1
    negative = np.zeros([neg, attr_num], dtype=np.object)
    positive = np.zeros([pos, attr_num], dtype=np.object)
    neg = 0
    pos = 0
    for i in range(np.size(data, 0)):
        if data[i, attrs[attribute]] < -1:
            negative[neg, :] = data[i, :]
            neg += 1
        elif data[i, attrs[attribute]] > 1:
            positive[pos, :] = data[i, :]
            pos += 1
    return negative, positive


def save_KNN_img(attribute, object_attr_vector):
    data = load_label()
    negative, positive = get_corresponding_attr_dataset(data, attribute)
    img_names = os.listdir("./lfw//" + object_attr_vector[0][0].replace(" ", "_"))
    Image.open("./lfw//" + object_attr_vector[0][0].replace(" ", "_") + "//" + img_names[0]).save("./input_img//" + object_attr_vector[0][0] + ".jpg")
    neg = KNN(object_attr_vector, negative, K)
    pos = KNN(object_attr_vector, positive, K)
    for i in range(K):
        filename = neg[i].name[0].replace(" ", "_")
        img_num = neg[i].num[0]
        Image.open("./lfw//" + filename + "//" + filename + "_" + "%04d" % (img_num) + ".jpg").save("./source//" + str(i) + ".jpg")
        filename = pos[i].name[0].replace(" ", "_")
        img_num = pos[i].num[0]
        Image.open("./lfw//" + filename + "//" + filename + "_" + "%04d" %(img_num) + ".jpg").save("./target//" + str(i) + ".jpg")
    pass

if __name__=="__main__":
    #Mouth Slightly Open
    #Mustache
    #Smiling
    #Senior
    #Middle Aged
    #Eyes Open
    #Eyeglasses
    target_attr = "Mustache"
    data = load_label()
    save_KNN_img(target_attr, data[0, :])


