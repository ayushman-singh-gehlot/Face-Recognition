import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras import backend as K
K.set_image_data_format('channels_first')
import numpy as np
import tensorflow as tf
from util import *
from inception_blocks import *


print("\n\t\t--------------Face Recognition--------------")
print("Example of Input expected (96*96 image):  <images\img13.1.jpg>")
faceRecoModel = FRModel(input_shape=(3, 96, 96))
print("Total Params:", faceRecoModel.count_params())

print("please wait for few seconds")

def triplet_loss( y_pred, alpha = 0.2): 
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))    
    return loss



faceRecoModel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy']) 
load_weights_from_FaceNet(faceRecoModel)

database={}
database["danielle"] = "images/img10.jpg"
database["younes"] = "images/img6.jpg"
database["tian"] = "images/img11.jpg"
database["kian"] = "images/img4.jpg"
database["dan"] = "images/img9.jpg"
database["sebastiano"] = "images/img7.jpg"
database["bertrand"] = "images/img1.jpg"
database["kevin"] = "images/img5.jpg"
database["felix"] = "images/img12.jpg"
database["benoit"] = "images/img3.jpg"
database["arnaud"] = "images/img8.jpg"
database["ayushman"] = "images/img13.jpg"

                

def distance(img1,img2):
    encodingImg1=img_to_encoding(img1,faceRecoModel)
    encodingImg2=img_to_encoding(img2,faceRecoModel)
    dist=np.linalg.norm(encodingImg1-encodingImg2)
    return dist

def faceVerification(img1,img2):
    
    dist = distance(img1,img2)
    if dist < 0.7:
        #print("These two pictures are of SAME person")
        return dist,True
    else:
        #print("These two pictures are of DIFFERENT person")
        return dist,False

def faceRecognition(img1,database):
    min=10
    personName="nil"
    for name in database:
        dist,res=faceVerification(img1,database[name])
        if dist<min and res:
            min=dist
            personName=name
            #print(name,min)
    return personName,min

def main():
    imageOfFace=input("enter image path of face to be recognized : ")
    Name,dist=faceRecognition(imageOfFace,database)
    if Name!="nil":
        print("Person Identified : ",Name,"(",dist,")")
    else:
        print("Person Not Present in database")    
 
if __name__ == '__main__': 
	main() 