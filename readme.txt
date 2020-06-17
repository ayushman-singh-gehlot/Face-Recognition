this i as face recognition model using FaceNet.

codes inception_blocks.py and util.py are borrowed from FaceNet

->model take image as input(image size 96 * 96)
->complete image path is needed for input ex :- images/img/13.1.jpg
->It also has a small database(using dictionary) which conatain a name and a image corresponding to it.

model was trained using tripletLoss and degree of difference concept.
weights of tarined model is stored in folder which is used and loaded in this model. 
code works well for face verification and face recognition.
one can also add new images and update database(dict) with same. 
  