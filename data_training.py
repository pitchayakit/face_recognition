import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display

# img=mpimg.imread('tam_test.jpg')
# imgplot = plt.imshow(img)
# plt.show()


# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

# Load a sample picture and learn how to recognize it.
tam_image = face_recognition.load_image_file("data_set/pitchayakit.jpg")
tam_face_encoding = face_recognition.face_encodings(tam_image)[0]

# Load a second sample picture and learn how to recognize it.
azura_image = face_recognition.load_image_file("data_set/azura.jpg")
azura_face_encoding = face_recognition.face_encodings(azura_image)[0]

carrie_image = face_recognition.load_image_file("data_set/Carrie.jpg")
carrie_face_encoding = face_recognition.face_encodings(carrie_image)[0]

chanwit_image = face_recognition.load_image_file("data_set/Chanwit.jpg")
chanwit_face_encoding = face_recognition.face_encodings(chanwit_image)[0]

cheng_hong_image = face_recognition.load_image_file("data_set/cheng_hong.jpg")
cheng_hong_face_encoding = face_recognition.face_encodings(cheng_hong_image)[0]

dorris_image = face_recognition.load_image_file("data_set/Dorris.jpg")
dorris_face_encoding = face_recognition.face_encodings(dorris_image)[0]

felix_image = face_recognition.load_image_file("data_set/Felix.jpg")
felix_face_encoding = face_recognition.face_encodings(felix_image)[0]

irma_image = face_recognition.load_image_file("data_set/Irma.jpg")
irma_face_encoding = face_recognition.face_encodings(irma_image)[0]

nathaporn_image = face_recognition.load_image_file("data_set/Nathaporn.jpg")
nathaporn_face_encoding = face_recognition.face_encodings(nathaporn_image)[0]

nisa_dwi_septia_image = face_recognition.load_image_file("data_set/nisa_dwi_septia.jpg")
nisa_dwi_septia_face_encoding = face_recognition.face_encodings(nisa_dwi_septia_image)[0]

phanuvich_image = face_recognition.load_image_file("data_set/Phanuvich.jpg")
phanuvich_face_encoding = face_recognition.face_encodings(phanuvich_image)[0]

puri_image = face_recognition.load_image_file("data_set/puri.jpg")
puri_face_encoding = face_recognition.face_encodings(puri_image)[0]

rahayu_setyowati_image = face_recognition.load_image_file("data_set/Rahayu_Setyowati.jpg")
rahayu_setyowati_face_encoding = face_recognition.face_encodings(rahayu_setyowati_image)[0]

randi_image = face_recognition.load_image_file("data_set/Randi.jpg")
randi_face_encoding = face_recognition.face_encodings(randi_image)[0]

reamber_image = face_recognition.load_image_file("data_set/Reamber.jpg")
reamber_face_encoding = face_recognition.face_encodings(reamber_image)[0]

trio_image = face_recognition.load_image_file("data_set/Trio.jpg")
trio_face_encoding = face_recognition.face_encodings(trio_image)[0]

vincent_ch_image = face_recognition.load_image_file("data_set/Vincent_Ch.jpg")
vincent_ch_face_encoding = face_recognition.face_encodings(vincent_ch_image)[0]

william_image = face_recognition.load_image_file("data_set/william.jpg")
william_face_encoding = face_recognition.face_encodings(william_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    tam_face_encoding,
    azura_face_encoding,
    carrie_face_encoding,
    chanwit_face_encoding,
    cheng_hong_face_encoding,
    dorris_face_encoding,
    felix_face_encoding,
    irma_face_encoding,
    nathaporn_face_encoding,
    nisa_dwi_septia_face_encoding,
    phanuvich_face_encoding,
    puri_face_encoding,
    rahayu_setyowati_face_encoding,
    randi_face_encoding,
    reamber_face_encoding,
    trio_face_encoding,
    vincent_ch_face_encoding,
    william_face_encoding
]

print(known_face_encodings)