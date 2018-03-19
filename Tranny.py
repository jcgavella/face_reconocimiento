
import cv2, sys, numpy, os
size = 4
fn_haar = 'haarcascade_frontalface_alt.xml'
fn_dir = 'att_faces/orl_faces'


print('Creando una lista de imagenes y de nombres correspondientes')
# Crear una lista de imagenes y una lista de nombres correspondientes
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(fn_dir):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(fn_dir, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(im_width, im_height) = (112, 92)

print('Creando una matriz Numpy de las dos listas anteriores')
# Crear una matriz Numpy de las dos listas anteriores
(images, lables) = [numpy.array(lis) for lis in [images, lables]]



print('Comenzando el entrenamiento')
# OpenCV entrena un modelo a partir de las imagenes
model0 = cv2.createFisherFaceRecognizer()
#model0 = cv2.face.createFisherFaceRecognizer()
model0.train(images, lables)
model0.save("Entrenador.yml")

print('Entrenamiento completado con exito')
print('Programa finalizado')

