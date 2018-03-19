## Codigo creado por Jorge Martin
## https://pixelexperiment.wordpress.com
## Puedes copiar, modificar o distribuir este Software libremente
## Pero no elimines estos creditos por favor, gracias.

import cv2, sys, numpy, os, time, webbrowser, signal
size = 4
fn_haar = 'haarcascade_frontalface_alt.xml'
fn_dir = 'att_faces/orl_faces'

# Part 1: Creando fisherRecognizer
os.system("clear")

print('Cargando...')
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

## Crear una matriz Numpy de las dos listas anteriores
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

## OpenCV entrena un modelo a partir de las imagenes


model1 = cv2.createFisherFaceRecognizer()
#model1 = cv2.face.createFisherFaceRecognizer()
model1.load("Entrenador.yml")
##Declaro la variable cNombre1 para evitar falsos positivos
cNombre1 = 0
cNombre2 = 0

#Nivel seguridad
Seguridad = 0



os.system("clear")
print ("Carga compretada")
print ("Selecciona el nivel de seguridad")
print ("1 = Seguridad baja")
print ("10 =Seguridad media")
Seguridad = int(input("30 = Seguridad Alta\n"))
os.system("clear")


# Part 2: Utilizar fisherRecognizer en funcionamiento la camara
haar_cascade = cv2.CascadeClassifier(fn_haar)
webcam = cv2.VideoCapture(0)
while True:
    
    (rval, frame) = webcam.read()
    frame=cv2.flip(frame,1,0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mini = cv2.resize(gray, (gray.shape[1] / size, gray.shape[0] / size))
    faces = haar_cascade.detectMultiScale(mini)
    for i in range(len(faces)):
        face_i = faces[i]
        print ("aqui..")
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))
        # Intentado reconocer la cara
        prediction = model1.predict(face_resize)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Escribiendo el nombre de la cara reconocida
        # [1]


        if prediction[1]<500:
            cv2.putText(frame, '%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

	    #La variable cara tendra el nombre de la persona reconocida  
            cara = '%s' % (names[prediction[0]])
	    #Si cara = Nombre1 escribir Nombre1

            if cara == "Nombre1":
                cv2.putText(frame,
                'Nombre1',
                (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                cv2.imshow('OpenCV', frame)
                # Limpio variables de otras personas y
                # sumo 1 a la del seleccionado por camara,
                #si llega a 30 abre la pagina Web                 
                cNombre2 = 0
       
                cNombre1 = cNombre1 + 1
                
                if cNombre1 == Seguridad:

                    print('Se ha detectado a:')
                    print(cara)
                    url = "http://www.WEB1.com"
                    webbrowser.open_new(url)
                    print("Si deseas analizar otra cara introduce 1")
                    rep = int(input('En caso contrario introduce 0\n') )                
                    
                    if rep == 0:
                        os.kill(os.getppid(), signal.SIGHUP)
                    else:
                        cNombre1 = 0
                        os.system("clear")


            if cara == "Nombre2":
                cv2.putText(frame,
                'Nombre2',
                (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                cv2.imshow('OpenCV', frame)               
                cNombre1 = 0

                cNombre2 = cNombre2 + 1
                
                if cNombre2 == Seguridad:
                    print('Se ha detectado a:')
                    print(cara)
                    url = "http://www.WEB2.com/"
                    webbrowser.open_new(url)
                    print("Si deseas analizar otra cara introduce 1")
                    rep = int(input('En caso contrario introduce 0\n') )                
                    
                    if rep == 0:
                        os.kill(os.getppid(), signal.SIGHUP)
                    else:
                        cNombre2 = 0
                        os.system("clear")
                    
                    
                
            key = cv2.waitKey(10)
            if key == 27:
                    break

## Codigo creado por Jorge Martin
## https://pixelexperiment.wordpress.com
## Puedes copiar, modificar o distribuir este Software libremente
## Pero no elimines estos creditos por favor, gracias.
