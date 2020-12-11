# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 17:04:58 2020

@author: DAVID
"""
from tkinter import *
from tkinter import ttk
import tkinter as tk
from PIL import Image, ImageTk
import PIL.Image, PIL.ImageTk
import cv2
import os
import time
import numpy as np
import imutils


print("Librerias leidas")

#-------------------------------------------
#            FUNCIONES
#-------------------------------------------

class RegistrarRostro(ttk.Frame):
    
    """Pestaña 1, Capturar rostros y entrenar"""
    
    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        
        self.cap = cv2.VideoCapture(0)
        self.master = self.master
        self.cap = self.cap
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
       
        #-------Detector facial OpenCV
        self.faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        
        #-------Funciones
        self.Crear_Widgets()
        self.Cuadro()
    
    def Cuadro(self):
        self.canvas = Canvas(self, height=500, width=620)
        self.canvas.grid(column=0, row=0, sticky='NW', padx=0, pady=40)
        self.canvas.create_rectangle(20, 20, 620, 500, fill="white", outline="gainsboro") 
        self.canvas1 = Canvas(self, height=170, width=170)
        self.canvas1.grid(row=0, column=1, sticky="NW", padx=60, pady=250)       
        self.canvas1.create_rectangle(5, 5, 150, 150, fill="white", outline="gainsboro") 
        
    def Crear_Widgets(self):
        
        #----------text variables
        self.DNI = tk.StringVar()
        self.Nombre = tk.StringVar()    
        self.Apellidos = tk.StringVar()
        self.Edad = tk.StringVar()
        
        #----------Buttons
        
        self.button1 = ttk.Button(self, text="Prender", cursor="hand2", command=self.WebcamTK).grid(row=0, column=0, sticky='NW', padx=20, pady=10) 
        self.button2 = ttk.Button(self, text="Apagar", cursor="hand2", command=self.Apagar_WebCam).grid(row=0, column=0, sticky="NW", padx=120, pady=10)
        self.button3 = ttk.Button(self, text="Enviar", cursor="hand2", command=self.Crear_Carpeta).grid(row=0, column=1, sticky="NW", padx=150, pady=120)
       
        #----------Label
        self.label1 = ttk.Label(self, text="Nombre:").grid(row=0, column=1, sticky='NW', padx=20, pady=90)
        
        #----------Text boxes
        self.textbox3 = ttk.Entry(self, textvariable=self.Nombre).grid(row=0, column=1, sticky='NW', padx=100, pady=90)
        
        
    def WebcamTK(self):
        
        ret, self.frame = self.cap.read()
        
        self.frame = cv2.flip(self.frame, 1)
        self.image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        
        self.Reco_Facial(self.image)
        
        self.image = Image.fromarray(self.image)
        self.image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(20, 20, anchor=tk.NW, image=self.image)
        
        self.master.after(10, self.WebcamTK) 
        
    def Reco_Facial(self, img):
        
        self.faces = self.faceClassif.detectMultiScale(img, 1.2, 5)
        
        if self.faces == ():
            return False 
        else:
            for(x, y, w, h) in self.faces:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                img = cv2.putText(img,'PRESIONE CAPTURAR',(10,20), 2, 0.5,(255,255,255),1,cv2.LINE_AA)
                
    def Crear_Carpeta(self):
        global newpath, id_input, progressbar
        
        if not os.path.exists('face_data/'):
            os.makedirs('face_data/')
            
        #----------Buttons
        self.button4 = ttk.Button(self, text="Capturar", cursor="hand2", command=self.Capturar_Rostro).grid(row=0, column=0, sticky="NW", padx=20, pady=550)
        self.button5 = ttk.Button(self, text="Entrenar", cursor="hand2", command=self.Entrenamiento).grid(row=0, column=0, sticky="NE", padx=2, pady=550)
        
        #----------Extraccion
        self.id_input = self.Nombre.get()
        
        self.newpath = r'face_data/'+ self.id_input
        if not os.path.exists(self.newpath):
            os.makedirs(self.newpath)
            
    
    def Capturar_Rostro(self):
        
        self.count = 0
        
        while True:
            ret, self.frame = self.cap.read()
            
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            
            self.faces = self.faceClassif.detectMultiScale(self.gray, 1.3, 5)
            
            if self.faces == ():
                return False
            else:
                for(x, y, w, h) in self.faces:
                    self.rostro = self.gray[y:y+h, x:x+w]
                    self.rostro = cv2.resize(self.rostro, (150,150), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(self.newpath + '/rostro_{}.jpg'.format(self.count), self.rostro)
                    
                    #Mostrar rostros capturados
                    self.foto = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.rostro))
                    self.canvas1.create_image(0,0, anchor=tk.NW, image=self.foto)
                    
                    
                    self.count += 1
                    print(self.count)
            
            time.sleep(0)
            if self.count == 50:
                self.count +=1
                return False
    
    def Entrenamiento(self):
        
        self.dataPath = 'D:/1_Investigacion/2.Segundo trabajo/Codigo/face_data' #Direccion de carpeta donde se almacena los rostros capturados
        self.peopleList = os.listdir(self.dataPath)
        print('Lista de personas', self.peopleList)
        
        self.labels = []
        self.facesData = []
        self.label = 0
        
        for self.nameDir in self.peopleList:
            self.personPath = self.dataPath + '/' + self.nameDir
            print('Leyendo las imágenes')
            
            for self.fileName in os.listdir(self.personPath):
                print('Rostros: ', self.nameDir + '/' + self.fileName)
                self.labels.append(self.label)
                self.facesData.append(cv2.imread(self.personPath+'/'+self.fileName,0))
            
            self.label +=1
            
        # Métodos para entrenar el reconocedor
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Entrenando el reconocedor de rostros
        print("Entrenando...")
        self.progreso()
        self.face_recognizer.train(self.facesData, np.array(self.labels))
        # Almacenando el modelo obtenido
        self.face_recognizer.write('2modeloLBPHFace.xml')
        print("Modelo almacenado")
        
        
    def progreso(self):
        
        self.i = 0
        self.progressbar = ttk.Progressbar(self, orient=HORIZONTAL, length=200, mode='determinate')
        self.progressbar.grid(row=0, column=0, sticky="NW", padx=110, pady=551)
        
        while(self.progressbar["value"] < self.progressbar["maximum"]):
            self.progressbar.update()
            self.progressbar["value"] = self.i**2
            self.i += 0.01
    
        self.progressbar.destroy()
        
    def Apagar_WebCam(self):
        #self.destroy()
        self.cap.release()  # liberar cámara web
        #cv2.destroyAllWindows()  # no es obligatorio en esta aplicación
        
#---------------------------------------------------------------------------------------------------------------------------------------------        
        
class ReconocimientoFacial(ttk.Frame):
    
    """Pestaña 2, Reconocimiento Facial"""
    
    def __init__(self, master=None):
        
        ttk.Frame.__init__(self, master)
        
        self.cap = cv2.VideoCapture(0)
        self.master = self.master
        self.cap = self.cap
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
       
        #-------Detector facial OpenCV
        self.faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        
        #-------Funcionaes
        self.Cuadro_2()
        self.Crear_Widgets_2()
        self.Ingresar_Carpeta()
        
        
        
    def Cuadro_2(self):
        self.canvas = Canvas(self, height=530, width=900)
        self.canvas.grid(column=0, row=0, sticky='NW', padx=0, pady=40)
        self.canvas.create_rectangle(20, 20, 880, 530, fill="white", outline="gainsboro")
        
    def Crear_Widgets_2(self):
        
        #----------Buttons
        
        self.button1 = ttk.Button(self, text="Prender", cursor="hand2", command=self.WebcamTK_2).grid(row=0, column=0, sticky='NW', padx=20, pady=10) 
        self.button2 = ttk.Button(self, text="Apagar", cursor="hand2", command=self.Apagar_WebCam_2).grid(row=0, column=0, sticky="NW", padx=120, pady=10)

    def WebcamTK_2(self):
        
        ret, self.frame = self.cap.read()
        
        self.frame = imutils.resize(self.frame, height=100, width=860)
        self.image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
       
        self.Reconocimiento_Facial(self.image)
        
        self.image = Image.fromarray(self.image)
        self.image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(20, 20, anchor=tk.NW, image=self.image)
        
        self.master.after(10, self.WebcamTK_2)
    
    def Ingresar_Carpeta(self):
        
        global imagePaths, face_recognizer
        
        self.dataPath = 'D:/1_Investigacion/2.Segundo trabajo/Codigo/face_data'
        
        self.imagePaths = os.listdir(self.dataPath)
        
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        #Leyendo el Modelo
        self.face_recognizer.read('modeloLBPHFace.xml')
    
    def Reconocimiento_Facial(self, img):
        
        self.faces = self.faceClassif.detectMultiScale(img, 1.2, 5)
        
        if self.faces == ():
            return False
        else:
            
            self.auxFrame = img.copy()
            
            for(x,y,w,h) in self.faces:
                
                self.rostro = img[y:y+h,x:x+w]
                self.rostro = cv2.resize(self.rostro,(150,150),interpolation= cv2.INTER_CUBIC)
                self.result = self.face_recognizer.predict(self.rostro)
                
                img = cv2.putText(img,'{}'.format(self.result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
                img = cv2.putText(img,'{}'.format(self.imagePaths[self.result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        
    def Apagar_WebCam_2(self):
        #self.destroy()
        self.cap.release()  # liberar cámara web
        #cv2.destroyAllWindows()  # no es obligatorio en esta aplicación
        
#-------------------------------------------
#            INTERFAZ
#-------------------------------------------

def main():
    
    ventana = Tk()
    ventana.title('Reconocimiento biometrico')
    ventana.resizable(0,0) #Tamaño estatico
    ventana.iconbitmap("img/logo3.ico")
    ventana.geometry("900x610")
    
    #Preparando notebook (tabs)
    notebook = ttk.Notebook(ventana)
    notebook.pack(fill='both', expand='yes')#empaqueta el lienzo en un marco / formulario
    
    Pes1 = ttk.Frame(notebook)
    Pes2 = ttk.Frame(notebook)
    
    notebook.add(Pes1, text="Registrar Rostros")
    notebook.add(Pes2, text="Reconocimiento Facial")
    
    #Crear marcos de pestañas
    Pestana1 = RegistrarRostro(master=Pes1)
    Pestana1.grid()
    
    Pestana2 = ReconocimientoFacial(master=Pes2)
    Pestana2.grid()

    #Main loop
    ventana.mainloop()

if __name__ == '__main__':
    main()