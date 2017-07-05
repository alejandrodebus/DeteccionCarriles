# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def regionInteres(img, vertices):
    # Se crea una mascara del tamaño de la imagen del mismo tipo
    mascara = np.zeros_like(img)

    if len(img.shape) > 2:
        cant_canales = img.shape[2]
        mascara_ignorada = (255,) * cant_canales
    else:
        mascara_ignorada = 255

    cv2.fillPoly(mascara, vertices, mascara_ignorada)

    imagen_enmascarada = cv2.bitwise_and(img, mascara)

    return imagen_enmascarada

def dibujarLineas2(img, lineas):

    if lineas is not None:
        for linea in lineas:
            for x1, y1, x2, y2 in linea:
                pendiente, interseccion = np.polyfit((x1, x2), (y1, y2), 1)
                if abs(pendiente) > 0:
                    cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 10)
    return img

def dibujarLineas(img, lineas):

    min_y = img.shape[0]
    max_y = img.shape[0]

    # Para dibujar las lineas se realiza una extrapolacion
    izq_x = []
    izq_y = []
    pendiente_izq = []
    der_x = []
    der_y = []
    pendiente_der = []


    if lineas is not None:
        for linea in lineas:
            for x1, y1, x2, y2 in linea:
                pendiente, interseccion = np.polyfit((x1, x2), (y1, y2), 1)
                longitud = np.sqrt((y2-y1)**2 + (x2-x1)**2)
                if pendiente < 0:
                    pendiente_izq += [pendiente]
                    izq_x += [x1, x2]
                    izq_y += [y1, y2]
                elif pendiente > 0:
                    pendiente_der += [pendiente]
                    der_x += [x1, x2]
                    der_y += [y1, y2]

                min_y = min(min(y1, y2), min_y)

    pendiente_izq_media = np.mean(pendiente_izq)
    pendiente_der_media = np.mean(pendiente_der)
    izq_x_media = np.mean(izq_x)
    izq_y_media = np.mean(izq_y)
    der_x_media = np.mean(der_x)
    der_y_media = np.mean(der_y)

    interseccion_izq = izq_y_media - (pendiente_izq_media * izq_x_media)
    interseccion_der = der_y_media - (pendiente_der_media * der_x_media)

    if ((len(pendiente_izq)>1) and (len(pendiente_der)>1)):
        izq_sup_x = int((min_y - interseccion_izq)/pendiente_izq_media)
        izq_inf_x = int((max_y - interseccion_izq)/pendiente_izq_media)
        der_sup_x = int((min_y - interseccion_der)/pendiente_der_media)
        der_inf_x = int((max_y - interseccion_der)/pendiente_der_media)

        cv2.line(img, (izq_sup_x, min_y), (izq_inf_x, max_y), (255,0,0), 15)
        cv2.line(img, (der_sup_x, min_y), (der_inf_x, max_y), (255,0,0), 15)

    return img

# Leer video
cap = cv2.VideoCapture('solidYellowLeft.mp4')

# Parametros de HoughLinesP
rho = 1
theta = np.pi/180
threshold = 1
minLineLength = 20
maxLineGap = 1

# Region de interes
# Para video Challenge.mp4
# p1 = [245, 645]
# p2 = [559, 450]
# p3 = [747, 450]
# p4 = [1075, 645]
# Para video solidYellowLeft.mp4 y solidWhiteRight.mp4
p1 = [123, 531]
p2 = [425, 336]
p3 = [543, 336]
p4 = [901, 531]
puntos = np.array([[p1, p2, p3, p4]], np.int32)

while(True):

    ret, image = cap.read()

    # Cada frame es convertido a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicacion de filtro Gaussiano
    kernel_size = 11
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    # Deteccion de bordes mediante algoritmo de Canny
    edges = cv2.Canny(blur_gray, 50, 150)

    img_enmascarada = regionInteres(edges, puntos)

    cv2.imshow('Enmascarada', img_enmascarada)

    # Deteccion de lineas mediante algoritmo de Hough
    lineas = cv2.HoughLinesP(img_enmascarada, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)

    # for linea in lineas:
    #     for x1,y1,x2,y2 in linea:
    #         cv2.line(image,(x1,y1),(x2,y2), (0,255,0), 2)

    img_lineas = dibujarLineas(image, lineas)

    cv2.imshow('Frames', img_lineas)

    if cv2.waitKey(10) == 27:
        break
