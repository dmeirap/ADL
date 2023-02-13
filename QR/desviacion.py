from math import pow, sqrt

listaDatos = list()
restar = list()
elevar = list()

def imprimir(texto, valor):
    print('\n{} {}'.format(texto, valor))

def sumatoria(datos):
        sumatoria = float(sum(datos))
        return sumatoria
    
def media(datos):
    n = len(datos)
    mediana = sumatoria(datos) / n
    return round(mediana,2)

def restar_media_datos(datos):

    mediana = media(datos)
    imprimir ('Media: ', mediana)

    for i in datos:
        op = i - mediana
        restar.append(op)

def elevar_cuadrado(datos):

    for i in datos:
        op = pow (i, 2)
        elevar.append(op)


def raiz_datos():
    desviacion = sqrt(media(elevar))
    return round(desviacion,2)

def main(angulo):

    repetir = 3
    listaDatos.clear()
    for i in range(repetir):
            number = angulo
            listaDatos.append(number)

    suma = sumatoria(listaDatos)
    imprimir ('Sumatoria: ', suma)

    restar_media_datos(listaDatos)

    elevar_cuadrado(restar)

    mediana = media(elevar)
    imprimir ('Varianza: ', mediana)

    desviacion = raiz_datos()
    imprimir ('Desviación Estándar: ', desviacion)
    print ('\n')

    
    
if __name__ == '__main__': #Incio
    main()