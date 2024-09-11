import csv
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases, LdaModel, TfidfModel
from gensim.corpora import Dictionary
import json
#USAR LA VERSION 12.0 DE SCYPI

INPUT_FILE = "datos_Qatar.csv"
REVIEWS_VARIABLE = "Reviews"
NUM_TOPICS = 3
CHUNK_SIZE = 2000 # Tamaño de los bloques de documentos, como en este caso es superior lo hace en un solo bloque
EPOCHS = 20 # Epochs
ITERATIONS = 400


# EL SISTEMA DE CARGA DE DATOS ESTA IMPLEMENTADO, PERO EL PROGRAMA AUN NO USA LOS DATOS
# ESTA HARDCODEADO O CON IMPUTS
def read_config():
    # Pre: Nada
    # Post: Las variables globales cargadas con los datos del json
    global INPUT_FILE, REVIEWS_VARIABLE, NUM_TOPICS, CHUNK_SIZE, EPOCHS, ITERATIONS

    try:
        with open('config.json', 'r') as archivo:
            datos_json = json.load(archivo)
            INPUT_FILE = datos_json.get('INPUT_FILE', None)    # Path del archivo de entrada
            REVIEWS_VARIABLE = datos_json.get('REVIEWS_VARIABLE', None)
            NUM_TOPICS = datos_json.get('NUM_TOPICS', None)
            CHUNK_SIZE = datos_json.get('CHUNK_SIZE', None)  # Tamaño de los bloques de documentos, como en este caso es superior lo hace en un solo bloque
            EPOCHS = datos_json.get('EPOCHS', None)  # Epochs
            ITERATIONS = datos_json.get('ITERATIONS', None)
            print("Datos cargados exitosamente desde el archivo JSON.")

    except FileNotFoundError:
        print("El archivo JSON no se encontró.")
    except json.JSONDecodeError:
        print("Error al decodificar el archivo JSON.")
    except Exception as e:
        print(f"Error: {e}")

def verificar_nombre(nombre_base):
    #Comprobamos si ya existe un archivo con ese nombre, si existe se añade (numero)
    numero = 0
    nombre_archivo = nombre_base
    while os.path.exists(nombre_archivo):
        numero += 1
        nombre_archivo = f"{nombre_base.split('.csv')[0]}({numero}).csv"
    return nombre_archivo

def crearCsv():
    nombre=verificar_nombre('results.csv')
    with open(nombre, 'w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow(["Topico","Stop-words"])
        return nombre

def anadirEntradaCsv(nombre,topic,palabras_relevantes):
    with open(nombre, 'a', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        palabras = ", ".join(palabras_relevantes)
        escritor_csv.writerow([topic, palabras])

def preprocess_data():
    data = read_data()
    # Tokenizador de palabras
    tokenizer = RegexpTokenizer(r'\w+')
    data = data.apply(lambda x: x.lower())
    data = data.apply(lambda x: tokenizer.tokenize(x))  # Tokeniza las palabras
    data = data.apply(lambda x: [word for word in x if not word.isnumeric()])  # Elimina los números

    # Quitar stop words
    nltk.download('stopwords') #Se descarga/actualiza el diccionario de stopwords
    stop_words = set(stopwords.words('english'))
    data = data.apply(lambda x: [word for word in x if word not in stop_words])

    nltk.download('wordnet') #Se descarga/actualiza una base de datos léxica del ingles
    lemmatizer = WordNetLemmatizer()
    data = data.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])  # Lematiza las palabras

    # Añadir bigramas y trigramas a los datos
    bigram = Phrases(data, min_count=20)
    for i in range(len(data)):
        for word in bigram[data[i]]:
            if '_' in word:
                data[i].append(word)

    # Quitar palabras comunes y poco comunes
    dictionary = Dictionary(data)
    dictionary.filter_extremes(no_below=5, no_above=0.70)

    # BOW
    corpus = [dictionary.doc2bow(text) for text in data]

    #Creamos el vector tf-idf
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    return corpus_tfidf, dictionary


def read_data():
    datos = pd.read_csv(INPUT_FILE)
    data = datos[REVIEWS_VARIABLE]
    return data


def train_data(corpus, dictionary):
    num_topics = NUM_TOPICS #Numero de topics
    chunksize = CHUNK_SIZE  # Tamaño de los bloques de documentos, como en este caso es superior lo hace en un solo bloque
    passes = EPOCHS  # Epochs
    iterations = ITERATIONS #Num de iteraciones que se quieren hacer
    eval_every = None  # Evaluar la perplejidad del modelo

    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus, #Obtenido en el preproceso
        id2word=id2word, #Obtenido del diccionario
        chunksize=chunksize, #Parametro modificable en le json
        alpha='auto', #Calculo automatico de alfa
        eta='auto', #Calculo automatico de beta
        iterations=iterations, #Parametro modificable en el json
        num_topics=num_topics, #Parametro modificalble en el json
        passes=passes,
        eval_every=eval_every
    )


    top_topics = model.top_topics(corpus)
    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    # Mostrar las 15 palabras más frecuentes de cada tópico

    nombre = crearCsv()
    for i in range(num_topics):
        palabras = []
        print(f"Tópico {i + 1}:")
        topic_words = model.show_topic(i, topn=15)
        #Añadimos al array las palabras para pasarlas luego al csv
        #Todo: Optimizar -Ander
        for word in topic_words:
            palabras.append(word[0])
        print([word[0] for word in topic_words])
        anadirEntradaCsv(nombre,f"Tópico {i + 1}",palabras)


if __name__ == "__main__":
    read_config()#Lee la configuracion del json
    corpus, dictionary = preprocess_data()#Preproceso de datos
    train_data(corpus, dictionary)#Entrenar diccionario
