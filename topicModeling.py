from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import json

# --Variables para el programa--
INPUT_FILE = "airlines_reviewsTrainDev.csv"
REVIEWS_VARIABLE = "reviews"
VECTOR = 0  # 0 -> Count Vectorizer / 1 -> tf-idf
MIN_NUM_TOPICS = 0  # SIN IMPLEMENTAR TODAVIA
MAX_NUM_TOPICS = 0  # SIN IMPLEMENTAR TODAVIA
RANDOM_STATE = 0  # SIN IMPLEMENTAR TODAVIA
MAX_ITER = 10  # SIN IMPLEMENTAR TODAVIA
LEARNING_METHOD = 'online'  # SIN IMPLEMENTAR TODAVIA
LEARNING_OFFSET = 50.  # SIN IMPLEMENTAR TODAVIA
LEARNING_DECAY = 0.7  # SIN IMPLEMENTAR TODAVIA

# EL SISTEMA DE CARGA DE DATOS ESTA IMPLEMENTADO, PERO EL PROGRAMA AUN NO USA LOS DATOS
# ESTA HARDCODEADO O CON IMPUTS
def read_config():
    #Pre: Nada
    #Post: Las variables globales cargadas con los datos del json
    global INPUT_FILE, VECTOR, REVIEWS_VARIABLE, MIN_NUM_TOPICS, MAX_NUM_TOPICS, RANDOM_STATE, MAX_ITER, LEARNING_METHOD, LEARNING_OFFSET, LEARNING_DECAY

    try:
        with open('config.json', 'r') as archivo:
            datos_json = json.load(archivo)
            INPUT_FILE = datos_json.get('INPUT_FILE', None)    # Path del archivo de entrada
            REVIEWS_VARIABLE = datos_json.get('REVIEWS_VARIABLE', None)
            VECTOR = datos_json.get('VECTOR', None)
            MIN_NUM_TOPICS = datos_json.get('MIN_NUM_TOPICS', None)
            MAX_NUM_TOPICS = datos_json.get('MAX_NUM_TOPICS', None)
            RANDOM_STATE = datos_json.get('RANDOM_STATE', None)
            MAX_ITER = datos_json.get('MAX_ITER', None)
            LEARNING_METHOD = datos_json.get('LEARNING_METHOD', None)
            LEARNING_OFFSET = datos_json.get('LEARNING_OFFSET', None)
            LEARNING_DECAY = datos_json.get('LEARNING_DECAY', None)
            print("Datos cargados exitosamente desde el archivo JSON.")

    except FileNotFoundError:
        print("El archivo JSON no se encontró.")
    except json.JSONDecodeError:
        print("Error al decodificar el archivo JSON.")
    except Exception as e:
        print(f"Error: {e}")


# Leer datos del fichero .csv, solamente columna de Reviews.
def read_data_from_file():
    data = pd.read_csv('airlines_reviewsTrainDev.csv')
    reviews = data['Reviews']
    return reviews

def vectorize_reviews():
    #Pre: Nada
    #Out: Se devuelve el vector tf-idf y el nombre de las palabras usadas en el tf-idf
    reviews = read_data_from_file()
    opcion = input("Elige: 0 = countVectorizer, 1 = tf-idf: ")
    if opcion == 0:
        matrix_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        matrix = matrix_vectorizer.fit_transform(reviews)
    else:
        matrix_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        matrix = matrix_vectorizer.fit_transform(reviews)
    matrix_feature_names = matrix_vectorizer.get_feature_names_out()
    return matrix, matrix_feature_names


def run_lda():
    #Pre: el vector tf-idf y el nombre de las palabras usadas en el tf-idf (cargado abajo)
    #Post: Se imprimen los resultados de los topics
    matrix, matrix_feature_names = vectorize_reviews()  # Vectorizamos los datos
    num_topics = 3
    try:
        num_topics = int(input('Enter the number of topics: '))
    except ValueError:
        print('Please enter a valid number. Eg: 5')
        num_topics = int(input('Enter the number of topics: '))

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0, max_iter=100, learning_method='online',
                                    learning_offset=50., learning_decay=0.7).fit(matrix)

    reviews = read_data_from_file()
    print()
    for topic_idx, topic in enumerate(lda.components_):
        top_terms_indices = topic.argsort()[:-10 - 1:-1]  # Obtener los índices de los 10 términos más importantes
        top_terms = [matrix_feature_names[i] for i in top_terms_indices] #se ordenan por relevancia
        print(f"Top términos para el tema {topic_idx}:")
        print(top_terms)

        # Obtenemos los índices de los documentos más representativos para cada tema
        topic_distribution = lda.transform(matrix)
        representative_docs_ind = sorted(enumerate(topic_distribution[:, topic_idx]), key=lambda x: x[1],
                                             reverse=True)[:4] #Obtenemos los 4 mas relevantes del tema
        print("Top reviews del tema (Sin procesar):")
        #for i, _ in representative_docs_ind:
            #print("-", reviews[i])
        print()


if __name__ == "__main__":
    #read_config() //Como aun no usamos los datos en el programa, no hace falta cargarlos
    run_lda()
