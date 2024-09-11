import nltk
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.models import TfidfModel, Phrases
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import IPython

# IPython necesario para poder ejecutar aunque no se use de forma directa

input_file = "datos_Qatar.csv"


def read_data():
    datos = pd.read_csv(input_file)
    data = datos["Reviews"]
    return data


def preprocess_data():
    data = read_data()
    # Tokenizador de palabras
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    data = data.apply(lambda x: x.lower())
    data = data.apply(lambda x: tokenizer.tokenize(x))  # Tokeniza las palabras
    data = data.apply(lambda x: [word for word in x if not word.isnumeric()])  # Elimina los números

    # Quitar stop words
    nltk.download('stopwords')  # Se descarga/actualiza el diccionario de stopwords
    stop_words = nltk.corpus.stopwords.words('english')
    data = data.apply(lambda x: [word for word in x if word not in stop_words])

    nltk.download('wordnet')  # Se descarga/actualiza una base de datos léxica del ingles
    lemmatizer = nltk.WordNetLemmatizer()
    data = data.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])  # Lematiza las palabras

    # Añadir bigramas y trigramas a los datos
    bigram = Phrases(data, min_count=5, threshold=100)
    trigram = Phrases(bigram[data], threshold=100)

    '''
    Si se quiere tener en cuenta solamente los bigramas y trigramas habría que poner:
    data_gramas = []
    cambiar los appends del for a data_gramas.append(word)
    por último comentar la línea del filter_extremes y cambiar el donde aparece data por data_gramas
    '''
    for i in range(len(data)):
        for word in bigram[data[i]]:
            if '_' in word:
                data[i].append(word)
        for word in trigram[data[i]]:
            if '_' in word:
                data[i].append(word)

    # Quitar palabras comunes y poco comunes
    dictionary = corpora.Dictionary(data)
    dictionary.filter_extremes(no_below=5, no_above=0.70)

    # BOW
    corpus = [dictionary.doc2bow(text) for text in data]

    # Creamos el vector tf-idf
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    return corpus_tfidf, dictionary, data


def train_data(corpus, dictionary):
    num_topics = 4  # Numero de topics
    chunksize = 2000  # Tamaño de los bloques de documentos, en este caso es superior lo hace en un solo bloque
    passes = 20  # Epochs
    iterations = 400  # Num de iteraciones que se quieren hacer
    eval_every = None  # Evaluar la perplejidad del modelo

    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,  # Obtenido en el preproceso
        id2word=id2word,  # Obtenido del diccionario
        chunksize=chunksize,  # Parametro modificable en le json
        alpha='auto',  # Calculo automatico de alfa
        eta='auto',  # Calculo automatico de beta
        iterations=iterations,  # Parametro modificable en el json
        num_topics=num_topics,  # Parametro modificalble en el json
        passes=passes,
        eval_every=eval_every,
        random_state=100,  # Para poder aumentar la reproducibilidad
        update_every=1,  # 1 online, 0 batch; online es más rápido para datasets grandes
        per_word_topics=True
    )

    return model, num_topics


def show_results(model, num_topics, corpus):

    top_topics = model.top_topics(corpus)
    pprint(model.print_topics())

    # Cuanta más baja sea la perplejidad, mejor será el modelo
    print("\nPerplexity: ", model.log_perplexity(corpus))

    # Coherencia del modelo
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)


def downloadVisualization(model, corpus, dictionary):
    vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'visualizar_lda.html')
    # Abrir el archivo en el navegador y jugar con el gráfico


# Encontrar el número óptimo de tópicos
def selectBestNumTopics(dictionary, corpus, limit, start, step):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model, n_topics = train_data(corpus, dictionary)
        model_list.append(model)
        top_topics = model.top_topics(corpus)
        avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
        coherence_values.append(avg_topic_coherence)


    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.show()


if __name__ == "__main__":
    corpus, dictionary, data = preprocess_data()
    model, num_topics = train_data(corpus, dictionary)
    show_results(model, num_topics, corpus)
    downloadVisualization(model, corpus, dictionary)
    selectBestNumTopics(dictionary, corpus, 50, start=1, step=6)
