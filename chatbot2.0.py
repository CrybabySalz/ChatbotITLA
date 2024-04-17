# Carlos Salazar | 2021-1932

import nltk
import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def limpiar_consola():
    os.system('cls' if os.name == 'nt' else 'clear')

lemmatizer = WordNetLemmatizer()

with open('BaseConocimiento.txt', 'r') as file:
    knowledge_base = file.read()

sentences = nltk.sent_tokenize(knowledge_base)


stop_words = set(stopwords.words('spanish'))

def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words and word.isalnum()]
    return ' '.join(tokens)

preprocessed_sentences = [preprocess(sentence) for sentence in sentences]


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

def get_response(question):
    try:
        question = preprocess(question)
        question_tfidf = vectorizer.transform([question])

        similarity_scores = cosine_similarity(question_tfidf, tfidf_matrix)

        most_similar_index = np.argmax(similarity_scores)
        max_similarity = similarity_scores[0][most_similar_index]

        if max_similarity < 0.2:
            return f"No entiendo \"{question}\". Por favor, intenta reescribir tu pregunta."

        response = sentences[most_similar_index]
    
        response_cleaned = re.sub(r'\([^)]*\)', '', response).strip()
        
        return response_cleaned
    except:
        return "Lo siento, no tengo información sobre eso."



limpiar_consola()
print("¡Hola! Puedes preguntarme informaciones relacionadas al ITLA. Escribe 'salir' para terminar.")
while True:
    question = input("Tú: ")
    if question.lower() == 'salir':
        print("Chatbot: De nada, vuelta pronto.")
        break
    response = get_response(question)
    print("Respuesta:", response)



