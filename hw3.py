from string import digits, ascii_lowercase, punctuation
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from tqdm import tqdm
from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse
import numpy as np
import os
import re
import nltk
import json

morph = MorphAnalyzer()
tokenizer = WordPunctTokenizer()
stop = set(stopwords.words("russian"))
count_vectorizer = CountVectorizer()
tf_vectorizer = TfidfVectorizer()
tfidf_vectorizer = TfidfVectorizer()


def preprocessing(text):
    """Очищает текст от стоп-слов, слов на латинице, цифр."""
    t = tokenizer.tokenize(text.lower())
    lemmas = [morph.parse(word)[0].normal_form for word in t
              if word not in punctuation and word not in stop and not set(word).intersection(digits)
              and not set(word).intersection(ascii_lowercase)]
    return " ".join(lemmas)


def making_corpus(path):
    """Составляет корпус с леммами из файлов."""
    texts = []
    lemmas = []
    with open(path, 'r', encoding='utf-8') as f:
        corpus = list(f)[:50000]
    for line in tqdm(corpus):
        answers = json.loads(line)['answers']
        if answers:
            values = np.array(map(int, [i['author_rating']['value'] for i in answers if i != '']))
            answer = answers[np.argmax(values)]['text']
            texts.append(answer)
            lemmas.append(preprocessing(answer))
    return texts, lemmas


def indexating_corpus(corpus, k=2, b=0.75):
    """Преобразует корпус в матрицу Document-Term."""
    count = count_vectorizer.fit_transform(corpus)
    tf = tf_vectorizer.fit_transform(corpus)
    x_idf = tfidf_vectorizer.fit_transform(corpus)
    idf = tfidf_vectorizer.idf_
    len = count.sum(axis=1)
    lens = k * (1 - b + b * len / len.mean())
    matrix = sparse.lil_matrix(tf.shape)
    for i, j in zip(*tf.nonzero()):
        matrix[i, j] = (tf[i, j] * (k + 1) * idf[j])/(tf[i, j] + lens[i])
    return matrix.tocsr()


def indexating_query(query):
    """Преобразует запрос в вектор."""
    return count_vectorizer.transform([query])


def count_bm25(query, corpus):
    """Считает близость по BM25"""
    return corpus.dot(query.T)


def find_docs(query, corpus, answers):
    """Выполняет поиск"""
    lemmas = preprocessing(query)
    if lemmas:
        query_index = indexating_query(lemmas)
        bm25 = count_bm25(query_index, corpus)
        ind = np.argsort(bm25.toarray(), axis=0)
        return np.array(answers)[ind][::-1].squeeze()


def main():
    corpus, lemmas = making_corpus("/content/drive/MyDrive/Colab Notebooks/Инфопоиск/questions_about_love.jsonl")
    matrix = indexating_corpus(lemmas)
    query = input("Что вы хотите узнать? Введите запрос: ")
    while query != 'нет':
        docs = find_docs(query, matrix, corpus)
        print("Выполняется запрос ...")
        print(*docs[:10], sep='\n')
        print("Вы хотите узнать что-то еще? Если нет, напишите - нет")
        query = input("Что вы хотите узнать? Введите запрос: ")


if __name__ == "__main__":
    main()
