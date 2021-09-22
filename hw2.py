import os
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from tqdm import tqdm
from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"  # для удаления латиницы и пунктуации
stopwords_ru = stopwords.words("russian")  # стандартный список стоп-слов
add_stopwords_ru = ["которых", "которые", "твой", "котйоро", "которого",
                    "сих", "ком", "свой", "твоя", "этими", "слишком",
                    "нами", "всему", "будь", "саму", "чаще", "ваше",
                    "сами", "наш", "затем", "еще", "самих", "наши",
                    "ту", "каждое", "мочь", "весь", "этим", "наша",
                    "своих", "оба", "который", "зато", "те", "этих",
                    "вся", "ваш", "такая", "теми", "ею", "которая",
                    "нередко", "каждая", "также", "чему", "собой",
                    "самими", "нем", "вами", "ими", "откуда", "такие",
                    "тому", "та", "очень", "сама", "нему", "алло",
                    "оно", "этому", "кому", "тобой", "таки", "твоё",
                    "каждые", "твои", "мой", "нею", "самим", "ваши",
                    "ваша", "кем", "мои", "однако", "сразу", "свое",
                    "ними", "всё", "неё", "тех", "хотя", "всем", "тобою",
                    "тебе", "одной", "другие", "аааа", "само", "эта",
                    "буду", "самой", "моё", "своей", "такое", "всею",
                    "будут", "своего", "кого", "свои", "мог", "нам",
                    "особенно", "её", "самому", "наше", "кроме", "вообще",
                    "вон", "мною", "никто", "это", "ты", "как", "что",
                    "не", "но", "ааааааа", "аааааау", "аба"]  # дополнительный список стоп-слов
morph = MorphAnalyzer()
vectorizer = TfidfVectorizer(analyzer='word')

def preprocessing(text):
    clean_text = re.sub(patterns, " ", text)
    lemmas = []
    for token in clean_text.split():
        if token not in stopwords_ru:
            token = token.strip()
            lemma = morph.normal_forms(token)[0]
            lemmas.append(lemma)
    for lemma in lemmas:
        if lemma in add_stopwords_ru:
            lemmas.remove(lemma)
    return " ".join(lemmas)

def making_corpus(directory):
    filenames = []
    for path, dirs, files in os.walk(directory):
        for file in files:
            filenames.append(os.path.join(path, file))
    corpus = []
    for file in tqdm(filenames):
        with open(file, encoding="utf-8") as f:
            text = f.read()
        lemmas = preprocessing(text)
        corpus.append(' '.join(lemmas))
    return corpus, filenames

def indexating_corpus(vectorizer, corpus):
    documentterm_matrix = vectorizer.fit_transform(corpus)
    return documentterm_matrix

def indexating_query(vectorizer, query):
    documentterm_matrix = vectorizer.transform([preprocessing(query)]).toarray()
    return documentterm_matrix

def counting_similarity(query_matrix, corpus_matrix):
    cos_sim = cosine_similarity(query_matrix, corpus_matrix)
    return cos_sim

def main():
    corpus, filenames = making_corpus(directory)
    corpus_matrix = indexating_corpus(vectorizer, corpus)
    while True:
        query = input("Введите запрос: ")
        query_matrix = indexating_query(vectorizer, query)
        cos_sim = counting_similarity(corpus_matrix, query_matrix)
        answer = sorted(range(len(cos_sim.flatten())), key=lambda k: cos_sim.flatten()[k], reverse=True)
        for item in answer:
            print(filenames[item])


if __name__ == '__main__':
    main()