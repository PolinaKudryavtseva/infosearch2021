import os
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import numpy as np
from tqdm import tqdm
from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

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
vectorizer = CountVectorizer(analyzer="word")


def preprocessing(file):
    with open(file, encoding="utf-8") as f:
        text = f.read()
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
        else:
            pass
    return lemmas

filenames = []
for path, dirs, files in os.walk("friends-data"):
    for file in files:
        filenames.append(os.path.join(path, file))

corpus = []
for file in tqdm(filenames):
    lemmas = preprocessing(file)
    corpus.append(' '.join(lemmas))
print()

X = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names()

matrix_freq = np.asarray(X.sum(axis=0)).ravel()
matrix_freq_sorted = sorted(list(range(len(matrix_freq))),
                            key=lambda num: matrix_freq[num])[::-1]

print("Самое частотное слово корпуса:")
print(vectorizer.get_feature_names()[matrix_freq_sorted[0]])
print()
print("Самое редкое слово корпуса:")
print(vectorizer.get_feature_names()[matrix_freq_sorted[-1]])
print()

print("Слова, которые есть в каждой серии:")
for i, arr in enumerate(X.T.toarray()):
    if np.all(arr):
        print(vectorizer.get_feature_names()[i])
print()

characters = [["моника", "мон"], ["рэйчел", "рейч"],
             ["чендлер", "чэндлер", "чен"],
             ["фиби", "фибс"], ["росс"],
             ["джоуи", "джои", "джо"]]

characters_dict = {}
for character in characters:
    characters_dict[character[0]] = [vectorizer.vocabulary_.get(name) for name in character
                          if vectorizer.vocabulary_.get(name) is not None]

characters_freq = {}
for key, values in characters_dict.items():
    inplot = [X.T[value][0].toarray().sum() for value in values]
    freq = sum(inplot)
    characters_freq[key] = freq

print("Самый статистически популярный персонаж:")
print(sorted(characters_freq.items(), key=lambda item: item[1], reverse=True)[0][0], "-", sorted(characters_freq.items(), key=lambda item: item[1], reverse=True)[0][1])
print()
