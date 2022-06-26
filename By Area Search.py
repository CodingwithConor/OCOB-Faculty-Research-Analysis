# import necessary libraries
import pandas as pd
import sklearn as sk
import itertools
import glob
import math
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import operator
import csv

# create list of tokens from text
def token_gen(text):
    # initialize list
    new_document = []

    # introduce lemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()

    # create tokens
    tokens = word_tokenize(text)

    # lower tokens
    tokens = [w.lower() for w in tokens]

    # pre-processing
    table = str.maketrans("", "", string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    for w in words:
        root_word = wordnet_lemmatizer.lemmatize(w)
        new_document.append(root_word)

    return(new_document)

# function to create dictionaries
def dict_formatter(text):
    word_dict = dict.fromkeys(total, 0)
    for word in text:
        word_dict[word] += 1
    dict_list.append(word_dict)
    return word_dict

# Now writing the TF function:
def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict

# creating the log portion of the Excel table we saw earlier
def computeIDF(docList):
    idfDict = {}
    N = len(docList)

    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))

    return idfDict

# The actual calculation of TF*IDF from the table above:
def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val * idfs[word]
    return tfidf

# sort each dictionary of tfidfs by biggest
def sorter(idf):
    sorted_tuples_x = sorted(idf.items(), key=operator.itemgetter(1), reverse = True)
    sorted_dict_x = {k: v for k, v in sorted_tuples_x}
    sliced_dict = dict(itertools.islice(sorted_dict_x.items(), 25))
    # adds top 15 relevant words to a list of all of the top 15's
    listed_words.append(list(sliced_dict.keys()))
    # returns just the 15 most relevant keys per document
    return sliced_dict


# add all top words by paper into a large list
def scrape_top_words(sorted_tfidfs):
    for x in sorted_tfidfs:
        get_list = list(x.keys())
        for y in get_list:
            all_relevant_words.append(y)
    return(all_relevant_words)

# if any words occur more than once, they're common to different papers which is an interesting connection
def duplicate_finder(possible_word_list):
    word_and_count = {}
    for word in possible_word_list:
        occurences = possible_word_list.count(word)
        if occurences >= 4:
            common_words.append(word)
            word_and_count[word] = occurences
    t = dict(sorted(word_and_count.items(), key=operator.itemgetter(1), reverse = True)[:25])
    print(t)
    return(common_words)

def final_breakdown(document_index, top_words):
    title_list = []
    for title in document_index:
        for item in title[1]:
            for word in top_words:
                if word == item:
                    categorized_dicts[word] = title_list.append(title)
                    if title not in title_list:
                        title_list.append(title)
                        highlighted_word_list.append(word)

def topic_generator(word_list, title_index):
    temp_title_list = []
    for title in title_index:
        for element in title[1]:
            if element == "csr":
                final_dict[element] = title[0]
                print(final_dict)


# initialize lists to hold file corpus, doc list, list of dictionaries, and sorted tfidfs
file_corpus = []
doc_list = []
dict_list = []
sorted_tfidfs = []
all_relevant_words = []
count = []
common_words = []
listed_words = []
title_list = []
highlighted_word_list = []
categorized_dicts = {}
list_breakdown = []
document_index = {}
final_dict = {}
counted_words = []

# give path of directory
path_of_the_directory = r"ECON Texts"
text_files = glob.glob(f"{path_of_the_directory}/*.txt")

# write files into dictionary of titles and text
for title in text_files:
    with open(title, "rb") as in_file:
        data = in_file.read().decode("utf-8")
        in_file.close()

        # tokenize doc
        doc = token_gen(data)

        # add docs to list
        doc_list.append(doc)

        # add docs to dictionary to reference titles later
        document_index[title] = doc

        for item in doc:
            file_corpus.append(item)

# join them to remove common duplicate words
total = set(file_corpus)

# fill dict list and compute corpus-wide idf
for doc in doc_list:
    wordDict = dict_formatter(doc)
idfs = computeIDF(dict_list)

# iterate through doc list to calculate tfidf for each doc
for doc in doc_list:
    wordDict = dict_formatter(doc)
    tfx = computeTF(wordDict, doc)
    idfFirst = computeTFIDF(tfx, idfs)
    output = sorter(idfFirst)
    sorted_tfidfs.append(output)
    #print(output)

top_words = scrape_top_words(sorted_tfidfs)
dupes = duplicate_finder(top_words)

# documents with their top 15 words, list of tuples of lists?
documents_zipped = list(zip(text_files,listed_words))

print(documents_zipped)

topics = topic_generator(dupes,documents_zipped)
print(topics)