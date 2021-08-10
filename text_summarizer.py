# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:24:03 2021

@author: Tohid(Ted) Naseri
"""


#####################################################################################
# ------------------------------- Importing Libraries------------------------------ #
#####################################################################################
from flask import Flask, render_template, request, redirect, flash
from flask_mail import Mail, Message

from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
import math


import matplotlib
matplotlib.use('Agg')# For ploting in flask
import matplotlib.pyplot as plt
from io import BytesIO# For ploting in flask
import base64# For ploting in flask


import seaborn as sns
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request
import bs4 as bs
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize, sent_tokenize
import re

import pandas as pd
import numpy as np
nltk.download('punkt')  # one time execution

# For saving and loading machine learning model
import pickle

#####################################################################################
# ------------------------------- General Functions ------------------------------- #
#####################################################################################
def TextFromUrl(link) -> str:
    """input: link
    output: a string including all the text according to the input URL.
    """
    scraped_data = urllib.request.urlopen(link)
    article = scraped_data.read()
    parsed_article = bs.BeautifulSoup(article, 'lxml')
    paragraphs = parsed_article.find_all('p')
    article_text = ""
    for p in paragraphs:
        article_text += p.text

    # Removing references where are in Square Brackets with numbers:
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    # Removing extra spaces:
    article_text = re.sub(r'\s+', ' ', article_text)
    return article_text


def findThreshold(sentence_scores) -> int:
    """input: a dictionary having score of each sentence
    output: an int showing the theshold value of sentence score
    Threshold: We can use different values for the threshold, as a common practice we use "average score"
    """
    sum_values = 0
    for i in sentence_scores:
        sum_values += sentence_scores[i]

    average = (sum_values/len(sentence_scores))
    return average


def generateSummary(useThreshold, targetNumber, sentences, sentence_scores):
    """There are two options for generating summary:
    1- based on threshold
    2- based on count of sentences defined by user
    Final sentences in the summary will be ordered acoording to the sequence in the original text
    """
    summary_sent_count = 0
    summary = ""
    selected_score = []
    if(useThreshold == True):
        threshold = findThreshold(sentence_scores)
        print(threshold)
        threshold *= 1.96
        print(threshold)
        for i, sent in enumerate(sentences):
            if sentence_scores[i] >= threshold:
                summary += " "+sent
                summary_sent_count += 1
                selected_score.append(sentence_scores[i])
                
                
    elif(useThreshold == False):
        res = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        resLst = [0]*targetNumber
        for i in range(targetNumber):
            resLst[i] = res[i][0]#saving ids of selected sentences
            selected_score.append(res[i][1])#saving scores of selected sentences

        # To get the order of sentence in summary, resLst should be sorted:
        resLst.sort()
        
        
        print(res[:20])
        
        for index in resLst:
            summary += " "+sentences[index]
            summary_sent_count += 1
        # print(summary)
    
    x1 = round(np.min(list(sentence_scores.values())),4)
    x2 = round(np.mean(list(sentence_scores.values())),4)
    x3 = round(np.max(list(sentence_scores.values())),4)
    x4 = round(np.min(list(selected_score)),4)
    x5 = round(np.mean(list(selected_score)),4)
    x6 = round(np.max(list(selected_score)),4)
    
    lstValue = [x1, x2, x3, x4, x5, x6]
    lstVariable = ['min_score', 'avg_score', 'max_score','min_score', 'avg_score', 'max_score']
    lstType = ['Input_Text', 'Input_Text', 'Input_Text', 'Summary', 'Summary','Summary']
    
    stat_df = pd.DataFrame({'Variable':lstVariable, 'Value':lstValue,'Type':lstType})
    
    return summary, summary_sent_count, stat_df


def freqTable(article_text, stemFlag=1) -> dict:
    """
    input: original text
    output: frequency table of words within the txt
    if stemFlag == 1 --> uses stem of words
    if stemFlag == 0 --> uses words directly. Useful for final statistics report

    steps:
    1- cleaning text(special characters, digits, stopwords)
    2- tokenizing text into sentences
    3- building frequency table based on stem of words
    """
    # Removing special characters and digits
    modified_text = re.sub('[^a-zA-Z]', ' ', article_text)
    modified_text = re.sub(r'\s+', ' ', modified_text)
    modified_text = modified_text.lower()

    # Converting Text To Sentences, We have to use the original text since we need fullstop.
    sentence_list = sent_tokenize(article_text)
    stopwords = nltk.corpus.stopwords.words('english')
    ps = PorterStemmer()
    word_freq = {}
    wordLst = word_tokenize(modified_text)
    for word in wordLst:
        if(stemFlag == 1):
            word = ps.stem(word)
        if word not in stopwords:
            if word not in word_freq.keys():
                word_freq[word] = 1
            else:
                word_freq[word] += 1
    return word_freq


def top10word(article_text):
    """input: article_text, frequency table of stem of words
    output: dataframe showing top 10 words along with their frequency
    If frequency table is already prepared, the function will use it. Otherwise, first, it will build frequency table.
    """
    word_freq = freqTable(article_text, stemFlag=0)
    top10_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])
    top10_df.sort_values(by='Frequency', inplace=True,
                         ascending=False, ignore_index=True)
    top10_df = top10_df[:10]
    top10_df.set_index([pd.Index([i for i in range(1, 11)])], inplace=True)

    from IPython.display import display
    display(top10_df.loc[:10, ['Word', 'Frequency']])
    return top10_df

def plotHistogram(sentence_scores):
    """input: a dictionary of score for each sentence
    output: histogram of scores
    """
    scoreLst = list(sentence_scores.values())
    scoreLst = [x*100 for x in scoreLst]# For Visualization purposes
    #plt.hist(scoreLst)
        
    #sns.set_palette("summer")
    #sns.histplot(scoreLst, kde=True, color='red')
    
    fig, ax = plt.subplots(figsize=(5.5,4))
    n, bins, patches = plt.hist(scoreLst, bins=20, facecolor='blue', edgecolor='white', alpha=0.9)
    n = n.astype('int') # it MUST be integer
    # Good old loop. Choose colormap of your taste
    for i in range(len(patches)):
        #x = n[i]/max(n)
        x = (len(n)-1)/len(n) - i/len(n)
        #patches[i].set_facecolor(plt.cm.seismic(x))
        patches[i].set_facecolor(plt.cm.autumn(x))
        patches[i].set_facecolor(plt.cm.winter(x))
        #patches[i].set_facecolor(plt.cm.plasma(x))
        #patches[i].set_facecolor(plt.cm.Spectral(x))
        #patches[i].set_facecolor(plt.cm.hot(x))
    
    # Add title and labels with custom font sizes
    plt.title('Distribution of Sentences Score', fontsize=12)
    plt.xlabel('Score of Sentences (× 100)', fontsize=10)
        
    plt.ylabel('Count of Sentences', fontsize=10)
    
    img1 = BytesIO()
    plt.savefig(img1, format='png', bbox_inches='tight')
    plt.close()
    img1.seek(0)
    return img1
    


def plotBar(stat_df):
    """input: a dataframe including statistics of sentences score for input text and summary
    output: barplot of comparison input text vs summary
    """
    
    fig, ax = plt.subplots(figsize=(5.5,4))
    sns.barplot(x=stat_df['Variable'], y=stat_df['Value'], hue=stat_df['Type'], ax=ax);
    ax.set_title('Sentences Score Input Text vs Summary')
    ax.set_xlabel('Statistics')
    ax.set_ylabel('Score Value');
    ax.tick_params(axis='x', labelrotation=0)
    
    img2 = BytesIO()
    plt.savefig(img2, format='png', bbox_inches='tight')
    plt.close()
    img2.seek(0)
    return img2

def make_bar_plot(X, y, title='Title', xlbl='X_Label', ylbl='Y_Label', xRotation=90, annotation=False):
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(X, y, ax=ax)
    #data.plot(kind='bar', legend = False, ax=ax)
    patches = ax.patches
    n = len(patches)
    if(n==2):
        patches[0].set_facecolor('cornflowerblue')
        patches[1].set_facecolor('orangered')
    else:
        for i in range(n):
            x = (n-1)/n - i/n
            patches[i].set_facecolor(plt.cm.brg(x))
            
    if(annotation == True):
        for p in ax.patches:
            x=p.get_bbox().get_points()[:,0]
            y=p.get_bbox().get_points()[1,1]
            ax.annotate('{:.1f}%'.format(y), (x.mean(), y), ha='center', va='bottom')

    if(xRotation != 0):
        ax.tick_params(axis='x', labelrotation=xRotation)
    
    #x_pos = np.arange(len(df["word"]))
    #plt.xticks(x_pos, df["word"])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlbl, fontweight="bold")
    ax.set_ylabel(ylbl, fontweight="bold")
    #plt.show()
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    return img
    
    
#####################################################################################
# ------------------------------- Word Frequecy Class ------------------------------#
#####################################################################################
class word_frequency:
    def __init__(self, byThreshold=False, byNumber=10):
        """Constructor for word_frequency class
        Two methods are available to choose sentences for the final summary. The user will define.
        Based on threshold--> sentences having score beyond the threshold will be choosed,
        Based on count of sentence (e.i N) --> top N senetnces will be choosed.
        """
        self.useThreshold = byThreshold
        self.targetNumber = byNumber

    def freqTable(self, article_text) -> dict:
        """
        input: original text
        output: frequency table of words within the txt

        steps:
        1- cleaning text(special characters, digits, stopwords)
        2- tokenizing text into sentences
        3- building frequency table based on stem of words
        """
        # Removing special characters and digits
        modified_text = re.sub('[^a-zA-Z]', ' ', article_text)
        modified_text = re.sub(r'\s+', ' ', modified_text)
        modified_text = modified_text.lower()

        # Converting Text To Sentences, We have to use the original text since we need fullstop.
        sentence_list = sent_tokenize(article_text)
        stopwords = nltk.corpus.stopwords.words('english')
        ps = PorterStemmer()
        word_freq = {}
        wordLst = word_tokenize(modified_text)
        for word in wordLst:
            word = ps.stem(word)
            if word not in stopwords:
                if word not in word_freq.keys():
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
        return word_freq

    def scoreSentences(self, sentences, word_freq) -> dict:
        """input: list of sentences and word frequency table
        output: a dictionary having score for each sentence
        """
        ps = PorterStemmer()
        # first, we set all scores as 0 and then we will modify based on calculation.
        sentence_scores = {i: 0 for i in range(len(sentences))}
        # normalizing word frequency dictionary in a way that maximum frequency get equal to one.
        maxFreq = max(list(word_freq.values()))
        word_freq = {k: v/maxFreq for k, v in word_freq.items()}
        for i, sent in enumerate(sentences):
            wordLst = word_tokenize(sent.lower())
            count = 0
            for word in wordLst:
                word = ps.stem(word)
                if word in word_freq:
                    count += 1
                    sentence_scores[i] += word_freq[word]
            if count != 0:
                sentence_scores[i] = round(sentence_scores[i]/count, 4)
        return sentence_scores

    def run_summarization(self, text):
        """The main function of the class which includes 4 steps:
        1- preparing word frequency table
        2- tokenizing sentences
        3- calculating score of each sentences
        4- finally, generating summary
        """
        word_freq = self.freqTable(text)
        # print(word_freq)
        sentences = sent_tokenize(text)
        input_sent_count = len(sentences)
        sentences = [x for x in sentences if len(word_tokenize(x))>3]
        sentence_scores = self.scoreSentences(sentences, word_freq)
        summary, summary_sent_count, stat_df = generateSummary(self.useThreshold, self.targetNumber, sentences, sentence_scores)
        top10_df = top10word(text)
        img1 = plotHistogram(sentence_scores)
        img2 = plotBar(stat_df)
        return summary, input_sent_count, summary_sent_count, sentence_scores, top10_df, img1, img2
    
#####################################################################################
# ------------------------------- Text Rank Class ----------------------------------#
#####################################################################################        
class text_rank:
    def __init__(self, byThreshold=False, byNumber=10):
        """Constructor of textRank class 
        Two methods (user options) are available to choose sentences for summary.
        Based on threshold--> sentences having score beyond the threshold will be choosed,
        Based on count of sentence (e.i N) --> top N senetnces will be choosed.
        """
        self.useThreshold = byThreshold
        self.targetNumber = byNumber

    def clean(self, txt) -> str:
        """input: a sentence
        output: modified text after cleaning which includes:
        1- removing special characters, numbers
        2- converting text into lowercase
        3- removing stopwords
        """
        # removing special characters, numbers:
        txt = re.sub('[^a-zA-Z]', ' ', txt)
        # managing multiple whitespaces:
        txt = re.sub(r'\s+', ' ', txt)
        txt = txt.lower()

        stopwords = nltk.corpus.stopwords.words('english')
        newTxt = " ".join([x for x in txt.split() if x not in stopwords])
        return newTxt

    def prepareSentences(self, article_text):
        """input: original text
        output: list of cleaned sentences
        To convert text into sentences, we have to use the original text since we need fullstop.
        """
        sentences = sent_tokenize(article_text)
        input_sent_count = len(sentences)
        sentences = [x for x in sentences if len(word_tokenize(x))>3]
        clean_sentences = ['']*len(sentences)
        for i, sent in enumerate(sentences):
            clean_sentences[i] = self.clean(sent)
        return sentences, clean_sentences, input_sent_count

    def extract_word_emedding(self) -> dict:
        # Extract word vectors having 100 collumns for each word
        word_embeddings = {}
        f = open('static/glove/glove.6B.50d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.array([float(x) for x in values[1:]])
            word_embeddings[word] = coefs
        f.close()
        return word_embeddings

    def scoreSentences(self, clean_sentences, word_embeddings) -> dict:
        """input: list of sentences
        output: a dictionary having the score for each sentence
        1- For each sentence, it makes a dictionary having words and the related coefficients
        2- It calculates the cosine similarity
        3- It applies page rank algorithme
        """
        sentences_vectors = []

        for sent in clean_sentences:
            if len(sent) != 0:
                # If word is not available in embedding list, we will set all coefficient as zero.
                notFoundLst = np.zeros((50,))
                coefLst = [word_embeddings.get(
                    word, notFoundLst) for word in sent.split()]
                coefLst = sum([word_embeddings.get(word, np.zeros((50,)))
                               for word in sent.split()])/(len(sent.split()))
            else:
                coefLst = np.zeros((50,))
            sentences_vectors.append(coefLst)

        count = len(clean_sentences)
        sim_mat = np.zeros([count, count])
        for i in range(count):
            for j in range(count):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(sentences_vectors[i].reshape(
                        1, 50), sentences_vectors[j].reshape(1, 50))[0, 0]

        # applying page rank:
        nx_graph = nx.from_numpy_array(sim_mat)
        sentence_scores = nx.pagerank(nx_graph)
        sentence_scores = {k:round(v, 4) for k,v in sentence_scores.items()}
        print(sentence_scores)
        return sentence_scores

    def run_summarization(self, text):
        """The main function of the class which includes 4 steps:
        1- preparing sentences
        2- extracting word embedding
        3- calculating score of each sentences
        4- finally, generating summary
        """
        sentences, clean_sentences, input_sent_count = self.prepareSentences(text)

        word_embeddings = self.extract_word_emedding()
        sentence_scores = self.scoreSentences(clean_sentences, word_embeddings)
        # print(sentence_scores)

        summary, summary_sent_count, stat_df = generateSummary(self.useThreshold, self.targetNumber, sentences, sentence_scores)
        top10_df = top10word(text)
        img1 = plotHistogram(sentence_scores)
        img2 = plotBar(stat_df)
        return summary, input_sent_count, summary_sent_count, sentence_scores, top10_df, img1, img2
    
#####################################################################################
# ------------------------------- TF-IDF Class -------------------------------------#
##################################################################################### 
class tf_idf:

    def __init__(self, byThreshold=False, byNumber=10):
        """Constructor of tf_idf class,
        Two methods (user options) are available to choose sentences for summary.
        Based on threshold--> sentences having score beyond the threshold will be choosed,
        Based on count of sentence (e.i N) --> top N senetnces will be choosed.
        """
        self.useThreshold = byThreshold
        self.targetNumber = byNumber

    def find_freq_matrix(self, sentences)->dict:
        """
        input: list of original sentences in text
        output: frequency matrix in form of dictionary

        freq_matrix:
           key: id of sentence
           value: a dictionary of word frequency in that sentence.
        """
        freq_matrix = {}
        stopWords = set(stopwords.words("english"))
        ps = PorterStemmer()

        for i, sent in enumerate(sentences):
            freq_in_sentence = {}
            words = word_tokenize(sent)
            for word in words:
                word = word.lower()
                word = ps.stem(word)
                if word in stopWords:
                    continue
                elif word in freq_in_sentence:
                    freq_in_sentence[word] += 1
                else:
                    freq_in_sentence[word] = 1
            
            freq_matrix[i] = freq_in_sentence
        return freq_matrix
    
    def find_tf_matrix(self, freq_matrix)->dict:
        """
        input: list of original sentences in text
        output: tf_matrix in form of a dictionary
        tf_matrix:
           key: id of sentence
           value: a dictionary of tf score of each word in that sentence.
        """
        tf_matrix = {}
        for sent, freq_table in freq_matrix.items():
            count_of_word = len(freq_table)
            tf_table = {k: v/count_of_word for k,v in freq_table.items()}
            tf_matrix[sent] = tf_table
        return tf_matrix

    def find_documents_per_words(self, freq_matrix)->dict:
        """input: frequency matrix of words in each sentence
        output: a dictionary that helps the calculation of IDF matrix. It shows how many sentences have a word
        """
        doc_per_word = {}
        for sent, freq_table in freq_matrix.items():
            for word, count in freq_table.items():
                if word in doc_per_word:
                    doc_per_word[word] += 1
                else:
                    doc_per_word[word] = 1
        return doc_per_word   

    def find_idf_matrix(self, freq_matrix, count_doc_per_words, total_documents):
        """input: frequency matrix, doc_per_word dictionary, and count of sentences in original text 
        output: idf_matrix which is a dictionary where:
        key: id of sentence
        value: dictionary of idf score of word in that sentence.
        """
        idf_matrix = {}
        for sent_id, freq_table in freq_matrix.items():
            idf_table = {}
            for word in freq_table.keys():
                idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))
            idf_matrix[sent_id] = idf_table
        return idf_matrix

    def find_tf_idf_matrix(self, tf_matrix, idf_matrix):
        """input: tf_matrix, idf_matrix
        output: tf_idf_matrix which is a dictionary where:
        key: id of sentence
        value: dictionary of word tf_idf score in that sentence
        """
        tf_idf_matrix = {}
        for (sent_id, tf_table), (sent_id, idf_table) in zip(tf_matrix.items(), idf_matrix.items()):
            tf_idf_table = {}
            for (word, tf), (word, idf) in zip(tf_table.items(), idf_table.items()):
                tf_idf_table[word] = tf * idf
            tf_idf_matrix[sent_id] = tf_idf_table
        return tf_idf_matrix

    def scoreSentences(self, tf_idf_matrix) -> dict:
        """Score a sentence by summation of its word's TF_idf score devided by no of words
        (that have tf_idf score in the sentence)."""
        # first, we set all scores as 0 and then we will modify based on calculation.
        sentence_scores = {i: 0 for i in range(len(tf_idf_matrix))}
        for sent_id, tf_idf_table in tf_idf_matrix.items():
            totalScore = 0
            wordCnt = len(tf_idf_table)
            for word, score in tf_idf_table.items():
                totalScore += score

            # Normalizing sentence score:
            if wordCnt != 0:
                sentence_scores[sent_id] = totalScore / wordCnt
        return sentence_scores

    def run_summarization(self, text):
        """main function of the class which includes 7 steps:
        1- preparing sentences
        2- calculating frequency matrix and tf matrix 
        3- calculating documents per word dictionary
        4- calculating idf matrix
        5- calculating tf_idf matrix
        6- calculating the score of each sentences
        7- finally, generating summary
        """
        sentences = sent_tokenize(text)
        input_sent_count = len(sentences)
        sentences = [x for x in sentences if len(word_tokenize(x))>3]
        total_documents = len(sentences)
        freq_matrix = self.find_freq_matrix(sentences)
        tf_matrix = self.find_tf_matrix(freq_matrix)
        count_doc_per_words = self.find_documents_per_words(freq_matrix)
        idf_matrix = self.find_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
        tf_idf_matrix = self.find_tf_idf_matrix(tf_matrix, idf_matrix)
        sentence_scores = self.scoreSentences(tf_idf_matrix)
        summary, summary_sent_count, stat_df = generateSummary(self.useThreshold, self.targetNumber, sentences, sentence_scores)
        top10_df = top10word(text)
        img1 = plotHistogram(sentence_scores)
        img2 = plotBar(stat_df)
        return summary, input_sent_count, summary_sent_count, sentence_scores, top10_df, img1, img2

    
#####################################################################################
# ------------------------------- Lex Rank Class -----------------------------------#
##################################################################################### 
class lex_rank:
    def __init__(self, targetNumber=10):
        """Constructor of lex_rank class,
        User can define number of sentences in the summary.
        Otherwise, as a default, 10 sentences wil be returned in form of summary.
        """
        self.targetNumber = targetNumber
    
    def run_summarization(self, text):
        """input: original text
        output: summary of the original text using LexRankSummarizer from sumy library
        In addition, top 10 stem of words will be calculated.
        """
        # The summarizer will apply cleaning by itself. We do not need cleaning.
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        lexRankSummary = summarizer(parser.document, self.targetNumber) 
        summary = ''
        for sent in lexRankSummary:
            summary += str(sent) + ' '
        top10_df = top10word(text)
        input_sent_count = len(sent_tokenize(text))
        summary_sent_count = self.targetNumber
        return summary, input_sent_count, summary_sent_count, top10_df
    
     
#####################################################################################
# ------------------------------- Flask Application --------------------------------#
#####################################################################################
app = Flask(__name__,static_folder="static")

@app.route('/wordFreq', methods=['GET','POST'])
def wordFreq():
    summarizerName = 'Word Frequency'
    if request.method == 'POST':
        if request.form['btn'] == 'Submit':
            errorFlag=0
            try:
                link = request.form['input_url']
                article_text = TextFromUrl(link)
            except:
                flash('Error: Article text not found. Please enter another URL!')
                errorFlag = 1
                return render_template('wordFreq.html', original_text = " ", type_summarizer = summarizerName)
            if(errorFlag == 0):
                summaryLength = request.form['length']
                if(summaryLength=="threshold"):
                    obj1 = word_frequency(True, 1)
                else:
                    summaryLength = int(summaryLength)
                    obj1 = word_frequency(False, summaryLength)
                summary, input_sent_count, summary_sent_count, sentence_scores, top10_df, img1, img2 = obj1.run_summarization(article_text)

                img1.seek(0)
                plot_url1 = base64.b64encode(img1.getvalue()).decode('utf8')
                img2.seek(0)
                plot_url2 = base64.b64encode(img2.getvalue()).decode('utf8')
                
                return render_template('wordFreq.html', scroll='reportAnchor', original_text = article_text,
                                       output_summary = summary,
                                       type_summarizer = summarizerName,
                                       input_sent_count = input_sent_count,
                                       summary_sent_count = summary_sent_count,
                                       plot_url1=plot_url1,
                                       plot_url2=plot_url2,
                                       column_names=top10_df.columns.values, 
                                       row_data=list(top10_df.values.tolist()),
                                       zip=zip)
        
        elif request.form['btn'] == 'Reset':
            return render_template('wordFreq.html', original_text = " ", output_summary =" ", type_summarizer = summarizerName)
    else:
        return render_template('wordFreq.html', original_text = " ", type_summarizer = summarizerName)
    
@app.route('/tf-idf', methods=['GET','POST'])
def tfidf():
    summarizerName = 'TF-IDF'
    if request.method == 'POST':
        if request.form['btn'] == 'Submit':
            errorFlag=0
            try:
                link = request.form['input_url']
                article_text = TextFromUrl(link)
            except:
                flash('Error: Article text not found. Please enter another URL!')
                errorFlag = 1
                return render_template('tf-idf.html', original_text = " ", type_summarizer = summarizerName)
            if(errorFlag == 0):
                summaryLength = request.form['length']
                if(summaryLength=="threshold"):
                    obj2 = tf_idf(True, 1)
                else:
                    summaryLength = int(summaryLength)
                    obj2 = tf_idf(False, summaryLength)
                summary, input_sent_count, summary_sent_count, sentence_scores, top10_df, img1, img2 = obj2.run_summarization(article_text)

                img1.seek(0)
                plot_url1 = base64.b64encode(img1.getvalue()).decode('utf8')
                img2.seek(0)
                plot_url2 = base64.b64encode(img2.getvalue()).decode('utf8')
                
                return render_template('tf-idf.html', scroll='reportAnchor', original_text = article_text,
                                       output_summary = summary,
                                       type_summarizer = summarizerName,
                                       input_sent_count = input_sent_count,
                                       summary_sent_count = summary_sent_count,
                                       plot_url1=plot_url1,
                                       plot_url2=plot_url2,
                                       column_names=top10_df.columns.values, 
                                       row_data=list(top10_df.values.tolist()),
                                       zip=zip)
        
        elif request.form['btn'] == 'Reset':
            return render_template('tf-idf.html', original_text = " ", output_summary =" ", type_summarizer = summarizerName)
    else:
        return render_template('tf-idf.html', original_text = " ", type_summarizer = summarizerName)

    
@app.route('/lexRank', methods=['GET','POST'])
def lexRank():
    summarizerName = 'LexRank'
    if request.method == 'POST':
        if request.form['btn'] == 'Submit':
            errorFlag=0
            try:
                link = request.form['input_url']
                article_text = TextFromUrl(link)
            except:
                flash('Error: Article text not found. Please enter another URL!')
                errorFlag = 1
                return render_template('lexRank.html', original_text = " ", type_summarizer = summarizerName)
            if(errorFlag == 0):
                summaryLength = request.form['length']
                summaryLength = int(summaryLength)
                obj3 = lex_rank(summaryLength)
                summary, input_sent_count, summary_sent_count, top10_df = obj3.run_summarization(article_text)

                return render_template('lexRank.html', scroll='reportAnchor', original_text = article_text,
                                       output_summary = summary,
                                       type_summarizer = summarizerName,
                                       input_sent_count = input_sent_count,
                                       summary_sent_count = summary_sent_count,
                                       column_names=top10_df.columns.values, 
                                       row_data=list(top10_df.values.tolist()),
                                       zip=zip)
        
        elif request.form['btn'] == 'Reset':
            return render_template('lexRank.html', original_text = " ", output_summary =" ", type_summarizer = summarizerName)
    else:
        return render_template('lexRank.html', original_text = " ", type_summarizer = summarizerName)
 
    
@app.route('/textRank', methods=['GET','POST'])
def textRank():
    summarizerName = 'TextRank'
    if request.method == 'POST':
        if request.form['btn'] == 'Submit':
            errorFlag=0
            try:
                link = request.form['input_url']
                article_text = TextFromUrl(link)
            except:
                flash('Error: Article text not found. Please enter another URL!')
                errorFlag = 1
                return render_template('textRank.html', original_text = " ", type_summarizer = summarizerName)
            if(errorFlag == 0):
                spaceLimitation = 1
                if(spaceLimitation == 1):
                    messageText ="""Message: In this method to evaluate
                    the score of each sentence an embedding table of 
                    size 330 megabyte is required. Due to the online 
                    space limitation this method has been diactivated. 
                    However, the source code is available on the github.""" 
                    flash(messageText)
                    return render_template('textRank.html', scroll='reportAnchor', original_text = " ", type_summarizer = summarizerName)
                else:
                    summaryLength = request.form['length']
                    if(summaryLength=="threshold"):
                        obj4 = text_rank(True, 1)
                    else:
                        summaryLength = int(summaryLength)
                        obj4 = text_rank(False, summaryLength)
                    summary, input_sent_count, summary_sent_count, sentence_scores, top10_df, img1, img2 = obj4.run_summarization(article_text)
    
                    img1.seek(0)
                    plot_url1 = base64.b64encode(img1.getvalue()).decode('utf8')
                    img2.seek(0)
                    plot_url2 = base64.b64encode(img2.getvalue()).decode('utf8')
                    
                    return render_template('textRank.html', scroll='reportAnchor', original_text = article_text,
                                           output_summary = summary,
                                           type_summarizer = summarizerName,
                                           input_sent_count = input_sent_count,
                                           summary_sent_count = summary_sent_count,
                                           plot_url1=plot_url1,
                                           plot_url2=plot_url2,
                                           column_names=top10_df.columns.values, 
                                           row_data=list(top10_df.values.tolist()),
                                           zip=zip)
        
        elif request.form['btn'] == 'Reset':
            return render_template('textRank.html', original_text = " ", output_summary =" ", type_summarizer = summarizerName)
    else:
        return render_template('textRank.html', original_text = " ", type_summarizer = summarizerName)


@app.route('/text_summary', methods=['GET','POST'])
def text_summary():
    if request.method == 'POST':
        return render_template('text_summary.html')
    else:
        return render_template('text_summary.html')
       

@app.route('/')
def home_page():
    title = 'Ted Home Page'
    return render_template('index.html', title=title)

if __name__ == '__main__':
    
    app.debug = True
    #app.run()
    app.run(debug=True, use_reloader=False)