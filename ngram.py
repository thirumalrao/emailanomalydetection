__author__ = 'thirumal'
#-*-coding: utf-8 -*-

import nltk
import os
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from ast import literal_eval
import Analysis
from Analysis import Analysis
from nltk import FreqDist
import matplotlib.pyplot
from matplotlib import pyplot

class Ngram:

    def __init__(self, path):
        self.path = path


    def loadCorpus(self):
        ''' loads the corpus, reads from all files and writes to a single large file result.txt in working directory.
        :return:a list of tokens for the entire corpus
        '''
        corpus_root = self.path
        rawcontent = PlaintextCorpusReader(corpus_root, ".*")

        read_files = rawcontent._fileids
        print 'files to be read ' + str(read_files)

        with open("result.txt", "wb") as outfile:
            for f in read_files:
                with open(os.path.join(self.path + '/' + f), "rb") as infile:
                    outfile.write(infile.read())
        raw = open("result.txt").read()
        rawwords = nltk.wordpunct_tokenize(raw)
        return rawwords



    def preprocessData(self, tokens):
        ''' removes punctuations, stop words in english,
        :param tokens: list of tokens for entire corpus that has to be preprocessed
        :return: preprocessed corpus that is devoid of punctuations and stop words. returns type nltk.text
        '''

        custom_stop_words = ['docID', 'segmentNumber', 'Body', 'X-From', 'X-To', 'X-cc', 'X-bcc', 'X-Folder',
                             'X-Origin', 'X-FileName']
        stoppedWords = []
        stop_words = set(stopwords.words("english"))
        for w in custom_stop_words:
            stop_words.add(w)
        stop_words.update()
        for token in tokens:
            if token not in (stop_words):
                stoppedWords.append(token)

        text = nltk.Text(stoppedWords)
        #words = [w.lower() for w in text if w.isalpha()]
        words = [w for w in text if len(w) > 2]
        print 'total words after removing stop words:' + str(len(words))
        Analysis().save_obj(words,'preprocessedData')
        return words

    def stemWords(self,text):
        '''
        this method stems the words in the corpus
        :param text: a list of all words after preprocessing is done
        :return:list of stemmed words
        '''
        ps = PorterStemmer()
        stemmedWords=[]

        for w in text:
            t = self.ensure_unicode(w)
            stemmedWords.append(ps.stem(t))

        return stemmedWords

    def ensure_unicode(self,v):
        if isinstance(v, str):
            v = v.decode('utf8')
        return unicode(v)  # convert anything not a string to unicode too

    def createBigrams(self,tokens):
        custom_bigrams = list(nltk.bigrams(tokens))
        print 'bigrams created successfully'
        return custom_bigrams


    def createTrigrams(self,tokens):
        custom_trigrams = list(nltk.trigrams(tokens))
        print 'trigrams created successfully'
        return custom_trigrams

    def createPOSTagging(self, tokens):
        posTaggedList = []
        try:
            for i in tokens:
                words = nltk.word_tokenize(i)
                posTagged = nltk.pos_tag(words)
                posTaggedList.append(posTagged)
        except Exception as e:
            print(str(e))
        print 'POS tagging completed successfully'
        return posTaggedList

    def bigramfrequencyDistribution(self,tokens,custom_bigrams):
        ''' will find frequency distribution for each bigram in the corpus and write to a file
        :param tokens: words of corpus in type tokens
        :param custom_bigrams: bigrams in type list
        :return:void
        '''

        uniqWords = sorted(set(tokens)) # Calculating unique words.
        print(len(uniqWords))
        all_words_Freq = nltk.FreqDist(uniqWords)
        bigrams_freq = nltk.FreqDist(custom_bigrams)
        bigrams_freq_dict={}
        for k,v in bigrams_freq.items():
            bigrams_freq_dict[k]=v
            #print(k,v)
            
        return bigrams_freq_dict


    def writeToFile(self,text,filename):
        '''
        :param text: this is a list of strings that needs to be written to the file
        :param filename: name of the file to be created.
        :return: NA
        '''
        filename = filename + '.txt'
        fileobject = open(filename, "wb")
        for line in text:
            fileobject.write(' '.join(str(s) for s in line) + '\n')
        fileobject.close()

    def writeDictToFile(self,dict,filename):
        with open(filename+'.txt','w') as f:
            f.write(dict)

    def readDictFromFile(self,dict,filename):
        with open(filename+'.txt','r') as f:
            return literal_eval(f.read())
            
    def plot_freq_dist_graphs(self, all_words_Freq, bigrams_freq):
        FreqDist(all_words_Freq).plot(50, cumulative=False)
        FreqDist(bigrams_freq).plot(50, cumulative=False)
        FreqDist(trigrams_freq).plot(50, cumulative=False)
