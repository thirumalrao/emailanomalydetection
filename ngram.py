__author__ = 'thirumal'
#-*-coding: utf-8 -*-

import nltk
import os
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from ast import literal_eval
import sys
from nltk.tag.util import untag

class Ngram:

    def __init__(self, path):
        self.path = path


    def loadCorpus(self):
        ''' loads the corpus, reads from all files and writes to a single large file result.txt in working directory.
        :return:a list of tokens for the entire corpus
        '''
        corpus_root = self.path
        rawcontent = PlaintextCorpusReader(corpus_root, ".*")
        reload(sys)
        #sys.setdefaultencoding('utf-8')
        read_files = rawcontent._fileids
        print 'files to be read ' + str(read_files)
        with open("result.txt", "wb") as outfile:
            for f in read_files:
                if(f != '.DS_Store'):
                    with open(os.path.join(self.path + '/' + f), "rb") as infile:
                        outfile.write(infile.read())
        raw = open("result.txt").read()
        #rawwords = nltk.wordpunct_tokenize(raw)
        return raw

    def createTokens(self,emailText):
        '''

        :param emailText: it is a string
        :return: tokens of type list
        '''
        tokens = nltk.wordpunct_tokenize(emailText)
        return tokens


    def preprocessData(self, tokens):
        ''' removes punctuations, stop words in english,
        :param tokens: list of tokens for entire corpus that has to be preprocessed
        :return: preprocessed corpus that is devoid of punctuations and stop words. returns type nltk.text
        '''

        custom_stop_words = ['docid', 'segmentnumber', 'body', 'x-from', 'x-to', 'x-cc', 'x-bcc', 'x-folder',
                             'x-origin', 'x-fileName','monday','tuesday','wednesday','thursday','friday','saturday','sunday',
                             'section','thanks','content-type','x-origin','c-fileName','mime-version','subject','from',
                             'content- transfer-encoding','message-id','enron','com','january','february','march','april',
                             'may','june','july','august','september','october','november','december','enron.com','inc.',
                             'moreover','u.s','henry','news','copyright','mon','tue','wed','thu','fri','sat','sun','pdt','pst','jan','feb',
                             'mar','apr','jun','jul','aug','sep','oct','nov','dec']
        stoppedWords = []
        stop_words = set(stopwords.words("english"))
        for w in custom_stop_words:
            stop_words.add(w)
        stop_words.update()
        print stop_words
        for token in tokens:
            if (token.lower() not in stop_words):
                t = self.ensure_unicode(token)
                stoppedWords.append(token)


        words = [w for w in stoppedWords if w.isalpha()]
        words_new = [w for w in words if len(w) > 2]
        text = nltk.Text(words_new)
        print 'total words after removing stop words:' + str(len(text))
        return text

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
            v = v.decode('latin-1')
        return unicode(v)  # convert anything not a string to unicode too

    def createBigrams(self,tokens):
        custom_bigrams = list(nltk.bigrams(tokens))
        return custom_bigrams


    def createTrigrams(self,tokens):
        custom_trigrams = list(nltk.trigrams(tokens))
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

    def createPOSTagging(self, tokens):
        posTagged = nltk.pos_tag(tokens)
        print ('POS tagging completed successfully')
        #print (posTagged)
        return posTagged

    def removeProperNouns(self,postaggedList):
        '''
        removes proper nouns
        :param postaggedList: this is a list of tuples. Example - [('Michael', 'NNP'), ('Jackson', 'NNP')]
        :return: list of tuples which are not NNP or NNPS
        '''
        listWithoutNouns = [word for word,pos in postaggedList if pos not in ['NNP','NNPS']]
        return listWithoutNouns
