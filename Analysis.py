__author__ = 'thirumal'

import pickle
import os

import ngram


class Analysis:
    def __init__(self):
        self.Ngram = ngram.Ngram('/Users/thirumal/Documents/iit/CS522/project/raw.en')

    def compareBigrams(self, bigrams1, bigrams2):
        common = []
        for grams1 in bigrams1:
            if grams1 in bigrams2:
                common.append(grams1)
        return common

    def sequenceProcess(self):
        '''
        this method is used as a driver that calls different methods for processing data
        :return: void
        '''
        tokens = self.Ngram.loadCorpus()
        preprocessedText = self.Ngram.preprocessData(tokens)
        stemmedWords = self.Ngram.stemWords(preprocessedText)
        bigramslist = self.Ngram.createBigrams(stemmedWords)
        self.Ngram.writeToFile(bigramslist, 'bigramListFile')
        trigramsList = self.Ngram.createTrigrams(stemmedWords)
        self.Ngram.writeToFile(trigramsList, 'trigramListFile')
        self.save_obj(bigramslist, 'bigramsList')
        self.save_obj(trigramsList, 'trigramsList')
        posTaggedList = self.Ngram.createPOSTagging(stemmedWords)
        print 'length of bigramlist- %d ', len(bigramslist)
        bigram_freq_dist = self.Ngram.bigramfrequencyDistribution(stemmedWords, bigramslist)
        self.Ngram.writeDictToFile(bigram_freq_dist,'bigramFreqDict')
        #self.save_obj(bigram_freq_dist, 'bigramfreqdict')
        #bigram_freq = self.load_obj('bigramfreqdict')

    def save_obj(self, obj, objname):
        '''
        this method dumps the python object in a .pkl file
        :param obj: python object that needs to be written to a file
        :param objname: name that will be used to create a pkl file
        :return: void
        '''
        with open(os.getcwd() + '/' + objname + '.pkl', 'wb') as f:
            pickle.dump(obj, f)

    def load_obj(self, objname):
        '''
        retrieves a pkl file and converts back into a python object
        :param objname: name of the pkl file
        :return: pkl file content in the form of a python object
        '''
        with open(os.getcwd() + '/' + objname + '.pkl', 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    Analysis().sequenceProcess()
