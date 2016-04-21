__author__ = 'thirumal'

import pickle
import os

from com.iit.cs.cs522.emailAnomalyDetection import ngram


class Analysis:

    def __init__(self):
        self.Ngram = ngram.Ngram('/Users/thirumal/Documents/iit/CS522/project/sampledata')

    def compareBigrams(self,bigrams1, bigrams2):
        common=[]
        for grams1 in bigrams1:
            if grams1 in bigrams2:
                common.append(grams1)
        return common

    def sequenceProcess(self):
        tokens = self.Ngram.loadCorpus()
        preprocessedText = self.Ngram.preprocessData(tokens)
        stemmedWords = self.Ngram.stemWords(preprocessedText)
        bigramslist = self.Ngram.createBigrams(stemmedWords)
        trigramsList = self.Ngram.createTrigrams(stemmedWords)
        posTaggedList = self.Ngram.createPOSTagging(stemmedWords)
        self.Ngram.writeToFile(bigramslist, 'bigramList')
        print 'length of bigramlist- %d ', len(bigramslist)
        bigram_freq_dist=self.Ngram.bigramfrequencyDistribution(stemmedWords,bigramslist)
        self.save_obj(bigram_freq_dist,'bigramfreqdict')
        bigram_freq = self.load_obj('bigramfreqdict')

    def save_obj(self,obj, objname):
        with open(os.getcwd()+'/' + objname + '.pkl', 'wb') as f:
            pickle.dump(obj, f)

    def load_obj(self,objname ):
        with open(os.getcwd()+'/' + objname + '.pkl', 'rb') as f:
            return pickle.load(f)

if __name__ == '__main__':
    Analysis().sequenceProcess()
