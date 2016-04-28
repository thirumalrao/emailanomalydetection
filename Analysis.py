__author__ = 'thirumal'

import pickle
import os
from datetime import datetime
import ngram
from gensim.models import Word2Vec
import logging
import globalconstants
import statistics


class Analysis:
    def __init__(self):
        self.Ngram = ngram.Ngram('/Users/thirumal/Documents/iit/CS522/project/sampledata')
        self.emailbigramList = [line.strip() for line in open("emailbigramListFile.txt", 'r')]
        self.wikibigramList = [line.strip() for line in open("bigramListFile.txt", 'r')]
        self.wiki_bigram_count = globalconstants.WIKI_TOTAL_BIGRAM_COUNT
        self.email_bigram_count = globalconstants.EMAIL_TOTAL_BIGRAM_COUNT


    def compareBigrams(self, bigrams1, bigrams2):
        common = []
        for grams1 in bigrams1:
            if grams1 in bigrams2:
                common.append(grams1)
        return common

    def processDataAndCreateNgrams(self):
        '''
        this method is used as a driver that calls different methods for processing data
        :return: void
        '''
        tokens = self.Ngram.loadCorpus()
        preprocessedText = self.Ngram.preprocessData(tokens)
        print 'loaded preprocessedData.pkl successfully'
        print type(preprocessedText)
        stemmedWords = self.Ngram.stemWords(preprocessedText)
        print 'stemming is successful, saving it in pickle now...'
        self.save_obj(preprocessedText,'stemmedWords')
        print 'saved in stemmedWords.pkl successfully'
        words = self.load_obj('stemmedWords')
        bigramslist = self.Ngram.createBigrams(words)
        self.Ngram.writeToFile(bigramslist, 'bigramListFile')
        trigramsList = self.Ngram.createTrigrams(words)
        self.Ngram.writeToFile(trigramsList, 'trigramListFile')
        self.save_obj(bigramslist, 'bigramsList')
        self.save_obj(trigramsList, 'trigramsList')
        posTaggedList = self.Ngram.createPOSTagging(stemmedWords)
        print 'length of bigramlist- %d ', len(bigramslist)
        bigramslist = self.load_obj('bigramsList')
        bigram_freq_dist = self.Ngram.bigramfrequencyDistribution(words, bigramslist)
        self.Ngram.writeDictToFile(bigram_freq_dist,'bigramFreqDict')
        self.save_obj(bigram_freq_dist, 'bigramfreqdict')
        bigram_freq = self.load_obj('bigramfreqdict')

    def processDataforWikiCorpus(self):
        '''
        this method is used as a driver that calls different methods for processing data
        :return: void
        '''
        print datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        tokens = self.Ngram.loadCorpus()
        preprocessedText = self.Ngram.preprocessData(tokens)
        self.save_obj(preprocessedText,'preprocessedWords')
        print 'saved preprocessedData.pkl successfully'
        print datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        #words = self.load_obj('preprocessedWords')
        words = preprocessedText
        bigramslist = self.Ngram.createBigrams(words)
        print 'bigrams created successfully'
        print datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        self.Ngram.writeToFile(bigramslist, 'bigramListFile')
        print 'bigrams written to file successfully'
        print datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        trigramsList = self.Ngram.createTrigrams(words)
        print 'trigrams created successfully'
        print datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        self.Ngram.writeToFile(trigramsList, 'trigramListFile')
        print 'trigrams written to file successfully'
        self.save_obj(bigramslist, 'bigramsList')
        self.save_obj(trigramsList, 'trigramsList')
        print 'bigram and trigram saved as pkl file successfully'
        print datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        #bigramslist = self.load_obj('bigramsList')
        print 'bigram Frequency distribution computation will be started...'
        bigram_freq_dist = self.Ngram.bigramfrequencyDistribution(words, bigramslist)
        print 'bigram frequency distribution completed successfully'
        print datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        self.Ngram.writeDictToFile(bigram_freq_dist,'bigramFreqDict')
        print 'bigram frequency distribution written to file succesfully '
        self.save_obj(bigram_freq_dist, 'bigramfreqdict')
        print 'bigram frequency distribution saved as pkl file'
        #bigram_freq = self.load_obj('bigramfreqdict')
        print datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print '--------------wiki corpus processing completed -------------------'


    def createModel(self):
        raw_sentences = self.Ngram.loadCorpus()
        #raw_sentences = [['first', 'sentence'], ['second', 'sentence']]
        print 'sentence tokenizing successful'
        self.save_obj(raw_sentences,'wikiSentenceTokens')
        print 'wikiSentenceTokens saved as pkl file successfully'
        raw_sentences=self.load_obj('wikiSentenceTokens')
        model = Word2Vec(raw_sentences, size=750, window=5, min_count=1, workers=5)
        model_name = "750features_40minwords_20context"
        #model.save(model_name)
        #new_model = Word2Vec.load(model_name)
        print 'total sentences:' + str(len(raw_sentences))
        b = model.most_similar(positive=['California'], negative=['man'])
        #a = model.doesnt_match('California')
        a = model.doesnt_match("breakfast cereal dinner lunch".split())


    def save_obj(self, obj, objname):
        '''
        this method dumps the python object in a .pkl file
        :param obj: python object that needs to be written to a file
        :param objname: name that will be used to create a pkl file
        :return: void
        '''
        with open(os.getcwd() + '/' + objname + '.pkl', 'wb') as f:
            pickle.dump(obj, f)
        f.close()

    def load_obj(self, objname):
        '''
        retrieves a pkl file and converts back into a python object
        :param objname: name of the pkl file
        :return: pkl file content in the form of a python object
        '''
        with open(os.getcwd() + '/' + objname + '.pkl', 'rb') as f:
            return pickle.load(f)

    def createModeltest(self):
        from gensim.models import word2vec
        sentences = word2vec.Text8Corpus('result.txt')
        model = word2vec.Word2Vec(sentences, size=200)
        model_name = "750features_40minwords_20context"
        model.save(model_name)
        new_model = Word2Vec.load(model_name)
        a = new_model.most_similar(positive=['Utilities', 'California'], negative=['man'], topn=1)
        print a
        b = model.doesnt_match("breakfast cereal dinner lunch".split())
        print b

    def computeWikiBigramFrequencyForTestMail(self,test_email_bigram):
        '''
        this method will compute the frequency distribution of each bigram of the test mail in wiki corpus.
        :return: dictionary of bigrams and its corresponding frequency count in wiki corpus.
        '''

        wikibigramfreq={}
        for testbigram in test_email_bigram:
            d = self.wikibigramList.count(testbigram)
            wikibigramfreq[testbigram] = d
        return wikibigramfreq

    def computeEmailBigramFrequencyForTestMail(self,test_email_bigram):
        '''
        this method will compute the frequency distribution of each bigram of the test mail in wiki corpus.
        :return: dictionary of bigrams and its corresponding frequency count in wiki corpus.
        '''
        emailbigramfreq={}
        print 'test email bigram'
        print test_email_bigram
        for testbigram in test_email_bigram:
            print 'testbigram -- '
            print testbigram
            print 'length of email bigram list -- ' + str(len(self.emailbigramList))
            d = self.emailbigramList.count(testbigram)
            emailbigramfreq[testbigram] = d
        return emailbigramfreq

    def computeAnomalyScoreForEmail(self,wikibigramfreq, emailbigramfreq, testmailbigrams):
        '''
        will compute anomaly score for an email by comparing bigram distribution in email dataset and wiki corpus
        :param wikibigramfreq: wikibigram frequency for test mail as python dictionary
        :param emailbigramfreq: emailbigram frequency for test mail as python dictionary
        :param testmailbigrams: list of bigrams for the mail whose anomaly score is to be computed
        :return: anomaly score for that email
        '''
        wikiscore=0
        emailscore=0
        wiki_count=0
        email_count=0
        for test_bigram in testmailbigrams:
            if test_bigram in wikibigramfreq:
                wiki_count = wikibigramfreq[test_bigram]
            if test_bigram in emailbigramfreq:
                email_count = emailbigramfreq[test_bigram]
            wikiscore = wikiscore + (wiki_count*10000/self.wiki_bigram_count)
            emailscore = emailscore + (email_count*10000/self.email_bigram_count)

        anomalyscore = statistics.stdev([wikiscore,emailscore])
        return anomalyscore

    def tempsequence(self):
        print datetime.now()
        testemail = "min max"
        tokens = self.Ngram.createTokens(testemail)
        bigrams = self.Ngram.createBigrams(tokens)
        bigramlist=[]
        for b in bigrams:
            bigramlist.append(b[0]+' '+ b[1])
        emailbigramfreq = self.computeEmailBigramFrequencyForTestMail(bigramlist)
        print 'email bigram frequency distribution'
        print emailbigramfreq
        wikibigramfreq = self.computeWikiBigramFrequencyForTestMail(bigramlist)
        print 'wiki bigram frequency distribution'
        print wikibigramfreq
        score = self.computeAnomalyScoreForEmail(wikibigramfreq,emailbigramfreq,bigramlist)
        print 'score = ' + str(score)
        print datetime.now()

if __name__ == '__main__':
    Analysis().tempsequence()
