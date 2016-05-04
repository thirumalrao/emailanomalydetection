__author__ = 'thirumal'

import pickle
import os
from datetime import datetime
import ngram
from gensim.models import Word2Vec
import logging
import globalconstants
import statistics
import email
from email.parser import Parser
import nltk
import itertools
import csv
import sys


class Analysis:
    def __init__(self):
        self.Ngram = ngram.Ngram('/Users/thirumal/Documents/iit/CS522/project/emailsampledata')
        self.emailbigramList = [line.strip() for line in open("emailbigramList.txt", 'r')]
        self.wikibigramList = [line.strip() for line in open("bigramListFile.txt", 'r')]
        self.emailtrigramList = [line.strip() for line in open("emailtrigramList.txt", 'r')]
        self.wikitrigramList = [line.strip() for line in open("trigramListFile.txt", 'r')]
        self.wiki_bigram_count = globalconstants.WIKI_TOTAL_BIGRAM_COUNT
        self.email_bigram_count = globalconstants.EMAIL_TOTAL_BIGRAM_COUNT
        self.wiki_trigram_count = globalconstants.WIKI_TOTAL_TRIGRAM_COUNT
        self.email_trigram_count = globalconstants.EMAIL_TOTAL_TRIGRAM_COUNT
        logger = logging.getLogger(__name__)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
        logging.root.setLevel(level=logging.INFO)
        logger.info("running %s" % ' '.join(sys.argv))



    def compareBigrams(self, bigrams1, bigrams2):
        common = []
        for grams1 in bigrams1:
            if grams1 in bigrams2:
                common.append(grams1)
        return common

    def preprocessEmailData(self):
        '''
        this method is used as a driver that calls different methods for processing data
        :return: void
        '''
        parser = Parser() # Added for extracting only the body of the email
        tokens = self.Ngram.loadCorpus()
        email = parser.parsestr(tokens) # Added for extracting only the body of the email
        email_body_list = [email.split('Body:')[-1] for email in tokens.split('##########################################################')] # Added for extracting only the body of the email
        tokendata=[]
        for txt in email_body_list:
            tokendata.append(nltk.wordpunct_tokenize(txt))
        merged = list(itertools.chain.from_iterable(tokendata)) # flattening the tokendata which is a list of list.
        preprocessedText = self.Ngram.preprocessData(merged)
        stemmedwords = self.Ngram.stemWords(preprocessedText)
        self.Ngram.writeToFile(stemmedwords,'preprocessedEmailBody')
        posTaggedList = self.Ngram.createPOSTagging(stemmedwords)
        preprocessedListwithoutNouns = self.Ngram.removeProperNouns(posTaggedList)
        print 'removed proper nouns'
        print datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        bigramslist = self.Ngram.createBigrams(preprocessedListwithoutNouns)
        print 'bigrams created successfully'
        print datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        trigramslist = self.Ngram.createTrigrams(preprocessedListwithoutNouns)
        print 'trigrams created successfully'
        self.Ngram.writeToFile(bigramslist,'emailbigramList')
        self.Ngram.writeToFile(trigramslist,'emailtrigramList')
        print datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')




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
        bigramslist = self.load_obj('bigramsList')
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

    def computeWikiTrigramFrequencyForTestMail(self, test_email_trigram):
        '''
        this method will compute the frequency distribution of each trigrams of the test mail in wiki corpus.
        :return: dictionary of trigrams and its corresponding frequency count in wiki corpus.
        '''

        wikitrigramfreq = {}
        for testtrigram in test_email_trigram:
            d = self.wikitrigramList.count(testtrigram)
            wikitrigramfreq[testtrigram] = d
        return wikitrigramfreq

    def computeEmailBigramFrequencyForTestMail(self,test_email_bigram):
        '''
        this method will compute the frequency distribution of each bigram of the test mail in wiki corpus.
        :return: dictionary of bigrams and its corresponding frequency count in wiki corpus.
        '''
        emailbigramfreq={}
       # print test_email_bigram
        for testbigram in test_email_bigram:
           # print 'testbigram -- '
           # print testbigram
           # print 'length of email bigram list -- ' + str(len(self.emailbigramList))
            d = self.emailbigramList.count(testbigram)
            emailbigramfreq[testbigram] = d
        return emailbigramfreq

    def computeEmailTrigramFrequencyForTestMail(self, test_email_trigram):
        '''
        this method will compute the frequency distribution of each trigram of the test mail in wiki corpus.
        :return: dictionary of trigrams and its corresponding frequency count in wiki corpus.
        '''
        emailtrigramfreq = {}
        # print test_email_bigram
        for testtrigram in test_email_trigram:
            # print 'testbigram -- '
            # print testbigram
            # print 'length of email bigram list -- ' + str(len(self.emailbigramList))
            d = self.emailtrigramList.count(testtrigram)
            emailtrigramfreq[testtrigram] = d
        return emailtrigramfreq

    def computeAnomalyScoreForEmail_2grams(self, wikibigramfreq, emailbigramfreq, testmailbigrams):
        '''
        will compute anomaly score for an email by comparing bigram distribution in email dataset and wiki corpus
        :param wikibigramfreq: wikibigram frequency for test mail as python dictionary
        :param emailbigramfreq: emailbigram frequency for test mail as python dictionary
        :param testmailbigrams: list of bigrams for the mail whose anomaly score is to be computed
        :return: anomaly score for that email
        '''
        wikiscore = 0
        emailscore = 0
        wiki_count = 0
        email_count = 0
        for test_bigram in testmailbigrams:
            if test_bigram in wikibigramfreq:
                wiki_count = wikibigramfreq[test_bigram]
            if test_bigram in emailbigramfreq:
                email_count = emailbigramfreq[test_bigram]
            wikiscore = wikiscore + (wiki_count * 10000 / self.wiki_bigram_count)
            emailscore = emailscore + (email_count * 10000 / self.email_bigram_count)

        anomalyscore = statistics.stdev([wikiscore, emailscore])
        return anomalyscore

    def computeAnomalyScoreForEmail_3grams(self, wikitrigramfreq, emailtrigramfreq, testmailtrigrams):
        '''
        will compute anomaly score for an email by comparing trigram distribution in email dataset and wiki corpus
        :param wikitrigramfreq: wikitrigram frequency for test mail as python dictionary
        :param emailtrigramfreq: emailtrigram frequency for test mail as python dictionary
        :param testmailtrigrams: list of trigrams for the mail whose anomaly score is to be computed
        :return: anomaly score for that email
        '''
        wikiscore = 0
        emailscore = 0
        wiki_count = 0
        email_count = 0
        for test_trigram in testmailtrigrams:
            if test_trigram in wikitrigramfreq:
                wiki_count = wikitrigramfreq[test_trigram]
            if test_trigram in emailtrigramfreq:
                email_count = emailtrigramfreq[test_trigram]
            wikiscore = wikiscore + (wiki_count * 10000 / self.wiki_trigram_count)
            emailscore = emailscore + (email_count * 10000 / self.email_trigram_count)

        anomalyscore = statistics.stdev([wikiscore, emailscore])
        return anomalyscore

    def tempsequence(self):
        print datetime.now()

        rawtext = self.Ngram.loadCorpus()

        emailbodyDict = self.load_obj('emailBodyDict')
        bigramScoreDict={}
        trigramScoreDict={}
        bigramwriter = csv.writer(open('bigramScore.csv', 'wb'))
        trigramwriter = csv.writer(open('trigramScore.csv', 'wb'))
        for k,v in emailbodyDict.items():
            testemail = v
            tokens = self.Ngram.createTokens(testemail)
            preprocessdata = self.Ngram.preprocessData(tokens)
            bigrams = self.Ngram.createBigrams(preprocessdata)
            bigramlist = []
            for b in bigrams:
                bigramlist.append(b[0] + ' ' + b[1])
            emailbigramfreq = self.computeEmailBigramFrequencyForTestMail(bigramlist)
            #print 'email bigram frequency distribution'
            #print emailbigramfreq
            wikibigramfreq = self.computeWikiBigramFrequencyForTestMail(bigramlist)
            #print 'wiki bigram frequency distribution'
            #print wikibigramfreq
            bigramscore = self.computeAnomalyScoreForEmail_2grams(wikibigramfreq, emailbigramfreq, bigramlist)
            bigramwriter.writerow([k, bigramscore])
            print 'email no. - ' + str(k)
            print 'bigramscore = ' + str(bigramscore)
            bigramScoreDict[k] = bigramscore

            trigrams = self.Ngram.createTrigrams(preprocessdata)
            trigramlist = []
            for t in trigrams:
                trigramlist.append(t[0] + ' ' + t[1] + ' ' + t[2])
            emailtrigramfreq = self.computeEmailTrigramFrequencyForTestMail(trigramlist)
            wikitrigramfreq = self.computeWikiTrigramFrequencyForTestMail(trigramlist)
            trigramscore = self.computeAnomalyScoreForEmail_3grams(wikitrigramfreq, emailtrigramfreq, trigramlist)
            print 'trigramscore = ' + str(trigramscore)
            trigramScoreDict[k] = trigramscore
            trigramwriter.writerow([k,trigramscore])


if __name__ == '__main__':
    Analysis().tempsequence()
