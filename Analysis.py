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
        self.Ngram = ngram.Ngram('/Users/thirumal/Documents/iit/CS522/project/dtoEmails_TextOnly')
        self.emailbigramList = [line.strip() for line in open("emailbigramListFile.txt", 'r')]
        self.wikibigramList = [line.strip() for line in open("bigramListFile.txt", 'r')]
        self.emailtrigramList = [line.strip() for line in open("emailtrigramListFile.txt", 'r')]
        self.wikitrigramList = [line.strip() for line in open("trigramListFile.txt", 'r')]
        self.wiki_bigram_count = globalconstants.WIKI_TOTAL_BIGRAM_COUNT
        self.email_bigram_count = globalconstants.EMAIL_TOTAL_BIGRAM_COUNT
        self.wiki_trigram_count = globalconstants.WIKI_TOTAL_TRIGRAM_COUNT
        self.email_trigram_count = globalconstants.EMAIL_TOTAL_TRIGRAM_COUNT


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
        tokens = self.Ngram.loadCorpus()
        preprocessedText = self.Ngram.preprocessData(tokens)
        stemmedwords = self.Ngram.stemWords(preprocessedText)
        print datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        #bigramslist = self.Ngram.createBigrams(stemmedwords)
        #print 'bigrams created successfully'
        #print datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        trigramslist = self.Ngram.createTrigrams(stemmedwords)
        print 'trigrams created successfully'
        self.Ngram.writeToFile(trigramslist, 'emailtrigramListFile')
        print 'trigrams written to emailtrigramListFile.txt'
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
        print 'test email bigram'
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
        print 'test email trigram'
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

        a = "Not attending:  Drew Ries, Bob Johansen, Trang Dinh\
George Wasaff\
Kelly Higgason accepted a position to work for EES effective Jan. 1.  Looking\
for a new Asst. Gral. Counsel.\
Kim Rizzi validated her resignation last week.  For HR needs we will make\
contact with Dave Schafer's office.\
Ken Smith left the company last week to pursue new opportunities.  John to\
circulate memo.  Position open in Platforms & Processes.\
Staffing.  Juanita Andrade was the candidate selected for the Sr. Admin. \
Asst. position.  Leticia Flores to be considered for upgrade in Zhang's group.\
People Plan.  Meeting today at 10.30 AM with Bob Reimer and Don Miller.\
KGW will be off the rest of the year.  Derryl to lead next Monday's meeting.\
New floor tech.  For assistance, call the Resolution Center at 3-1411.\
Derryl Cleaveland\
Cynthia Barrows asked Bruce to contact auto makers for options on hybrid\
automobiles.\
DealBench.  e-Commerce conference for construction services will be held this\
week.  Background on infrastructure GSS can provide needed.  John Will to \
contact each team to get input.  Looking to use DealBench on Nuovo Pignone\
Phase V units.  Phil Foster in Italy meeting with transport cos.\
FreeMarkets.  Glen Meaken has started discussions with KGW.\
Sonoco.  Craig and EES to engage Enron on products Sonoco offers to assist on\
improving their operations.  Also looking for assistance to improve their\
procurement strategy.\
Nepco.  Opportunities continue.  Will meet with Greg again this week.  Nepco\
Europe willing to help as well.\
Analysts revised savings methodology.  Met with Rick Buy's group to \
incorporate what they use on origination projects.  Will also take a look at\
a course for analysts on how to look and analyze new deals.\
Outlook migration scheduled for Dec. 18 and 19.  Notes will need to be\
cleaned up.\
John Gillespie\
iBuyIt.  ETS and Steve Kean's organization (HR, Communications, Govmt. \
Affairs, NA) will try it.  Will contact Steve to get point person.  Active\
fronts:  EBS, EES, and Global IT.\
Andersen Consulting Off Site Re:  e-Commerce postponed until after the\
holidays.\
Peregrine.  Will talk with KGW off line.\
Kelly Higgason\
Kathy Clark starts today.\
Finalized agreements with Corestaff and GE Capital.\
Trying to close on Citibank and Cooper Cameron.\
Derryl to take a look at Contract Administration on how to manage the area.\
Calvin Eakins\
Cathy Riley contacted prime suppliers on 2nd tier.\
Progress on mentoring plan.  Formed committee.  Talked with Tony last week.\
Meeting with him and Beth this week Re:  Branding and Mentoring Program.\
Diversity Task Force.  Number one issue on survey is the need to do a better \
job on promoting and hiring and retention of women and minorities.  If anyone \
would like to view the 2000 Diversity Survey results, please stop by Calvin's\
office.  Diversity Task Force will be merged into the Vision & Values Task\
Force.\
Jennifer Medcalf\
Will prepare a trip report on trip to Europe.  Met with Brian Stanley Re:  \
Bringing in some of their spend, John Sheriff (asked for periodic e-mail with\
update), EBS (will set up conference calls), Shirley McCain Re:  Cellular,\
Beth Apollo (coming back to the US), Etol.\
EBS.  Re:  BMC, will contact Brad.\
Sony Electronics.  Final negotiation of confidentiality agreement Re:  Energy\
consumption.\
Compaq.  Meeting scheduled. \
SAP.  Commerce 1 and them had conference call.  Nothing for EBS at this point.\
Sam Kemp.  There might be an opportunity for him to move into GSS or have a\
GSS representation in Europe."

        testemail_list = [a]

        for i in testemail_list:
            testemail = i
            tokens = self.Ngram.createTokens(testemail)

          #  bigrams = self.Ngram.createBigrams(tokens)
          #  bigramlist = []
          #  for b in bigrams:
          #      bigramlist.append(b[0] + ' ' + b[1])
          #  emailbigramfreq = self.computeEmailBigramFrequencyForTestMail(bigramlist)
          #  print 'email bigram frequency distribution'
          #  print emailbigramfreq
          #  wikibigramfreq = self.computeWikiBigramFrequencyForTestMail(bigramlist)
          #  print 'wiki bigram frequency distribution'
          #  print wikibigramfreq
          #  score = self.computeAnomalyScoreForEmail_2grams(wikibigramfreq, emailbigramfreq, bigramlist)
          #  print 'score = ' + str(score)

            trigrams = self.Ngram.createTrigrams(tokens)
            trigramlist = []
            for t in trigrams:
                trigramlist.append(t[0] + ' ' + t[1] + ' ' + t[2])
            emailtrigramfreq = self.computeEmailTrigramFrequencyForTestMail(trigramlist)
            print 'email trigram frequency distribution'
            print emailtrigramfreq
            wikitrigramfreq = self.computeWikiTrigramFrequencyForTestMail(trigramlist)
            print 'wiki trigram frequency distribution'
            print wikitrigramfreq
            score = self.computeAnomalyScoreForEmail_3grams(wikitrigramfreq, emailtrigramfreq, trigramlist)
            print 'score = ' + str(score)
        testemail = "You may already know this, but I wanted to keep you updated.  All new \
commodity (power) business in California has been put on hold.  Yesterday, \
Eric Letke announced to the team of originators that we would be suspending \
new efforts until the regulatory/legislative environment is more solid in \
California."
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
    Analysis().preprocessEmailData()
