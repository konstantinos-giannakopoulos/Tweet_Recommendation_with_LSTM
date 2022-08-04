#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 12:01:57 2018

@author: konstantinos
"""

import filterInput
import process
import model

import glob
import xml.etree.ElementTree as ET
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
import time
import sys

class XMLReader(): 
    def __init__(self):
        self.stopwords = []
        self.stopwords = self.readStopwordsFile()
        
    def getColumn(self,matrix, i):
        return [row[i] for row in matrix]
    
    # Step 1b: 
    def locationsMapping(self,locations):
        # Read map-locations time
        mapChineseEnglishLocations = []
        with open('util/locmap.txt') as f:
            for line in f:
                line = line.strip()
                content = line.split(":")
                mapChineseEnglishLocations.append(content)
        chineselist = self.getColumn(mapChineseEnglishLocations,0)
        englishlist = self.getColumn(mapChineseEnglishLocations,1)
        #print(chineselist[0]) 
        #print(englishlist[0])
        #return mapChineseEnglishLocations
        # replace Chinese names with English names
        current = 0
        while current < len(locations):
            location = locations[current]
            if(location in chineselist):
                index = chineselist.index(location)
                locations[current] = englishlist[index]
            current+=1
        #print(locations)
        return locations
    
    def readStopwordsFile(self):
        # stopwords
        stopwords = []
        with open('util/stopwords.csv') as f:
            for line in f:
                line = line.strip()
                content = line.split(",")
                stopwords += content
        #print(stopwords)
        return stopwords
    
    
    # modified
    # Step 1: Parse an XML document
    def readInputXMLFiles(self):    
        alltweets = []
        alllocations = []
        alltimestamps = []
        alltimezones = []
        allusers = []
        alluserspopularity = []
        allids = []
        allinreplytoid = []
        
        # [1-66] : [21/Dec/2015 - 18/July/2016] : 1,027,944 tweets
        # [1-75] : [21/Dec/2015 - 06/Sep/2016]  : 1,242,135 tweets
        filenames = glob.glob("data/all/dataset[0-7][0-9].xml") # all data: 01-75
        #filenames = glob.glob("data/all/dataset[0][1-8].xml")  # test in small dataset
        #filenames = glob.glob("data/all/dataset[0-1][0-9].xml")
        for filename in filenames:
            with open(filename, 'r', encoding="utf-8"):# as content:
                print(filename)
                # read XML file
                tweets, timestamps, timezones, ids, inreplytoid, locations, users, userspopularity = self.parseXMLFile(filename);
                #print(len(tweets))
                alltweets += tweets
                alllocations += locations
                alltimestamps += timestamps
                alltimezones += timezones
                allids += ids
                allinreplytoid += inreplytoid
                allusers += users
                alluserspopularity += userspopularity
            #print(len(alltweets))
            
        # replace Chinese location names with English location names
        alllocations = self.locationsMapping(alllocations)
        
        return alltweets, alltimestamps, alltimezones, allids, allinreplytoid, alllocations, allusers, alluserspopularity
    
    # modified
    def parseXMLFile(self,filename):      
        # read XML file
        tree = ET.parse(filename)#, encoding='utf-8'))
        tweetList = tree.getroot()
        
        tweets = []
        locations = []
        timestamps = []
        timezones = []
        users = []
        userspopularity = []
        
        ids = []
        inreplytoid = []

        for tweet in tweetList:
            messagelanguage, messageid, messagetext, messageInReplyToId, messagetimestamp,locationname, userscreenname, userfollowers, userfriends = self.extractTweetData(tweet);
            if( (messagelanguage != "en") or (messagetext == " ") ):
                continue
            #print("\n\ntext: ", messagetext, "\ntime: ", messagetimestamp, "\nlocation:", locationname);    
            
            # stopword removal
            tokens = self.stopwordRemoval(messagetext)    
            if(len(tokens) == 0):
                continue
            text = ' '.join(tokens)
            #print(text)
            
            tweets.append(text)  
            
            ids.append(messageid)
            inreplytoid.append(messageInReplyToId)
            
            # timezones
            timestamps.append(messagetimestamp)
            timezone = self.implyTimezone(messagetimestamp)
            timezones.append(timezone)
            #print(timezone)
            
            # locations
            locations.append(locationname)
            users.append(userscreenname)
            
            # popularity of users
            userpopularity = self.implyPopularity(userfollowers, userfriends)
            userspopularity.append(userpopularity)
                   
        #sys.exit(0)
        return tweets, timestamps, timezones, ids, inreplytoid, locations, users, userspopularity
    
    
    
    def extractTweetData(self,tweet):
        messagetext = tweet.find("message/text").text;
        if( (messagetext is None) or (len(messagetext) <= 1) ):
            messagetext = " "
        else:
            messagetext = messagetext.lower()
            messagetext = messagetext.replace('.','. ')
            messagetext = messagetext.replace(',',', ')
            messagetext = messagetext.replace('?','? ')
            
        messageid = tweet.find("message/id").text;
        messagetimestamp = tweet.find("message/timestamp").text;
        messagelanguage = tweet.find("message/lang").text
        messageInReplyToId = tweet.find("message/inreplytoid").text
        locationname = tweet.find("location/name").text;
        userscreenname = tweet.find("user/screenname").text;
        userfollowers = int(tweet.find("user/followers").text);
        userfriends = int(tweet.find("user/friends").text);
        return messagelanguage, messageid, messagetext, messageInReplyToId, messagetimestamp, locationname, userscreenname, userfollowers, userfriends
    
    
    def stopwordRemoval(self,messagetext):
        #print(self.stopwords)
        tokensbefore = messagetext.split()
        #print("message before: ", tokensbefore)
        tokensafter = []
        for token in tokensbefore:
            # keep hashtags and mentioned users
            if (token == '@'):
                continue
            if (token.startswith('#') or token.startswith('@')) : 
                if(token.endswith(':') or token.endswith('…') or token.endswith('.')) :
                    token = token[:-1]
                tokensafter.append(token)
                continue    
            # remove last characters of words
            while ( token.endswith(('?', '.', '...', '!', ';', ',', '…', '~', ' ', '/', ':', '(', ')', '_', '*', '-', '。', '+')) ) :
                #print(token)
                token = token[:-1]
                #print(token)
            # remove first characters of words
            while( token.startswith('&') or token.startswith('.') or token.startswith('(') or token.startswith(')') or token.startswith('_') or token.startswith(':') or token.startswith('-')) :
                token = token[1:]
            # ignore long words
            if (len(token) > 15): 
                continue
            if (token.endswith('’s')):
                token = token[:-2]
                #tokensafter.append(token)
            if (token.startswith('https')):
                token = " " 
            
            # remove stopwords
            if (token != " ") and (token != ''):
                if (token not in self.stopwords):
                    tokensafter.append(token)  
            
                
        if len(tokensafter) == 0:
                tokensafter.append(" ")
        #print("message after: ",tokensafter, "\n")
        return tokensafter
    
    
    def implyTimezone(self,messagetimestamp):
        ts = parse(messagetimestamp);
        hour = ts.hour;
        if (hour>=6 and hour<18):
            return('day')
        else:
            return('night')
            
    def implyPopularity(self, userfollowers, userfriends):
        if userfriends == -1:
            userfriends = 0
        userpopularity = float((1.0 + float(userfollowers)) / (1.0 + float(userfriends)))
        return userpopularity
    
    
    # modified
    # timestamps: list of all timestamps
    # startoffset: start index of timestamps-list
    # windowlength: size of window in days
    # returns: endoffset, numTweets
    def timewindow(self,timestamps,startoffset,windowlength):
        startdate = parse(timestamps[startoffset])
        #print(startdate - parse(timestamps[len(timestamps)-1]))
        numTweets = 0;
        pos = startoffset
        while pos < len(timestamps):
            currentdate = parse(timestamps[pos])
            #diff = currentdate - startdate
            rdelta = relativedelta(currentdate,startdate)
            #print(diff, rdelta)
            if(rdelta.days < windowlength) :
                numTweets += 1
                #print(pos, "\t" ,timestamps[pos])
            else:
                break;
            pos += 1
        #print(pos)
        endoffset = pos
        #print("[",firstday,firstmonth, "]  -  [", lastday,lastmonth,"]")
        #print(startoffset,endoffset,(endoffset-startoffset))
        return endoffset, numTweets
        
   

###
###
###    
#class XMLReader():  
        
# 1. read XML files with tweets
#       readInputXMLFiles(): function for setting input files range
# 2. process data with time-windows 
#       windowlength: parameter for setting window length in days
def main(numfeatures, numiterations, windowlength, lstmposnegsamplingratio, filterInputChoice, filteroutTweetsThreshold, lstm_nb_epoch, lstm_batch_size, lstm_validation_split, lstm_optimizer):
    xmlreader = XMLReader()
    
    # log file
    filename = "output/log.txt"
    file = open(filename,'a') 
    log_window_report = "\n\n--- Experimental Set Up ---\n"
    log_window_report += "\nNumber of features:  " + str(numfeatures)
    log_window_report += "\nNumber of iterations:  " + str(numiterations)
    log_window_report += "\nLength of window (in days):  " + str(windowlength)
    log_window_report += "\nRatio of positive_datapoints/negative_samples:  " + str(lstmposnegsamplingratio)
    log_window_report += "\nFiltering-out input tweets (boolean 1:yes, 0:no):  " + str(filterInputChoice)
    log_window_report += "\nThreshold for filtering-out input tweets:  " + str(filteroutTweetsThreshold)
    log_window_report += "\nLSTM number of epochs:  " + str(lstm_nb_epoch)
    log_window_report += "\nLSTM batch size:  " + str(lstm_batch_size)
    log_window_report += "\nLSTM validation split:  " + str(lstm_validation_split)
    log_window_report += "\nLSTM optimizer:  " + lstm_optimizer
    log_window_report += "\n--- ------------------- ---\n"
    file.write(log_window_report) 
    file.flush()
    
    # 1. read XML files with tweets
    print("Loading Input: Reading input XML...") 
    tweets, timestamps, timezones, ids, inreplytoid, locations, users, userspopularity = xmlreader.readInputXMLFiles();
    print("\tnumber of all tweets: " , len(tweets))
    print("Loading Input is finished. XML file is read.") 
    
    
    # 2. process data with time-windows 
    endoftweets = len(tweets)
    windowcounter = 1;
    startoffset = 0; #665889;
    #windowlength = 7; # window length in days
    while True:
        start = time.time()        
        endoffset, numTweets = xmlreader.timewindow(timestamps, startoffset, windowlength)        
        print("\n\n[",windowcounter,"]", " from: ",startoffset, " to: ",endoffset, " num: ",numTweets)
        
        # log file
        log_window_report = "\n-----\n"+"Window ID: "+str(windowcounter)
        log_window_report += "\nTime Period:  " + str(timestamps[startoffset]) + "  -  " + str(timestamps[endoffset-1])
        log_window_report += "\nNumber of tweets: " + str(numTweets)
        file.write(log_window_report) 
        file.flush()
        
        tweetsportion = tweets[startoffset:endoffset]
        timestampsportion = timestamps[startoffset:endoffset]
        timezonesportion = timezones[startoffset:endoffset]
        idsportion = ids[startoffset:endoffset]
        inreplytoidportion = inreplytoid[startoffset:endoffset]
        locationsportion = locations[startoffset:endoffset]
        usersportion = users[startoffset:endoffset]
        userspopularityportion = userspopularity[startoffset:endoffset]
        #print(len(tweetsportion) , len(timestampsportion), len(timezonesportion), len(locationsportion), len (usersportion))
        
        # 1: yes filtering, filter out users with 1 tweet only and their tweets
        # 0 : no filtering
        if(filterInputChoice == 1):
            # filterInput.py : 
            tweetsfiltered, timestampsfiltered, timezonesfiltered, idsfiltered, inreplytoidfiltered, locationsfiltered, usersfiltered, userspopularityfiltered = filterInput.main(filteroutTweetsThreshold, tweetsportion, timestampsportion, timezonesportion, idsportion, inreplytoidportion, locationsportion, usersportion, userspopularityportion)
            # log file
            log_window_report = "\nNumber of tweets after filtering: " + str(len(tweetsfiltered))
            file.write(log_window_report) 
            file.flush()
            # process.py : Create userstags and wordstags matrices 
            userstags, wordstags, processedtweets, topicWordsDoc, usertweetPairList, usersUsersTweetsPairList, usersetlist, tweetsWithHashtags = process.main(tweetsfiltered, timestampsfiltered, timezonesfiltered, idsfiltered, inreplytoidfiltered, locationsfiltered, usersfiltered, userspopularityfiltered)
        else:
            # process.py : Create userstags and wordstags matrices 
            userstags, wordstags, processedtweets, topicWordsDoc, usertweetPairList, usersUsersTweetsPairList, usersetlist, tweetsWithHashtags = process.main(tweetsportion, timestampsportion, timezonesportion, idsportion, inreplytoidportion, locationsportion, usersportion, userspopularityportion)
        
        # model.py : HTLIC, Doc2Vector, TopicEnhDoc, LSTM
        model.main(userstags, wordstags, processedtweets, topicWordsDoc, windowcounter, usertweetPairList, usersUsersTweetsPairList, usersetlist, numfeatures, numiterations, lstmposnegsamplingratio, tweetsWithHashtags, lstm_nb_epoch, lstm_batch_size, lstm_validation_split, lstm_optimizer)
        
        '''    
        # process.py : Create userstags and wordstags matrices 
        ### old code: #userstags, wordstags, processedtweets, topicWordsDoc, positive_datapoints, negative_samples = process.main(tweetsportion, timestampsportion, timezonesportion, locationsportion, usersportion, userspopularityportion)
        #userstags, wordstags, processedtweets, topicWordsDoc, usertweetPairList, usersetlist = process.main(tweetsportion, timestampsportion, timezonesportion, locationsportion, usersportion, userspopularityportion)
        tweetsfiltered, timestampsfiltered, timezonesfiltered, locationsfiltered, usersfiltered, userspopularityfiltered = filterInput.main(tweetsportion, timestampsportion, timezonesportion, locationsportion, usersportion, userspopularityportion)
        userstags, wordstags, processedtweets, topicWordsDoc, usertweetPairList, usersetlist = process.main(tweetsfiltered, timestampsfiltered, timezonesfiltered, locationsfiltered, usersfiltered, userspopularityfiltered)
        # model.py : HTLIC, Doc2Vector, TopicEnhDoc, LSTM
        ### old code: #model.main(userstags, wordstags, processedtweets, topicWordsDoc, windowcounter, positive_datapoints, negative_samples)
        model.main(userstags, wordstags, processedtweets, topicWordsDoc, windowcounter, usertweetPairList, usersetlist, numfeatures, numiterations, lstmposnegsamplingratio)
        ### old code: #process.main(tweetsportion, timestampsportion, timezonesportion, locationsportion, usersportion, userspopularityportion)
        ''' 
        
        end = time.time()
        print("\nWindow: ", windowcounter ,"\tTime Elapsed: ", (end - start), " sec")
        
        # log file
        file = open(filename,'a')
        log_window_report = "\nTime Elapsed: " + str(end - start) + " sec" + "\n\n"
        file.write(log_window_report) 
        file.flush()
    
        windowcounter += 1
        if (numTweets == 0) or (endoffset == endoftweets):
            file.close()
            break;
        else:
            startoffset = endoffset
            
        #if(windowcounter == 13):
        #    file.close()
        #    break;
    
    print("\n------------------\nProgram ended.\n\n")
 
    
#if __name__ == "__main__":
#   main()
   # print("Running XMLReader...")
   # userstags, wordstags, tweets, topicWords = main()