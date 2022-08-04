#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:43:17 2018

@author: konstantinos
"""

import numpy as np
import math
import time # 
import sys 

#import datapoints

class Process(): 
    def __init__(self,tweets, timestamps, timezones, ids, inreplytoid, locations, users, userspopularity):
        self.tweets = tweets
        self.timestamps = timestamps
        self.timezones = timezones
        self.ids = ids
        self.inreplytoid = inreplytoid
        self.locations = locations
        self.users = users
        self.userspopularity = userspopularity
    
    def get_usertags(self):
        return self.userstags
    
    def get_wordstags(self):
        return self.wordstags
    
    def get_topicWords(self,words):
        self.topicWords = self.getColumn(words,1)
        return self.topicWords
    
    # get list of words for each tweet
    def getListOfWordsForEachTweet(self,tweets):
        wordlist = []
        pos = 0;
        while pos < len(tweets):
            messagetext = tweets[pos]
            wordlist.append(messagetext.split())
            pos += 1
        return wordlist
            
    
    # Step 2a: Counts frequency of words, Sorts and Returns the topk
    def messageToSortedFreqDict(self,tweets,topk):
        wordlist = []
        for messagetext in tweets:
            wordlist += messagetext.split()
    
        wordfreq = []
        wordfreq = [wordlist.count(w) for w in wordlist] # a list comprehension
    
        #print("Frequencies\n" + str(wordfreq) + "\n")
        #print("Pairs\n" + str(zip(wordlist, wordfreq)))
        #print (wordlist[3], wordfreq[3])
        #return wordlist,wordfreq
        dictionary = dict(zip(wordlist,wordfreq))
    
        # Sort a dictionary of word-frequency pairs in order of descending frequency.
        aux = [(dictionary[key], key) for key in dictionary]
        aux.sort()
        aux.reverse()
        sortedDictionary = aux
    
        #return topk
        return sortedDictionary[:topk]
    
    
    # 2b. sort words by TFIDF
    # http://web.cs.ucla.edu/~thuyvu/cikm13-tvu.pdf
    # Let T = {t_1, t_2, ..., t_|T|} be the set of tweet messages. 
    # The TFIDF score for each word w in tweets T 
    # is calculated using the following formulas:
    # 
    # TFIDF(w, T) = TF(w, T) × IDF(w, T)
    # IDF(w,T) = log (|T|) / (|t : w ∈ t and t ∈ T|)
    #
    # Where TF(w, T) returns the frequency of word w in all tweets T; 
    # |T| is the number of tweets and 
    # |t : w ∈ t and t ∈ T| is the number of tweets having word w.
    def estimateTfIdf(self,tweets,topk):
        N = len(tweets)
        
        # collect set of all words (no duplicates) : allwordset
        allwordset = set()
        pos = 0
        while pos < len(tweets):
            messagetext = tweets[pos]   
            #print(messagetext)
            ws = messagetext.split()
            for w in ws:
                allwordset.add(w) # add to set 
            pos += 1
            
        vocabularySize = len(allwordset)    
        print("\t\tVocabulary length: ", vocabularySize)
        #print(allwordset)
        
        # list of indexes of words (for each tweet) in allwordset
        wordlistIndex = []
        pos = 0;
        while pos < len(tweets):
            messagetext = tweets[pos]   
            #print(messagetext)
            ws = messagetext.split()
            tweetWordsIndex = []
            for w in ws:
                tweetWordsIndex.append(self.getSetIndex(allwordset,w))
            wordlistIndex.append(tweetWordsIndex) # keep duplicate words
            pos += 1

        #print(wordlistIndex)
        #print(len(wordlistIndex))
        
        # correspondence with vocabularySize and allwordset[]
        # df: counts the number of tweets having each vocabulary word 
        # tf: the frequency of word w in all tweets T
        tf = []
        df = []
        idf = []
        tfidf = []
        pos = 0
        while pos < vocabularySize:
            tf.append(0)
            df.append(0)
            idf.append(0)
            tfidf.append(0)
            pos += 1
        
        #print(len(df))
  
        # work with indexes only (not with words)
        pos = 0;
        while pos < len(tweets):
            #if(pos%100 == 0):
            #    print("\t\t\tNumber of tweets: ", pos)   
            indexlist = wordlistIndex[pos]
            for index in indexlist: # !!
                tf[index] += 1
            indexset = set(indexlist) # !!
            for index in indexset:  # !!
                df[index] += 1
            pos += 1
            
        #print("\t\ttf, df are calculated.")
              
        vocabularyIndex = 0
        while vocabularyIndex < vocabularySize:
            idf[vocabularyIndex] = math.log10(N/df[vocabularyIndex])
            tfidf[vocabularyIndex] = tf[vocabularyIndex] * idf[vocabularyIndex]
            vocabularyIndex += 1
            
        #print(idf)
        #print("\t\ttfidf is calculated.")
        
        allwordlist = list(allwordset)
        #vocabularyIndex = 0
        #while vocabularyIndex < vocabularySize:
        #    print(vocabularyIndex,"\t",allwordlist[vocabularyIndex],
        #          "\ttf: ", tf[vocabularyIndex],
        #          "\tidf: ", idf[vocabularyIndex],
        #          "\ttfidf: ", tfidf[vocabularyIndex])
        #    vocabularyIndex += 1
    
        dictionary = dict(zip(allwordlist,tfidf))
        # Sort a dictionary of word-frequency pairs in order of descending frequency.
        aux = [(dictionary[key], key) for key in dictionary]
        aux.sort()
        aux.reverse()
        sortedDictionary = aux
    
        #return topk
        return sortedDictionary[:topk]
    
    
    
    # Step 3: finds the topic words for each tweet-document
    def createTopicWordsDoc(self,words,tweets):
        #topicWords = XMLReader.getColumn(words,1)
        topicWords = self.get_topicWords(words)
        topicWordsDoc = [[-1]] * len(tweets)
    
        # list of words for each tweet
        wordlist = self.getListOfWordsForEachTweet(tweets)
    
        #print(wordlist[2])
        #print(wordlist[10], len(wordlist))
        tweetscount = 0
        for wlst in wordlist: # for each tweet wlst
            indexlist = [] # list of indexes
            wordpos = 0;
            while wordpos < len(wlst): # for each word w of tweet
                w = wlst[wordpos]
                #print("tweet index: ",tweetscount, "\tword: ",w)
                #print(w)
                if w in topicWords: # if w exists in topic words
                    topicwordindex = topicWords.index(w)
                    indexlist.append(topicwordindex)
                    #print("tweet index: ",tweetscount, "\tword: ",w, "\tindex: ",topicwordindex)
                wordpos += 1
            #print("tweet index: ",tweetscount, "\tlength list of indexes: ",len(indexlist), "\t", indexlist)
            topicWordsDoc[tweetscount] = list(indexlist)
            tweetscount += 1
        #print(len(topicWordsDoc))
        #print(topicWordsDoc)
        #print("topicWordsDoc vector created.")
        return topicWordsDoc
    
    
    def getColumn(self,matrix, i):
        return [row[i] for row in matrix]
    
    def getSetIndex(self,s, element):
        s = list(s);
        index = s.index(element);
        return index;
    
    
    # Step 5:
    # Creates the userstags matrix: [userindex][tagindex]
    #    usersSet : unique users for indexing (string values)
    #    tagsSet : unique tags for indexing (string values)
    #    users : list of all users (length of tweets) (string values)
    #    tags : list of all tags (length of tweets) (string values)
    def createUsersTags(self,usersSet, tagsSet, users, tags):
        r = len(usersSet)
        c = len(tagsSet)
        self.userstags = np.empty(shape=(r,c))
        self.userstags.fill(0)
    
        #userstags = [[0 for x in range(c)] for y in range(r)] 
        #print("element: ", userstags[0][1448])
        #print(len(users))
        #pos = 0;
        #while pos < len(users): 
        for pos in range(len(users)):
            #print(pos, users[0], tags[0])
            user = users[pos]
            tag = tags[pos]
            userindex = self.getSetIndex(usersSet, user)
            tagindex = self.getSetIndex(tagsSet, tag)
            #print(userindex, tagindex)
            self.userstags[userindex][tagindex] += 1
            #print(userindex, tagindex, self.userstags[userindex][tagindex])
            #pos += 1
        #print("userstags matrix created.")
        #return userstags
    
    # Step 6:
    # Creates the wordstags matrix: [wordindex][tagindex]
    #    tweets : each tweet for word indexing
    #    tagsSet : unique tags for indexing
    #    words : list of all topic words for each tweet (length of tweets)
    #    tags : list of all tags (length of tweets)
    def createWordsTags(self,tweets,tagsSet,words,tags):
        #topicWords = XMLReader.getColumn(words,1)
        #print(len(tweets),len(tagsSet),len(words),len(tags))
        topicWords = self.get_topicWords(words)
        r = len(words)
        c = len(tagsSet)
        self.wordstags = np.empty(shape=(r,c))
        self.wordstags.fill(0)
    
        # list of words for each tweet
        wordlist = self.getListOfWordsForEachTweet(tweets)
    
        #print(len(wordlist))
        pos = 0
        while pos < len(wordlist):
            tag = tags[pos]
            tagindex = self.getSetIndex(tagsSet, tag)
            wlst = wordlist[pos] # for each tweet wlst (listof words) 
            for w in wlst: # for each word
                if w in topicWords: # if w exists in topic words
                    wordindex = topicWords.index(w)
                    self.wordstags[wordindex][tagindex] += 1
            pos += 1
        
        #print("wordstags matrix created.")
        #return wordstags
    
    # Step 4: 
    def createTags(self,locations,timezones):
        #tags = []
        #for i in range(len(locations)):
        #    tags.append([locations[i], timezones[i]])
        #return tags
        zipped = zip(locations, timezones)
        tags = list(zipped)
        return tags
    
    
    # 1a1. split tweets per user
    # returns set of users and list with tweets for each user. correspondence with userset
    def splitTweetsPerUser(self,tweets,users,usersSet):
        #userset = set(users)
        #print(userset)
        usersetlist = list(usersSet)
        tweetindex_list = [[-1]] * len(usersSet)
        
        pos = 0
        while pos < len(usersetlist):
            username = usersetlist[pos]
            indices = [i for i, x in enumerate(users) if x == username]
            #print(indices)
            tweetindex_list[pos] = indices
            #print(username, "\t", indices)#, "\t\t", tweetindex_list[pos])
            pos += 1
      
        #print(tweetindex_list[pos])      
        return usersetlist, tweetindex_list
    
    # Creates pairs of users - tweets.
    # returns a list with pairs of <user_index,tweet_index>
    def createUserTweetList(self,usersetlist,tweetindex_list):
        usertweetList = []
        pos = 0 
        while pos < len(usersetlist):
            #user = usersetlist[pos]
            tweets = tweetindex_list[pos] 
            userindex = pos
            tweetpos = 0
            while tweetpos < len(tweets):
                pair = []
                tweetindex = tweets[tweetpos]
                pair.append(userindex)
                pair.append(tweetindex)
                tweetpos += 1
                usertweetList.append(pair)
            pos += 1
        return usertweetList
    
    # mentionedUsersTweetPairList: <user index id, tweet index id>
    # tweet mentions a user
    def findTweetsMentionedUsers(self,tweets,users,usersSet):
        #print(len(usersSet))
        usersSetList = list(usersSet);
        lowerUsersSetList = []
        for username in usersSetList:
            lowerUsersSetList.append(username.lower())
        #print(len(lowerUsersSetList))
        #print(lowerUsersSetList)
        mentionedUsersTweetPairList = []
        tweetpos = 0
        while tweetpos < len(tweets):
            messagetext = tweets[tweetpos]   
            words = messagetext.split()
            for word in words:                   
                if(word.startswith('@')):
                    mentioneduser = word[1:]
                    #print(word, mentioneduser)
                    if mentioneduser in lowerUsersSetList:
                        userindex = lowerUsersSetList.index(mentioneduser)
                        pair = []
                        tweetindex = tweetpos
                        pair.append(userindex)
                        pair.append(tweetindex)
                        #print(word, mentioneduser, pair)
                        if(pair not in mentionedUsersTweetPairList):
                            mentionedUsersTweetPairList.append(pair)
            tweetpos += 1
            
        return mentionedUsersTweetPairList
    
    # usersMentionedUsersPairList: <user index id, mentioned_user index id>
    # user mentions a user
    def findUsersMentionedUsers(self,tweets,users,usersSet):
        usersSetList = list(usersSet);
        lowerUsersSetList = []
        for username in usersSetList:
            lowerUsersSetList.append(username.lower())
            
        usersMentionedUsersPairList = []
        tweetpos = 0
        while tweetpos < len(tweets):
            messagetext = tweets[tweetpos]   
            authoruser = users[tweetpos] # string username
            if authoruser in lowerUsersSetList:
                userindex = lowerUsersSetList.index(authoruser)
            else:
                tweetpos += 1
                continue
            words = messagetext.split()
            for word in words:                   
                if(word.startswith('@')):
                    mentioneduser = word[1:]
                    if mentioneduser in lowerUsersSetList:
                        mentioned_userindex = lowerUsersSetList.index(mentioneduser)
                        pair = []
                        pair.append(userindex)
                        pair.append(mentioned_userindex)
                        
                        #print(word, mentioneduser, pair)
                        if(pair not in usersMentionedUsersPairList):
                            usersMentionedUsersPairList.append(pair)
            tweetpos += 1
            
        return usersMentionedUsersPairList
    
    # usersUsersTweetsPairList: <user index id, mentioned_user index id>
    # user and users' friends tweets
    def findUsersUsersTweets(self,tweets, users, usersSet, usertweetPairList, usersMentionedUsersPairList):
        usersUsersTweetsPairList = []
        #mentionedUsers = []
        #print(len(usertweetPairList))
        #print(len(usersMentionedUsersPairList)) 
    
        for userMentionedUserPair in usersMentionedUsersPairList:
            useru = userMentionedUserPair[0]
            mentionedUser = userMentionedUserPair[1]
            if(useru != mentionedUser):
                #mentionedUsers.append(mentionedUser)
                for userTweetPair in usertweetPairList:
                    usert = userTweetPair[0]
                    tweet = userTweetPair[1]
                    pair = []
                    if(usert == mentionedUser):
                        pair.append(useru)
                        pair.append(tweet)
                        if(pair not in usersUsersTweetsPairList):
                            usersUsersTweetsPairList.append(pair)    
                            
        #print(len(usersUsersTweetsPairList))
        #common = 0
        #for pair in usersUsersTweetsPairList:
        #    if(pair in usertweetPairList):
        #        common += 1
        #print("common: ", common)
            
        return usersUsersTweetsPairList
    
    def findTweetsWithHashtag(self, tweets):
        tweetsWithHashtags = []
        tweetpos = 0
        while tweetpos < len(tweets):
            messagetext = tweets[tweetpos]   
            words = messagetext.split()
            hashtag = 0 # no
            for word in words:
                if(word.startswith('#')):
                    hashtag = 1
                    break;
            if(hashtag == 1):
                tweetsWithHashtags.append(tweetpos)
            tweetpos += 1
        return tweetsWithHashtags
    

    # 1b1. split users according to their popularity = followers / friends
    # returns sets with indexes of influential and noninfluential users (usernames). In corespondence with users
    def splitUsers(self, userspopularity, users):
        influentialusers = []
        noninfluentialusers = []
        
        numTweets = len(userspopularity)
        pos = 0
        while pos < numTweets:
            userpopularity = userspopularity[pos]
            user = users[pos]
            #print(userpopularity)
            if(userpopularity > 10.0):
                influentialusers.append(user)
            else:
                noninfluentialusers.append(user)
            pos += 1
        
        influentialusersset = set(influentialusers)
        noninfluentialusersset = set(noninfluentialusers)
        return influentialusersset, noninfluentialusersset
    
    
    #1b2. Mathes influential users with their tweets and non-influential users with their tweets.
    # Returns :
    #    influentialuserssetlist : a list of unique influential users
    #    influential_tweetindex_list : a list of tweet indexes for each influential user (list of lists) 
    #    noninfluentialuserssetlist : a list of unique noninfluential users
    #    noninfluential_tweetindex_list : a list of tweet indexes for each noninfluential user (list of lists)
    def matchingUsersTweets(self,influentialusersset, noninfluentialusersset, usersetlist, tweetindex_list):
        influentialuserssetlist = list(influentialusersset)
        noninfluentialuserssetlist = list(noninfluentialusersset)
        influential_tweetindex_list = [[-1]] * len(influentialusersset)
        noninfluential_tweetindex_list = [[-1]] * len(noninfluentialusersset)
        
        influentialuserIndex = 0;
        while(influentialuserIndex < len(influentialuserssetlist)):
            influentialuser = influentialuserssetlist[influentialuserIndex]
            generalIndex = self.getSetIndex(usersetlist, influentialuser)
            influential_tweetindex_list[influentialuserIndex] = tweetindex_list[generalIndex]
            influentialuserIndex += 1
            
        noninfluentialuserIndex = 0;
        while(noninfluentialuserIndex < len(noninfluentialuserssetlist)):
            noninfluentialuser = noninfluentialuserssetlist[noninfluentialuserIndex]
            generalIndex = self.getSetIndex(usersetlist, noninfluentialuser)
            noninfluential_tweetindex_list[noninfluentialuserIndex] = tweetindex_list[generalIndex]
            noninfluentialuserIndex += 1
            
        return influentialuserssetlist, influential_tweetindex_list, noninfluentialuserssetlist, noninfluential_tweetindex_list

     
    # 1a2. find tweets with retweets
    def matchInReplyToId(self,ids,inreplytoid):
        length = len(inreplytoid)
        inreplytotweet = []
        tweetpos = 0
        while tweetpos < length:
            inreplytotweet.append(-1)
            tweetpos += 1

        tweetposr = 0
        while tweetposr < length:
            reply = inreplytoid[tweetposr]
            if(reply == "-1"):
                tweetposr += 1
                continue;
            tweetposo = 0
            while tweetposo < tweetposr: #tweetpos1:
                #print(ids[tweetpos2], reply1)
                if(ids[tweetposo] == reply):
                    inreplytotweet[tweetposr] = tweetposo
                    break;
                tweetposo += 1
                
            tweetposr += 1;
        
        return inreplytotweet;
    
        # Enchance user,tweet list with replies to this tweet
    def enchanceWithTweetReplies(self,userTweetPairList,inreplytotweet):
        pairpos = 0
        while pairpos < len(userTweetPairList):
            utpair = userTweetPairList[pairpos]
            u = utpair[0]
            t = utpair[1]
            #print(len(inreplytotweet), t)
            replyto = inreplytotweet[t]
            pair = []
            if (replyto != -1):
                pair.append(u)
                pair.append(replyto)
                if(pair not in userTweetPairList):
                    userTweetPairList.append(pair)
            pairpos += 1
        return userTweetPairList
    

# 1. Split dataset
#       1a. split tweets per user
#       1b1. split users according to their popularity into influential and noninfluential
#       1b2. find tweets of influential users and tweets of noninfluential users
# 2. Find topic words
#           topk : parameter for number of topic words 
#       2a. sort words by frequency 
#       2b. sort words by TFIDF
# 3. Find topic words in each document
# 4. Create tags vector
# 5. Create usertags matrix
# 6. Create wordstags matrix
# return : userstags matrix, wordstags matrix, tweets (list of tweets), topicWordsDoc (topic words per doc)
def main(tweets, timestamps, timezones, ids, inreplytoid, locations, users, userspopularity):
    process = Process(tweets, timestamps, timezones, ids, inreplytoid, locations, users, userspopularity)
    # usersSet content: <string username>
    usersSet = set(users)
    #print("\tnumber of all tweets: " , len(tweets))
    #print("\tnumber of all timestamps: " , len(timestamps))
    #print("\tnumber of all timezones: " , len(timezones))
    #print("\tnumber of all locations: " , len(locations))
    #print("\tnumber of all users: " , len(users), users)
        
    print("Step 1: Split dataset...")
    # 1a1. split tweets per user
    # usersetlist: string username, tweetindex_list: integer index of tweet
    usersetlist,tweetindex_list = process.splitTweetsPerUser(tweets,users,usersSet)
    #print(len(usersetlist),len(tweetindex_list))
    # 1a2. find tweets with retweets
    inreplytotweet = process.matchInReplyToId(ids,inreplytoid)     
    #print(inreplytotweet)
    
    # Create positive Datapoints
    # positive datapoints
    # usertweetPairList: <user index id, tweet index id>
    # user posts a tweet
    usertweetPairList = process.createUserTweetList(usersetlist,tweetindex_list)
    #print(usertweetPairList)
    #print(len(usertweetPairList))
    
    # positive datapoints
    # mentionedUsersTweetPairList: <user index id, tweet index id>
    # tweet mentions a user
    mentionedUsersTweetPairList = process.findTweetsMentionedUsers(tweets,users,usersSet)
    #print(mentionedUsersTweetPairList)
    #print(len(mentionedUsersTweetPairList))

    
    # usersMentionedUsersPairList: <user index id, mentioned_user index id>
    # user mentions a user
    usersMentionedUsersPairList = process.findUsersMentionedUsers(tweets,users,usersSet)
    #print(usersMentionedUsersPairList)
    #print(len(usersMentionedUsersPairList))
    
    # Create negative Datapoints
    # negative datapoints: Only use friends' tweets
    # usersUsersTweetsPairList: <user index id, tweet index id>
    # user and users' friends tweets
    usersUsersTweetsPairList = process.findUsersUsersTweets(tweets,users,usersSet,usertweetPairList,usersMentionedUsersPairList)
    #print(usersUsersTweetsPairList)
    #print(len(usersUsersTweetsPairList))
    # Enchance user,tweet list with replies to this tweet
    usersUsersTweetsPairList = process.enchanceWithTweetReplies(usersUsersTweetsPairList,inreplytotweet)
    #print(len(usersUsersTweetsPairList))
            
            
    # Union positive Datapoints
    # append usertweetPairList with mentionedUsersTweetPairList
    # positive datapoints: Union of usertweetPairList and mentionedUsersTweetPairList
    for pair in mentionedUsersTweetPairList:
        if(pair not in usertweetPairList):
            usertweetPairList.append(pair)
    #usertweetPairList.extend(mentionedUsersTweetPairList)
    #print(usertweetPairList)
    #print(len(usertweetPairList))
    # Enchance user,tweet list with replies to this tweet
    usertweetPairList = process.enchanceWithTweetReplies(usertweetPairList,inreplytotweet)
    #print(len(usertweetPairList))
    
    # used in negative datapoints: Only consider tweets with hashtags
    tweetsWithHashtags = process.findTweetsWithHashtag(tweets)
    #print(tweets[tweetsWithHashtags[0]])
    #print(len(tweetsWithHashtags))
    
    sim = 0;
    for pair in usertweetPairList:
        if pair in usersUsersTweetsPairList:
            usersUsersTweetsPairList.remove(pair);
            sim += 1;
    #print(len(usertweetPairList), len(usersUsersTweetsPairList),sim)
    print("\tTweets split per user done.")
    #sys.exit(0)
    
    
    # 1b1. split users according to their popularity into influential and noninfluential
    influentialusersset, noninfluentialusersset = process.splitUsers(userspopularity, users)
    #print("\tinfluential: ", len(influentialusersset), "\tnoninfluential: ", len(noninfluentialusersset))
    print("\tUsers split into influential and noninfluential is done.") 
    
    # 1b2. find tweets of influential users and tweets of noninfluential users
    influentialuserssetlist, influential_tweetindex_list, noninfluentialuserssetlist, noninfluential_tweetindex_list = process.matchingUsersTweets(influentialusersset, noninfluentialusersset, usersetlist, tweetindex_list)
    #print(len(influentialuserssetlist),len(influential_tweetindex_list))
    #pos = 0 
    #while pos < len(influentialuserssetlist):
    #    print(influentialuserssetlist[pos],"\t",influential_tweetindex_list[pos])
    #    pos += 1
    #print(len(noninfluentialuserssetlist),len(noninfluential_tweetindex_list))
    #pos = 0 
    #while pos < len(noninfluentialuserssetlist):
    #    print(noninfluentialuserssetlist[pos],"\t",noninfluential_tweetindex_list[pos])
    #    pos += 1
    influentialusertweetPairList = process.createUserTweetList(influentialuserssetlist,influential_tweetindex_list)
    noninfluentialusertweetPairList = process.createUserTweetList(noninfluentialuserssetlist, noninfluential_tweetindex_list)
    #print(influentialusertweetList)
    print("\tTweets of influential users and tweets of noninfluential users are collected.") 
    
    print("Step 1 finished. Split of dataset is done.")
    
    #sys.exit(0)
    
    """
    # 2a. sort words by frequency 
    print("Step 2: Finding Top frequent words...")
    topk = 500
    sortedDictionary = xmlreader.messageToSortedFreqDict(tweets,topk)
    #print(sortedDictionary)    
    topicwords = sortedDictionary
    #print(words)
    print("Step 2 finished. Top frequent words are found.")
    """ 
    
    # 2b. sort words by TFIDF
    print("Step 2: TFIDF ...")
    #start = time.time()
    topk = 500
    sortedTFIDF = process.estimateTfIdf(tweets,topk)
    #print(sortedTFIDF)
    topicwords = sortedTFIDF
    #end = time.time()
    #print("Time Elapsed: ", (end - start), " sec")
    print("Step 2 finished. TFIDF is estimated.")
    
    # 3. find topic words in each document
    print("Step 3: Finding Topic words per tweet...")
    topicWordsDoc = process.createTopicWordsDoc(topicwords,tweets)
    #print("\tlength of topicWordsDoc vector: ",len(topicWordsDoc))
    print("Step 3 finished. Topic words per tweet are found.")
    
    
    # 4. create tags vector
    print("Step 4: Creating Tags...")
    tags = process.createTags(locations,timezones)
    # tagsSet content: <string location, string timezone>
    tagsSet = set(tags)
    #print(tags)
    print("Step 4 finished. Tags are created.")
    
    # tagsSet content: <string location, string timezone>
    #tagsSet = set(tags)
    ### print(tagsSet)
    ### print("number of tags: " , len(tagsSet))
    # usersSet content: <string username>
    #usersSet = set(users)
    ### print("Users: " , usersSet)
    ### print("number of users: " , len(usersSet))
    ### wordsSet = set(sortedDictionary)
    ### print(len(wordsSet))
    ### sys.exit(0)
    
    # 5. create usertags matrix
    #    usersSet : unique users for indexing
    #    tagsSet : unique tags for indexing
    #    users : list of all users (length of tweets)
    #    tags : list of all tags (length of tweets)
    print("Step 5: Creating Matrix 'userstags'...")
    process.createUsersTags(usersSet,tagsSet, users, tags)
    userstags = process.get_usertags()
    print("\t[usersxtags]: ", userstags.shape)
    print("Step 5 finished. Matrix 'userstags' is created.")
            
    # 6. create wordstags matrix
    #    tweets : each tweet for word indexing
    #    tagsSet : unique tags for indexing
    #    words : list of all topic words for each tweet (length of tweets)
    #    tags : list of all tags (length of tweets)
    print("Step 6: Creating Matrix 'wordstags'...")
    process.createWordsTags(tweets,tagsSet,topicwords,tags) 
    wordstags = process.get_wordstags()
    print("\t[wordsxtags]: ", wordstags.shape)
    print("Step 6 finished. Matrix 'wordstags' is created.")
    
    #return userstags, wordstags, tweets, topicWordsDoc, positive_datapoints, negative_samples
    return userstags, wordstags, tweets, topicWordsDoc, usertweetPairList, usersUsersTweetsPairList, usersetlist, tweetsWithHashtags