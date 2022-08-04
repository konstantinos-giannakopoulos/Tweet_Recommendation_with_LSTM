#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 14:17:44 2018

@author: konstantinos
"""

#import sys

# filters out users with one tweet only and their tweets.
class FilterInput():
    
    # 1a. split tweets per user
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
    
    # Delete users with less than or equal one tweet and their tweets
    def collectUselessUserIds(self,filteroutTweetsThreshold,usersetlist,tweetindex_list):
        usersetlistidsUseless = []
        tweetidsUseless = []
        pos = 0 
        while pos < len(usersetlist):
            #print(usersetlist[pos],"\t",tweetindex_list[pos])
            if(len(tweetindex_list[pos]) <= filteroutTweetsThreshold):
                usersetlistidsUseless.append(pos)
                for tweetid in tweetindex_list[pos]:
                    tweetidsUseless.append(tweetid)
                #singletweetids.append(tweetindex_list[pos])
            pos += 1
        return usersetlistidsUseless, tweetidsUseless


def main(filteroutTweetsThreshold, tweets, timestamps, timezones, idsportion, inreplytoidportion, locations, users, userspopularity):
    filterInput = FilterInput()
    # usersSet content: <string username>
    usersSet = set(users)
    #print("\tnumber of all tweets: " , len(tweets))
    #print("\tnumber of all timestamps: " , len(timestamps))
    #print("\tnumber of all timezones: " , len(timezones))
    #print("\tnumber of all locations: " , len(locations))
    #print("\tnumber of all users: " , len(users), users)
        
    print("Step 0: Filter out dataset...")
    '''
    tweetsWithoutHashtags = filterInput.removeTweetsWithoutHashtag(tweets)
    #print(tweetsWithoutHashtags)
    #print(len(tweetsWithoutHashtags))
    reverseTweetsWithoutHashtags = sorted(tweetsWithoutHashtags, reverse=True)
    for tweetpos in reverseTweetsWithoutHashtags:
        del tweets[tweetpos]
        del timestamps[tweetpos]
        del timezones[tweetpos]
        del locations[tweetpos]
        del users[tweetpos]
        del userspopularity[tweetpos] 
    '''
    # 1a. split tweets per user
    # usersetlist: string username, tweetindex_list: integer index of tweet
    usersetlist,tweetindex_list = filterInput.splitTweetsPerUser(tweets,users,usersSet)
    #print(len(usersetlist),len(tweetindex_list))
    
    usersetlistidsUseless, tweetidsUseless = filterInput.collectUselessUserIds(filteroutTweetsThreshold,usersetlist,tweetindex_list)
    #print(len(usersetlistidsUseless))
    #print(len(tweetidsUseless))
    #print(tweetidsUseless)
    reverseTweetids = sorted(tweetidsUseless, reverse=True)
    #print(reverseTweetids)
    # remove         
    for tweetpos in reverseTweetids:
        del tweets[tweetpos]
        del timestamps[tweetpos]
        del timezones[tweetpos]
        del locations[tweetpos]
        del users[tweetpos]
        del userspopularity[tweetpos] 
        del idsportion[tweetpos]
        del inreplytoidportion[tweetpos]
        
    #pos = 0
    #while(pos < len(tweets)):
    #    tweets[pos] = pos
    #    pos += 1
    tweetsfiltered = tweets
    timestampsfiltered = timestamps
    timezonesfiltered = timezones
    locationsfiltered = locations
    usersfiltered = users
    userspopularityfiltered = userspopularity
    idsfiltered = idsportion
    inreplytoidfiltered = inreplytoidportion

    print("\tnumber of tweets after filtering: " , len(tweets))
    #for element in reversed(usersetlistids_withonetweet):
    #    del usersetlist[element]
    #    del tweetindex_list[element]
    #usersetlist,tweetindex_list = filterInput.splitTweetsPerUser(tweets,users,usersSet)
    #print("users remaining: ", len(usersetlist))
    #pos = 0 
    #while pos < len(usersetlist):
    #    print(usersetlist[pos],"\t",tweetindex_list[pos])
    #    pos += 1
    print("Step 0 finished. Filtering dataset is done.")
    #sys.exit(0)
    return(tweetsfiltered, timestampsfiltered, timezonesfiltered, idsfiltered, inreplytoidfiltered, locationsfiltered, usersfiltered, userspopularityfiltered)