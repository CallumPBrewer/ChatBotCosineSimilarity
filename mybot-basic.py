#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic chatbot design --- for your own modifications
"""
#######################################################
# Initialise Wikipedia agent
#######################################################
import wikipedia

import numpy as np
from numpy.linalg import norm
import math
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()

#######################################################
# Initialise weather agent
#######################################################
import json, requests
#insert your personal OpenWeathermap API key here if you have one, and want to use this feature
APIkey = "5403a1e0442ce1dd18cb1bf7c40e776f" 

#######################################################
#  Initialise AIML agent
#######################################################
import aiml
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the files are loaded.
# The optional brainFile argument specifies a brain file to load.
kern.bootstrap(learnFiles="mybot-basic.xml")
#######################################################
# Welcome user
#######################################################
print("Welcome to this chat bot. Please feel free to ask questions from me!")
#######################################################
# Main loop
#######################################################

while True:
    def tfidfVectorise(userSentence):
        listOfAllWords = []
        numberOfSamples = 0
        listOfAllSamples = []
        with open('csv.txt') as f:
            for line in f:
                numberOfSamples += 1
                line = line.split(",")
                fileQuestion = line[0]
                for word in fileQuestion.split(" "):
                    listOfAllWords.append(word)  
                listOfAllSamples.append(fileQuestion)
        
        userInputWords = userSentence.split(" ") 
        for word in userInputWords:
            listOfAllWords.append(word)
        listOfAllSamples.append(userSentence)
        numberOfSamples += 1
        
        #vectorise user sentence
        userInputWordsVector = []
        allUniqueWordsInUserSentence = []
        
        #loop through all samples
        for sample in listOfAllSamples:
            sample = sample.split(" ")
            for word in sample:
                #count every word in every sample
                numOfSamplesContainingWord = 0
                for sentence in listOfAllSamples:
                    sentence = sentence.split(" ")
                    if word in sentence:
                        numOfSamplesContainingWord += 1 
                if word not in allUniqueWordsInUserSentence:
                    tf = sentence.count(word)/len(sentence)
                    idf = math.log(numberOfSamples/numOfSamplesContainingWord,10)
                    userInputWordsVector.append(tf*idf)
                    allUniqueWordsInUserSentence.append(word)
        #print(allUniqueWordsInUserSentence)
        #print(userInputWordsVector)
        
        
        #vectorise every sentence
        
        closestMatch = ""
        greatestCosineSimilarity = 0
        #loop through every sample apart fro the sentence entered by the user
        for currentLine in listOfAllSamples[:-1]:
            currentLineAsList = currentLine.split(" ")
            sampleWordsVector = []
            allUniqueWordsInSample = []
            for sample in listOfAllSamples:
                sample = sample.split(" ")
                for word in sample:
                    #count every word in every sample
                    numOfSamplesContainingWord = 0
                    for sentence in listOfAllSamples:
                        sentence = sentence.split(" ")
                        if word in sentence:
                            numOfSamplesContainingWord += 1 
                    if word not in allUniqueWordsInSample:
                        tf = currentLineAsList.count(word)/len(currentLineAsList)
                        idf = math.log(numberOfSamples/numOfSamplesContainingWord,10)
                        sampleWordsVector.append(tf*idf)
                        allUniqueWordsInSample.append(word)
            cosineSimilarity = np.dot(userInputWordsVector,sampleWordsVector)/(norm(userInputWordsVector)*norm(sampleWordsVector))
            if cosineSimilarity > greatestCosineSimilarity:
                closestMatch = currentLine
                greatestCosineSimilarity = cosineSimilarity
        print("closest match: ",closestMatch," cosine similarity: ",greatestCosineSimilarity)
            
        #vectorise every sample
        for fileSentence in listOfAllSamples:
            fileSentence = fileSentence.split(" ")
            fileSentenceWordsVector = []
            for word in fileSentence:
                numOfSamplesContainingWord = 0
                for sentence in listOfAllSamples:
                    sentence = sentence.split(" ")
                    if word in sentence:
                        numOfSamplesContainingWord += 1    
                tf = fileSentence.count(word)/len(fileSentence)
                idf = math.log(numberOfSamples/numOfSamplesContainingWord)
                fileSentenceWordsVector.append(tf*idf)
            #print(fileSentence)
            #print(fileSentenceWordsVector)
    
    
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            print(params[2])
            break
        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3,auto_suggest=False)
                print(wSummary)
            except:
                print("Sorry, I do not know that. Be more specific!")
        elif cmd == 2:
            succeeded = False
            api_url = r"http://api.openweathermap.org/data/2.5/weather?q="
            response = requests.get(api_url + params[1] + r"&units=metric&APPID="+APIkey)
            if response.status_code == 200:
                response_json = json.loads(response.content)
                if response_json:
                    t = response_json['main']['temp']
                    tmi = response_json['main']['temp_min']
                    tma = response_json['main']['temp_max']
                    hum = response_json['main']['humidity']
                    wsp = response_json['wind']['speed']
                    wdir = response_json['wind']['deg']
                    conditions = response_json['weather'][0]['description']
                    print("The temperature is", t, "°C, varying between", tmi, "and", tma, "at the moment, humidity is", hum, "%, wind speed ", wsp, "m/s,", conditions)
                    succeeded = True
            if not succeeded:
                print("Sorry, I could not resolve the location you gave me.")
        elif cmd == 99:
            
            userInput = lemmatizer.lemmatize(userInput) 
            tfidfVectorise(userInput)
            userInputList = nltk.word_tokenize(userInput)
            sw = stopwords.words('english')
            Y_set = {w for w in userInputList if not w in sw} 
            
            
            listOfPairs = []
            with open('csv.txt') as f:
                lines = f.readlines()
                closestMatch = ["",0]
                for line in lines:
                    if line != "":
                        line = line.split(",")
                        fileQuestion = line[0]
                        fileQuestionTokenised = nltk.word_tokenize(fileQuestion)
                        sw = stopwords.words('english')
                        l1 =[];l2 =[]
                        X_set = {w for w in fileQuestionTokenised if not w in sw}
                        rvector = X_set.union(Y_set)
                        for w in rvector:
                            if w in X_set: l1.append(1) # create a vector
                            else: l1.append(0)
                            if w in Y_set: l2.append(1)
                            else: l2.append(0)
                        c = 0

                        # cosine formula 
                        for i in range(len(rvector)):
                            c+= l1[i]*l2[i]
                        cosine = c / float((sum(l1)*sum(l2))**0.5)
                        
                        if cosine > closestMatch[1]:
                            closestMatch = [line[1],cosine]

            print(closestMatch[0])
            #check csv file
    else:
        print(answer)
        

