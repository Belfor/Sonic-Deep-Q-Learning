# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:24:56 2020

@author: Belfor
"""
import csv
import random

class LevelManager():
    
    def __init__(self,training_file, validation_file):
    
        self.training = self.readFile(training_file)
        self.validation = self.readFile(validation_file)
    
    def readFile(self,file):
        env = []
        file =  open(file)
        reader = csv.reader(file,delimiter=',')
        for row in reader:
            env.append(row)
        return env
    
    def size_maps_training(self):
        return len(self.training)
    
    def listMap(self, training = True):
        if(training):
            for i in range(1,len(self.training) -1):
                print("{} -> {} - {}".format(i,self.training[i][0],self.training[i][1]))
        else:
            for i in range(1,len(self.validation) -1):
                print("{} -> {} - {}".format(i,self.validation[i][0],self.validation[i][1]))
    def getMap(self,level,training = True):
        if(training):
            return [self.training[level][0],self.training[level][1]]
        
        return [self.validation[level][0],self.validation[level][1]]
    
    def getRandomMap(self, test = None):
        if not test:
            rnd = random.randint(1,len(self.training) - 1)
        else:
            rnd = random.randint(1,len(self.validation) - 1)
        return [self.training[rnd][0],self.training[rnd][1]]
        
