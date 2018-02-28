# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:56:33 2018

@author: zdu
"""

class Enermy:
    life = 3
    
    def attack(self):
        print('ouch!')
        num = 10
        self.life = self.life - num
        
    def checklife(self):
        print(str(self.life + " life left"))
        
enermy1 = Enermy()
#enermy1.attack(enermy1,10)
enermy1.attack()
print(enermy1.life)