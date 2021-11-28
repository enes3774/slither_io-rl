
import selenium, time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pyautogui
import math
import keyboard
import time
import numpy as np
import cv2

class env():
    def __init__(self):
        self.browser = webdriver.Chrome()

        self.browser.get(r"https://slither.io");time.sleep(0.1)

        self.browser.find_element_by_xpath("/html/body/div[2]/div[4]/div[1]/input").click()
        self.browser.find_element_by_xpath("/html/body/div[2]/div[4]/div[1]/input").send_keys("enes3774")
        
        self.browser.maximize_window()
        self.score=10
        
        
    def step(self,degrees):
        degrees=(degrees*180)+180
        pyautogui.moveTo(int(962+(40*math.cos(degrees*math.pi/180))),int(598+(40*math.sin(degrees*math.pi/180))))

        self.browser.save_screenshot('screenie.png')
       
        
        typ = self.browser.find_element_by_xpath('/html/body/div[2]/div[5]/div')
        oppacity=typ.value_of_css_property('opacity')
        print(oppacity)
        if oppacity=="0.38":
            try:
                typ = self.browser.find_element_by_xpath("/html/body/div[13]").text
                numbers = [int(word) for word in typ.split() if word.isdigit()]
                score=numbers[0]
                reward=int(score)-int(self.score)#burayı iyi kontrol et!
                print(self.score,score)
                self.score=score
                done=0
              
                        
            except:
                done=0
                reward=0
                
        else:
           done=1
           reward=-100
        
        


        
        
        img = cv2.imread('screenie.png')
        return img,reward,done
    def reset(self):
        print("oyun yeniden başlıyorr")
        self.score=10
        time.sleep(5)
        try:
            
            self.browser.find_element_by_xpath("/html/body/div[2]/div[4]/div[1]/input").send_keys(Keys.ENTER)
        except:
            time.sleep(3)
            try:
                
                self.browser.find_element_by_xpath("/html/body/div[2]/div[4]/div[1]/input").send_keys(Keys.ENTER)
            except:
                    pass
        typ = self.browser.find_element_by_xpath('/html/body/div[2]/div[5]/div')
        oppacity=typ.value_of_css_property('opacity')
        i=1
        while (i==0):#bunu silebilirsin
          if oppacity=="0.38":
              i=0
              typ = self.browser.find_element_by_xpath('/html/body/div[2]/div[5]/div')
              oppacity=typ.value_of_css_property('opacity')
        i=1
        while (i==0):
          try:
           typ = self.browser.find_element_by_xpath("/html/body/div[13]").text
           
           
           i=0
          except:
              pass
           
        print("başlıyor")
        time.sleep(1)

        self.browser.save_screenshot('screenie.png')
        img = cv2.imread('screenie.png')
        return img




    
        
        