from calendar import c
import re
import cv2
import numpy as np
import os

class DataProcessor:

    left_bound = 30
    char_step = 45    
    upper_bound = 240
    lower_bound = 360

    def __init__(self, source) -> None:
        self.source = source
        self.banner_names = os.listdir(source)

    def geta(self) -> None:
        return 1

    def storeModifiedClues(self, target, numReg = 1, numRot=6, numBlur=1, numNoisy=1) -> None:
        for banner in self.banner_names:
            name = banner[2:-4]
            clue = ClueModifier(os.path.join(self.source, banner), name)

            if not os.path.exists(os.path.join(target, name)):
                os.mkdir(os.path.join(self.source, name))

            for i in range(numReg):
                self.save(clue.getData(), name, target, str(0)+str(i))

            for i in range(numBlur):
                self.save(clue.getBlurred(), name, target, str(1)+str(i))

            for i in range(numNoisy):
                self.save(clue.getNoisy(0.1), name, target, str(2)+str(i))
            
            for angle in [1, 2, 3, -3, -2, -1]:                
                for noise in range(2):
                    new_clue = ClueModifier(clue.getRotated(angle))
                    if noise == 1:
                        self.saveClue(new_clue.getNoisy(0.1), name, target, str(4)+str(angle)+str(noise))
                    else:
                        self.saveClue(new_clue.getData(), name, target, str(4)+str(angle)+str(noise))
            
            for i in range(numNoisy):
                self.save(clue.getNoisy(0.15), name, target, str(3)+str(noise))
            
            return True

    def storeCleanClues(self, target) -> None:
        for banner in self.banner_names:
            name = banner[2:-4]
            clue = ClueModifier(os.path.join(self.source, banner), name)

            if not os.path.exists(os.path.join(target, name)):
                os.mkdir(os.path.join(self.source, name))

            self.save(clue.getData(), name, target, str(0))
        
        return True

    def storeChars(self, source, target) -> None:
        
        for clue in os.listdir(source):
            for sample in os.listdir(os.path.join(source, clue)):
                chars = self.parseClue(os.path.join(source, clue, sample), clue)

                for char in chars:
                    if not os.path.exists(os.path.join(target, clue)):
                        os.mkdir(os.path.join(target, clue))
                    self.save(char[0], char[1], target, len(os.listdir(os.path.join(target, clue))))

        return True

    def save(self, data, name, target, id) -> None:
        cv2.imwrite(os.path.join(target, name, name + '_' + str(id) + '.png'), data)
        
        return True

    def parseClue(self, path, clue) -> list:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        char_list = []
        step = 0

        for char in list(clue):
            char_list.append((img[:, step*self.char_step+self.left_bound:self.left_bound+(step+1)*self.char_step], char))
            step += 1
        
        return char_list

class ClueModifier:

    top = 240
    bottom = 360

    def __init__(self, path, tag) -> None:
        self.tag = tag
        
        if (path is str):
            img = cv2.imread(path)[:,:,1]
            self.data = img[:,:,1]
        else:
            self.data = path
        
    def getTag(self) -> str:
        return self.tag
    
    def getData(self) -> np.ndarray:
        return self.data[self.top:self.bottom]
    
    def getRotated(self, angle):
        rows, cols = self.data.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        new_img = cv2.warpAffine(self.data,M,(cols,rows))

        return new_img[self.top:self.bottom]
    
    def getNoisy(self, noise):
        noise = np.random.normal(0, noise, self.getData().shape).astype(self.data.dtype)
        new_img = cv2.add(self.data, noise)

        return new_img
    
    def getBlurred(self, blur=3):
        new_img = cv2.GaussianBlur(self.getData(), (blur, blur), 0)

        return new_img
