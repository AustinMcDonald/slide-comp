import numpy as np
import glob
import os

from PIL import Image

# custom functions
from DefDefDefinitions import PE_Vals
from DefDefDefinitions import Image_Converter
from DefDefDefinitions import FrequencyFilterFunction
from DefDefDefinitions import FFT_Filter
from DefDefDefinitions import Unique_Circle
from DefDefDefinitions import Fit_2D_Gaussian
from DefDefDefinitions import gaussian_2d
from DefDefDefinitions import moving_average

from C_DefDef import *

from time import time

CWD = os.getcwd()
CWD = CWD+'/data/'


t1 = time()

## EXPO for time 0
#DIR = ['Exp/Poly/+0/1', 'Exp/Poly/+0/2']
#NAME = 'hist-Exp-poly.npy'
#DIR = ['Exp/Glass/+0/1', 'Exp/Glass/+0/2']
#NAME = 'hist-Exp-glass.npy'
#DIR = ['Exp/Copoly wo Ba/+0/1', 'Exp/Copoly wo Ba/+0/2']
#NAME = 'hist-Exp-woba.npy'
#DIR = ['Exp/Copoly w Ba/+0/1', 'Exp/Copoly w Ba/+0/2']
#NAME = 'hist-Exp-wba.npy'

## DES for time 0
#DIR = ['Des/Poly/+0/1', 'Des/Poly/+0/2']
#NAME = 'hist-Des-poly.npy'
#DIR = ['Des/Glass/+0/1', 'Des/Glass/+0/2']
#NAME = 'hist-Des-glass.npy'
#DIR = ['Des/Copoly wo Ba/+0/1', 'Des/Copoly wo Ba/+0/2']
#NAME = 'hist-Des-woba.npy'
#DIR = ['Des/Copoly w Ba/+0/1', 'Des/Copoly w Ba/+0/2']
#NAME = 'hist-Des-wba.npy'





datamin  = 0
datamax  = 320
numbins  = 320
bins     = np.linspace(datamin, datamax, numbins)
HIST     = np.zeros(numbins-1, dtype='int32')

for x in range(0,len(DIR)):
    
    Dir  = CWD+DIR[x]
    print("looking at "+Dir)
    end = 200
    DataFiles = glob.glob(Dir+'/*')
    DataFiles.sort()
    DataFiles = DataFiles[0:end]

    Row    = int(250)
    Col    = Row
    Yindex = int(256)
    Xindex = int(256)

    eOffset, eCoeff = PE_Vals(DataFiles[0])
    Test = Image_Converter(DataFiles[0], eOffset, eCoeff, Xindex, Yindex, Row, Col)
    Shape = Test.shape[0]
    r = 150
    Xoff = 195
    Yoff = 250
    X_circle, Y_circle = Unique_Circle(r, Xoff, Yoff)
    keep = np.where(Y_circle<Shape)
    Y_circle = Y_circle[keep]
    X_circle = X_circle[keep]
    keep = np.where(X_circle<Shape)
    Y_circle = Y_circle[keep]
    X_circle = X_circle[keep]
  
    FreqCut=0.03
    FreqCutWidth=0.04
    FilterArray = FrequencyFilterFunction(Shape,FreqCut,FreqCutWidth)


    for img in DataFiles:
        HIST_VAL = []
        eOffset, eCoeff = PE_Vals(img)
        ReducedImage = Image_Converter(img, eOffset, eCoeff, Xindex, Yindex, Row, Col)
        ReducedImage = FFT_Filter(ReducedImage, FilterArray)
        r_vals = ReducedImage[X_circle,Y_circle]

        r_cut  = ((X_circle-Xoff)**2+(Y_circle-Yoff)**2).max()
        HIST_VAL = Hist_maker(r_cut, Yoff, Xoff, ReducedImage.shape[0], ReducedImage)
        htemp, jnk = np.histogram(HIST_VAL, bins)
        HIST += htemp

np.save(NAME,HIST)
print("finished "+NAME)
t2 = time()
print(t2-t1)