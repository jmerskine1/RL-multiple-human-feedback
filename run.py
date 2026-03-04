# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:33:08 2018

@author: taku.yamagata
"""
import numpy as np
import main_oracle

main_oracle.main(
         simInfo='_C=p2-p9_Tr8_L=p2', # Filename header
         L  = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),    # probability to give a feedback
         C  = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])     # Human feedback confidence level)
         )
