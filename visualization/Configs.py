#!/usr/bin/env python
# -*- coding: utf-8 -*-


''' python inherent libs '''
import os
from collections import OrderedDict

''' third parts libs '''
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


''' custom libs '''

# COLOR = OrderedDict([("red", "#FF0000"),
#                      ("blue", "#0000FF"),
#                      ("green", "#008000"),
#                      ("purple", "#800080"),
#                      ("yellow", "#FFFF00"),
#                      ('saddlebrown', '#8B4513'),
#                      ('brown', '#A52A2A'),
#                      ("aqua", "#00FFFF"),
#                      ('midnightblue', '#191970'),
#                      ('darkgreen', '#006400'),
#                      ('darkblue', '#00008B')])


# colors_digial_map = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3, 'purple': 4, 'saddlebrown': 5, 'brown': 6, 'aqua': 7}
# digital_colors_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'purple', 5: 'saddlebrown', 6: 'brown', 7: 'aqua'}

# colors_cmp_map = {'red': plt.cm.Reds, 'blue': plt.cm.Blues, 'green': plt.cm.Greens, 'yellow': plt.cm.YlOrBr,
#                     'purple': plt.cm.Purples, 'saddlebrown': plt.cm.RdGy, 'brown': plt.cm.Oranges, 'aqua': plt.cm.Blues}


mat_colors_digial_map = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3, 'purple': 4, 'orange': 5, 'brown': 6, 'aqua': 7, 
                        'black': 8, 'pink': 9, 'tomato': 10, 'chocolate': 11, 'slateblue': 12, 'darkviolet': 13, 'coral': 14,
                        'olive': 15, 'turquoise': 16, 'seagreen': 17, 'khaki': 18, 'darkslateblue': 19, 'lightcyan': 20}
mat_digital_colors_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'purple', 5: 'orange', 6: 'brown', 7: 'aqua',
                        8: 'black', 9: 'pink', 10: 'tomato', 11: 'chocolate', 12: 'slateblue', 13: 'darkviolet', 14: 'coral',
                        15: 'olive', 16: 'turquoise', 17: 'seagreen', 18: 'khaki', 19: 'darkslateblue', 20: 'lightcyan'}



cv_colors_digial_map = {'red': 1, 'blue': 0, 'green': 2, 'yellow': 3, 'purple': 4, 'orange': 5, 'brown': 6, 'aqua': 7, 
                        'black': 8, 'pink': 9, 'tomato': 10, 'chocolate': 11, 'slateblue': 12, 'darkviolet': 13, 'coral': 14,
                        'olive': 15, 'turquoise': 16, 'seagreen': 17, 'khaki': 18, 'darkslateblue': 19, 'lightcyan': 20}
cv_digital_colors_map = {1: 'red', 0: 'blue', 2: 'green', 3: 'yellow', 4: 'purple', 5: 'orange', 6: 'brown', 7: 'aqua',
                        8: 'black', 9: 'pink', 10: 'tomato', 11: 'chocolate', 12: 'slateblue', 13: 'darkviolet', 14: 'coral',
                        15: 'olive', 16: 'turquoise', 17: 'seagreen', 18: 'khaki', 19: 'darkslateblue', 20: 'lightcyan'}

# cv_colors_cmp_map = {'red': (0,0,255), 'blue': (255,0,0), 'green': (0,255,0), 'yellow': [255,255,0], 
#                     'purple': (238,130,238), 'orange': (255,165,0), 'brown': (127,127,127), 'aqua': (75,0,130),
#                     'black': (0,0,0), 'pink': (255,255,255)}

cv_colors_cmp_map_func = lambda color_name: np.array(colors.to_rgb(color_name)) * 255

# yelloe:
#   - lower = np.array([22, 93, 0], dtype="uint8")
#   - upper = np.array([45, 255, 255], dtype="uint8")
# im[0,0,:]=[255,0,0]       # red
# im[0,1,:]=[255,165,0]     # orange
# im[0,2,:]=[255,255,0]     # yellow
# im[0,3,:]=[0,255,0]       # green
# im[0,4,:]=[0,0,255]       # blue
# im[0,5,:]=[75,0,130]      # indigo
# im[0,6,:]=[238,130,238]   # violet
# im[0,7,:]=[0,0,0]         # black
# im[0,8,:]=[127,127,127]   # grey
# im[0,9,:]=[255,255,255]   # white


if __name__=="__main__":
    for co in list(cv_colors_digial_map.keys()):
        cv_colors_cmp_map_func(co)