#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True

f = open( sys.argv[1], 'r' )
x_lst = []
y_lst = []
for line in f:
    line = line.strip()
    a = line.split()
    x_lst.append( float( a[ 0 ] ) )
    y_lst.append( float( a[ 1 ] ) )

x = np.array(x_lst)
y = np.array(y_lst)

fig, ax = plt.subplots()
ax.set_xlim( [0,10] )
ax.set_ylim( [-10,10] )
ax.plot(x, y, linewidth=2.0)
ax.set_xlabel( "t" )
ax.set_ylabel( "u(t)" )

plt.savefig( sys.argv[2] )
plt.close(fig)




