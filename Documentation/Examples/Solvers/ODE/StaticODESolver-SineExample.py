#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import numpy as np


f = open( sys.argv[1], 'r' )
x_lst = []
y_lst = []
for line in f:
   line = line.strip()
   a = line.split()
   x_lst.append( a[ 0 ] )
   y_lst.append( a[ 1 ] )

x = np.array(x_lst)
y = np.array(y_lst)

print( x )
print( y )

fig, ax = plt.subplots()
#ax.set_aspect('auto')
ax.set_xlim( xmin=0,xmax=10 )
ax.set_ylim( [0,10] )
ax.set_xbound(lower=0, upper=10)
#ax.set(xlim=(0,10), xticks=np.arange(1, 10), ylim=(0, 10), yticks=np.arange(1, 10))
ax.plot(x, y, linewidth=2.0)
plt.xlim( xmin=0, xmax=10 )
plt.savefig( sys.argv[2] )
plt.close(fig)




