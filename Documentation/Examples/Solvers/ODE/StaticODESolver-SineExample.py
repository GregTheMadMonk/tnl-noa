#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


f = open( "sine.gplt", 'r' )
x = []
y = []
for line in f:
   line = line.strip()
   a = line.split()
   x.append( a[ 0 ] )
   y.append( a[ 1 ] )
print( x )
print( y )

fig, ax = plt.subplots()
plt.xlim( 0, 10 )
ax.set_aspect('auto')
ax.set_xlim( 0,10)
ax.set_ylim( 0,10)
ax.set_xbound(lower=0, upper=10)
#ax.set(xlim=(0,10), xticks=np.arange(1, 10), ylim=(0, 10), yticks=np.arange(1, 10))
ax.plot(x, y, linewidth=2.0)
plt.savefig( f"StaticODESolver-SineExample.pdf")
plt.close(fig)




