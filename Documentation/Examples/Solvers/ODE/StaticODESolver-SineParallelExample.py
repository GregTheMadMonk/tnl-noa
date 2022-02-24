#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True

f = open( sys.argv[1], 'r' )
data_lst = []
size = 0
for line in f:
   line = line.strip()
   a = line.split()
   aux = []
   for num in a:
      aux.append( float( num ) )
   data_lst.append( aux )


arrays = []
for a in data_lst:
   arrays.append( np.array( a ) )

#x = arrays[ 0 ]
#arrays.remove( 0 )
n = len( arrays )

print( n )

fig, ax = plt.subplots( 1, n-1, figsize=(15, 3), sharey=True )
#fig, ax = plt.subplots( 1, n-1, sharey=True )
idx = 0
for array in arrays:
   if idx > 0:
      ax[ idx - 1 ].plot(arrays[0], array, linewidth=2.0)
      ax[ idx - 1 ].set_xlabel( "t" )
      ax[ idx - 1 ].set_ylabel( "u(t)" )
#      ax[ idx - 1 ].axis('equal')
   idx = idx + 1

plt.savefig( sys.argv[2] )
plt.close(fig)




