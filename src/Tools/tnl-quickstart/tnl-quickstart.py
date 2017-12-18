#! /usr/bin/python

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "Tomas Oberhuber"
__date__ = "$May 6, 2015 8:40:59 PM$"

import TNL

def generateRunScript( problemBaseName ):
    file = open( "run-" + problemBaseName, "w" )
    file.close()
    
print( "TNL Quickstart -- solver generator")
print( "----------------------------------")

definitions = {}

definitions['problemName'] = input( "Problem name:" )
definitions['problemBaseName'] = input( "Problem class base name (base name acceptable in C++ code):" )
definitions['operatorName'] = input( "Operator name:")

####
# Makefile
#
with open( TNL.__install_prefix__+"/share/tnl-" + TNL.__version__ + "/Makefile.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( "Makefile", 'w') as file:
    file.write( templateString.format(**definitions ) )

####
# Main files
#
with open( TNL.__install_prefix__+"/share/tnl-" + TNL.__version__ + "/main.h.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['problemBaseName']+".h", 'w') as file:
    file.write( templateString.format(**definitions ) )

with open( TNL.__install_prefix__+"/share/tnl-" + TNL.__version__ + "/main.cu.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['problemBaseName']+"-cuda.cu", 'w') as file:
    file.write( templateString.format(**definitions ) )

with open( TNL.__install_prefix__+"/share/tnl-" + TNL.__version__ + "/main.cpp.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['problemBaseName']+".cpp", 'w') as file:
    file.write( templateString.format(**definitions ) )

####
# Problem definition
#
with open( TNL.__install_prefix__+"/share/tnl-" + TNL.__version__ + "/problem.h.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['problemBaseName'] + "Problem.h", 'w') as file:
    file.write( templateString.format(**definitions ) )

with open( TNL.__install_prefix__+"/share/tnl-" + TNL.__version__ + "/problem_impl.h.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['problemBaseName'] + "Problem_impl.h", 'w') as file:
    file.write( templateString.format(**definitions ) )

####
# Operator
#
dimensions = [ '1', '2', '3' ]
for meshDimension in dimensions:
   definitions[ 'meshDimension' ] = meshDimension
   key = 'operatorGridSpecializationHeader_' + meshDimension + 'D'
   with open( TNL.__install_prefix__+"/share/tnl-" + TNL.__version__ + "/operator-grid-specialization.h.in", 'r') as ftemp:
       templateString = ftemp.read()
   definitions[ key ] = templateString.format( **definitions )

   with open( TNL.__install_prefix__+"/share/tnl-" + TNL.__version__ + "/explicit-laplace-grid-" + meshDimension + "d_impl.h.in", 'r') as ftemp:
      definitions[ 'explicitScheme' ] = ftemp.read();
   with open( TNL.__install_prefix__+"/share/tnl-" + TNL.__version__ + "/implicit-laplace-grid-" + meshDimension + "d_impl.h.in", 'r') as ftemp:
      definitions[ 'semiimplicitScheme' ] = ftemp.read();

   key = 'operatorGridSpecializationImplementation_' + meshDimension + 'D'
   with open( TNL.__install_prefix__+"/share/tnl-" + TNL.__version__ + "/operator-grid-specialization_impl.h.in", 'r') as ftemp:
       templateString = ftemp.read()
   definitions[ key ] = templateString.format( **definitions )

with open( TNL.__install_prefix__+"/share/tnl-" + TNL.__version__ + "/operator.h.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['operatorName'] + ".h", 'w') as file:
    file.write( templateString.format(**definitions ) )

with open( TNL.__install_prefix__+"/share/tnl-" + TNL.__version__ + "/operator_impl.h.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['operatorName'] + "_impl.h", 'w') as file:
    file.write( templateString.format(**definitions ) )

####
# Right-hand side
#
with open( TNL.__install_prefix__+"/share/tnl-" + TNL.__version__ + "/rhs.h.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['problemBaseName'] + "Rhs.h", 'w') as file:
    file.write( templateString.format(**definitions ) )

####
# Build config tag
#
with open( TNL.__install_prefix__+"/share/tnl-" + TNL.__version__ + "/build-config-tag.h.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['problemBaseName'] + "BuildConfigTag.h", 'w') as file:
    file.write( templateString.format(**definitions ) )

####
# Run script
#
with open( TNL.__install_prefix__+"/share/tnl-" + TNL.__version__ + "/run-script.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( "run-" + definitions['problemBaseName'], 'w') as file:
    file.write( templateString.format(**definitions ) )

