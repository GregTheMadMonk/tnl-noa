#! /usr/bin/python

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "Tomas Oberhuber"
__date__ = "$May 6, 2015 8:40:59 PM$"

import TNL.Config

def generateRunScript( problemBaseName ):
    file = open( "run-" + problemBaseName, "w" )
    file.close()
    
print( "TNL Quickstart -- solver generator")
print( "----------------------------------")

definitions = {}

definitions['problemName'] = input( "Problam name:" )
definitions['problemBaseName'] = input( "Problem class base name (base name acceptable in C++ code):" )
definitions['operatorName'] = input( "Operator name:")

####
# Makefile
#
with open( TNL.Config.tnl_install_prefix+"/share/tnl-" + TNL.Config.tnl_version + "/Makefile.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( "Makefile", 'w') as file:
    file.write( templateString.format(**definitions ) )

####
# Main files
#
with open( TNL.Config.tnl_install_prefix+"/share/tnl-" + TNL.Config.tnl_version + "/main.h.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['problemBaseName']+".h", 'w') as file:
    file.write( templateString.format(**definitions ) )

with open( TNL.Config.tnl_install_prefix+"/share/tnl-" + TNL.Config.tnl_version + "/main.cu.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['problemBaseName']+"-cuda.cu", 'w') as file:
    file.write( templateString.format(**definitions ) )

with open( TNL.Config.tnl_install_prefix+"/share/tnl-" + TNL.Config.tnl_version + "/main.cpp.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['problemBaseName']+".cpp", 'w') as file:
    file.write( templateString.format(**definitions ) )

####
# Problem definition
#
with open( TNL.Config.tnl_install_prefix+"/share/tnl-" + TNL.Config.tnl_version + "/problem.h.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['problemBaseName'] + "Problem.h", 'w') as file:
    file.write( templateString.format(**definitions ) )

with open( TNL.Config.tnl_install_prefix+"/share/tnl-" + TNL.Config.tnl_version + "/problem_impl.h.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['problemBaseName'] + "Problem_impl.h", 'w') as file:
    file.write( templateString.format(**definitions ) )

####
# Operator
#
dimensions = [ '1', '2', '3' ]
for meshDimensions in dimensions:
   definitions[ 'meshDimensions' ] = meshDimensions
   key = 'operatorGridSpecializationHeader_' + meshDimensions + 'D'
   with open( TNL.Config.tnl_install_prefix+"/share/tnl-" + TNL.Config.tnl_version + "/operator-grid-specialization.h.in", 'r') as ftemp:
       templateString = ftemp.read()
   definitions[ key ] = templateString.format( **definitions )

   with open( TNL.Config.tnl_install_prefix+"/share/tnl-" + TNL.Config.tnl_version + "/explicit-laplace-grid-" + meshDimensions + "d_impl.h.in", 'r') as ftemp:
      definitions[ 'explicitScheme' ] = ftemp.read();
   with open( TNL.Config.tnl_install_prefix+"/share/tnl-" + TNL.Config.tnl_version + "/implicit-laplace-grid-" + meshDimensions + "d_impl.h.in", 'r') as ftemp:
      definitions[ 'semiimplicitScheme' ] = ftemp.read();

   key = 'operatorGridSpecializationImplementation_' + meshDimensions + 'D'
   with open( TNL.Config.tnl_install_prefix+"/share/tnl-" + TNL.Config.tnl_version + "/operator-grid-specialization_impl.h.in", 'r') as ftemp:
       templateString = ftemp.read()
   definitions[ key ] = templateString.format( **definitions )

with open( TNL.Config.tnl_install_prefix+"/share/tnl-" + TNL.Config.tnl_version + "/operator.h.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['operatorName'] + ".h", 'w') as file:
    file.write( templateString.format(**definitions ) )

with open( TNL.Config.tnl_install_prefix+"/share/tnl-" + TNL.Config.tnl_version + "/operator_impl.h.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['operatorName'] + "_impl.h", 'w') as file:
    file.write( templateString.format(**definitions ) )

####
# Right-hand side
#
with open( TNL.Config.tnl_install_prefix+"/share/tnl-" + TNL.Config.tnl_version + "/rhs.h.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['problemBaseName'] + "Rhs.h", 'w') as file:
    file.write( templateString.format(**definitions ) )

####
# Build config tag
#
with open( TNL.Config.tnl_install_prefix+"/share/tnl-" + TNL.Config.tnl_version + "/build-config-tag.h.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( definitions['problemBaseName'] + "BuildConfigTag.h", 'w') as file:
    file.write( templateString.format(**definitions ) )

####
# Run script
#
with open( TNL.Config.tnl_install_prefix+"/share/tnl-" + TNL.Config.tnl_version + "/run-script.in", 'r') as ftemp:
    templateString = ftemp.read()
with open( "run-" + definitions['problemBaseName'], 'w') as file:
    file.write( templateString.format(**definitions ) )

