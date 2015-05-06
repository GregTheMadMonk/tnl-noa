#! /usr/bin/python

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "oberhuber"
__date__ = "$May 6, 2015 8:40:59 PM$"


def generateMakefile( solverName ):
    file = open( "Makefile", "w" )
    file.write( "")
    file.close()

print( "TNL Quickstart -- solver generator")
print( "----------------------------------")
projectName = input( "Project name: (whitespace characters are allowed)" )
solverName = input( "Solver name: (whitespace characters are NOT allowed)" )
generateMakefile( solverName )