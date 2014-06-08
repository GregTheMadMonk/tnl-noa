#!/usr/bin/env python

import sys;


class logToHtmlConvertor:

   def processFile( self, logFileName, htmlFileName ):
      
      print "Processing file", logFileName 
      print "Writing output to", htmlFileName 
      logFile = open( logFileName, 'r' )
      htmlFile = open( htmlFileName, 'w' )
      self.writeHtmlHeader( htmlFile, logFile )
      htmlFile.close()
      logFile.close()

   def writeHtmlHeader( self, htmlFile, logFile ):
      htmlFile.write("<html>" )
      htmlFile.write("   <body>" )
      for line in logFile:
         if line[ 0 ] != '#':
            return
         data = line[1:]
         leadingSpaces = len( data ) - len( data.lstrip(' ') )
         print "Leading spaces", leadingSpaces
      

arguments = sys.argv[ 1: ]
logFile = arguments[ 0 ]
htmlFile = arguments[ 1 ]
logConvertor = logToHtmlConvertor()
logConvertor.processFile( logFile, htmlFile )
