#!/usr/bin/env python

import sys;

class columnFormating:

   def __init__( self, data ):
      self.coloring = []
      self.sorting = "none"
      self.sortingFile = ""
      self.sortingData = []
      self.sortingNaNs = 0
      dataSplit = data.split( ' ' )
      currentFormating = ""
      for word in dataSplit:
         if word == "COLORING" or word == "SORT":
            currentFormating = word
            continue
         if currentFormating == "COLORING":
            self.coloring.append( word )
         if currentFormating == "SORT":
            if word == "+" or word == "-":
               self.sorting = word
            else:
               self.sortingFile = word


   def write( self, htmlFile, line ):
      color = ""
      if len( self.coloring ) > 0:
         for token in self.coloring:
            if token.find( "#" ) == 0:
               color = token
            else:
               try:
                  if float( token ) > float( line ):
                     break
               except ValueError:
                  color = ""
                  break
      if color != "":
         htmlFile.write( " bgcolor=\"" )
         htmlFile.write( color )
         htmlFile.write( "\"" )

      if self.sorting == "+" or self.sorting == "-":
         try:
            number = float( line )
            self.sortingData.append( number )
         except ValueError:
            self.sortingNaNs += 1

   def processSorting( self ):
      if self.sorting == "none":
         return
      if self.sorting == "+":
         self.sortingData.sort()
      if self.sorting == "-":
         self.sortingData.sort( reverse = True )
      sortFile = open( self.sortingFile, "w" )
      sortFile.write( "# Number of NaNs is ")
      sortFile.write( str( self.sortingNaNs ) )
      sortFile.write( "\n\n" )
      idx = 0
      for n in self.sortingData:
         sortFile.write( str( idx ) )
         sortFile.write( "\t" )
         sortFile.write( str( n ) )
         sortFile.write( "\n" )
         idx += 1
      sortFile.close()


class tableColumn:

   def __init__( self, level, data ):
      self.level = level
      self.maxLevel = 0
      self.height = 0
      self.subcolumns = []
      self.numberOfSubcolumns = 0
      self.rowspan = 0
      dataSplit = data.split( ':', 1 );
      label = dataSplit[ 0 ];   
      self.label = label.rstrip( ' ' );
      #print self.label
      if len( dataSplit ) == 2:
         self.formating = columnFormating( dataSplit[ 1 ] )
      else:
         self.formating = columnFormating( "" )

   def insertSubcolumn( self, level, label ):
      if level > self.maxLevel:
         self.maxLevel = level
      if level == self.level + 1:
         self.subcolumns.append( tableColumn( level, label ) )
      if level > self.level + 1:
         self.subcolumns[ len( self.subcolumns ) - 1 ].insertSubcolumn( level, label )

   def countSubcolumns( self ):
      if( len( self.subcolumns ) == 0 ):
         self.numberOfSubcolumns = 1
      else:
         self.numberOfSubcolumns = 0;
         for subcolumn in self.subcolumns:
            self.numberOfSubcolumns = self.numberOfSubcolumns + subcolumn.countSubcolumns()
      return self.numberOfSubcolumns

   def countHeight( self ):
      self.height = 1;
      if len( self.subcolumns ) == 0:
         return 1
      for subcolumn in self.subcolumns:
         self.height = max( self.height, subcolumn.countHeight() + 1 )
      return self.height

   def countRowspan( self, height ):
      self.rowspan = height - self.height + 1
      #print "Setting rowspan of ", self.label, " to ", self.rowspan
      for subcolumn in self.subcolumns:
         subcolumn.countRowspan( self.height - 1 )

   def recomputeLevel( self, level ):
      self.level = level
      for subcolumn in self.subcolumns:
         subcolumn.recomputeLevel( self.level + self.rowspan )

   def writeToColumnsHeader( self, htmlFile, height, currentLevel ):
      if currentLevel > self.level:
         for subcolumn in self.subcolumns:
            subcolumn.writeToColumnsHeader( htmlFile, self.height , currentLevel )
      if currentLevel == self.level:
         #print "Label  = ", self.label, " self.height = ", self.height, " height = ", height
         htmlFile.write( "            <td rowspan=" + str( self.rowspan ) + " colspan=" + str( self.numberOfSubcolumns) + ">" + self.label + "</td>\n" )

   def pickLeafColumns( self, leafColumns ):
      if len( self.subcolumns ) == 0:
         #print "Appending leaf column ", self.label
         leafColumns.append( self )
      else:
         for subcolumn in self.subcolumns:
            subcolumn.pickLeafColumns( leafColumns )

   def writeFormating( self, htmlFile, line ):
      self.formating.write( htmlFile, line )

   def processSorting( self ):
      self.formating.processSorting()
      
      
      
      
class logToHtmlConvertor:

   def __init__( self ):
      self.tableColumns = []
      self.maxLevel = 0
      self.leafColumns = []

   def processFile( self, logFileName, htmlFileName ):
      print "Processing file", logFileName 
      print "Writing output to", htmlFileName 
      logFile = open( logFileName, 'r' )
      htmlFile = open( htmlFileName, 'w' )
      self.writeHtmlHeader( htmlFile, logFile )
      htmlFile.close()
      logFile.close()

   def initColumnsStructure( self, logFile ):
      for line in logFile:
         if line[ 0 ] != '#':
            return
         data = line[1:]
         level = len( data ) - len( data.lstrip(' ') )
         level = level + 1
         label = data.lstrip( ' ')
         label = label.rstrip( '\n' )
         #print " Inserting column on level ", level, " and label ", label
         if level > self.maxLevel:
            self.maxLevel = level;
         if level == 1:
            self.tableColumns.append( tableColumn( 1, label ) )
         if level > 1:
            self.tableColumns[ len( self.tableColumns ) - 1 ].insertSubcolumn( level, label )

   def countSubcolumns( self ):
      for subcolumn in self.tableColumns:
         subcolumn.countSubcolumns();
   
   def countHeight( self ):
      for subcolumn in self.tableColumns:
         subcolumn.countHeight();
   
   def countRowspan( self ):
      for subcolumn in self.tableColumns:
         subcolumn.countRowspan( self.maxLevel )

   def recomputeLevel( self ):
      for subcolumn in self.tableColumns:
         subcolumn.recomputeLevel( 1 )

   def writeColumnsHeader( self, htmlFile ):
      level = 1
      while level <= self.maxLevel:
         #print "Writing columns on level ", level 
         htmlFile.write( "         <tr>\n")
         for column in self.tableColumns:
            column.writeToColumnsHeader( htmlFile, self.maxLevel, level )
         htmlFile.write( "         </tr>\n")
         level = level + 1

   def pickLeafColumns( self ):
      for subcolumn in self.tableColumns:
         subcolumn.pickLeafColumns( self.leafColumns )

   def writeTable( self, logFile, htmlFile ):
      firstLine = "true"
      for line in logFile:
         if len( line ) == 1 or firstLine == "true":
            if firstLine == "true":
               htmlFile.write( "         <tr>\n" )
               leafColumnsPointer = 0
               firstLine = "false"
            else:
               htmlFile.write( "         </tr>\n" )
               htmlFile.write( "         <tr>\n" )
               leafColumnsPointer = 0
         if len( line ) > 1:
            line = line.lstrip( ' ' )
            line = line.rstrip( '\n' )
            leafColumn = self.leafColumns[ leafColumnsPointer ]
            htmlFile.write( "             <td" )
            leafColumn.writeFormating( htmlFile, line )
            htmlFile.write( ">")
            htmlFile.write( line )
            htmlFile.write( "</td>\n" )
            leafColumnsPointer = leafColumnsPointer + 1
      htmlFile.write( "         </tr>\n" )

   def processSorting( self ):
      for column in self.leafColumns:
         column.processSorting()

   def writeHtmlHeader( self, htmlFile, logFile ):
      htmlFile.write( "<html>\n" )
      htmlFile.write( "   <body>\n" )
      htmlFile.write( "      <table border=1>\n")
      self.initColumnsStructure( logFile )
      self.countSubcolumns()
      self.countHeight()
      self.countRowspan()
      self.recomputeLevel()
      self.writeColumnsHeader( htmlFile )
      self.pickLeafColumns();
      self.writeTable( logFile, htmlFile )
      htmlFile.write( "      </table>\n")
      htmlFile.write( "   </body>\n" )
      htmlFile.write( "</html>\n" )
      self.processSorting()

      

arguments = sys.argv[ 1: ]
logFile = arguments[ 0 ]
htmlFile = arguments[ 1 ]
logConvertor = logToHtmlConvertor()
logConvertor.processFile( logFile, htmlFile )
