#!/usr/bin/env python3

import sys
import collections

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


    def get( self, value ):
        color = ""
        if len( self.coloring ) > 0:
            for token in self.coloring:
                if token.find( "#" ) == 0:
                    color = token
                else:
                    try:
                        if float( token ) > float( value ):
                            break
                    except ValueError:
                        color = ""
                        break

        html = ""
        if color != "":
            html += "bgcolor=\"{}\"".format(color)

        if self.sorting == "+" or self.sorting == "-":
            try:
                number = float( value )
                self.sortingData.append( number )
            except ValueError:
                self.sortingNaNs += 1

        return html

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

    def __init__( self, level, data, parentPath=None ):
        self.subcolumns = []
        self.height = 0
        self.numberOfSubcolumns = 0
        self.rowspan = 0

        self.level = level
        dataSplit = data.split( ':', 1 )
        self.label = dataSplit[ 0 ].strip()
        if len(dataSplit) == 2:
            self.attributes = dataSplit[1]
        else:
            self.attributes = ""

        if parentPath is None:
            self.path = []
        else:
            # make a copy!
            self.path = parentPath[:]
        self.path.append(self.label)

    def insertSubcolumn( self, level, label ):
        if level == self.level + 1:
            self.subcolumns.append( tableColumn( level, label, self.path ) )
        if level > self.level + 1:
            self.subcolumns[ -1 ].insertSubcolumn( level, label )

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

    def getColumnHeader( self, currentLevel ):
        if currentLevel == self.level:
            return "        <td rowspan=" + str( self.rowspan ) + " colspan=" + str( self.numberOfSubcolumns) + ">" + self.label + "</td>\n"
        return ""

    def pickLeafColumns( self, leafColumns ):
        if len( self.subcolumns ) == 0:
            leafColumns.append( self )
        else:
            for subcolumn in self.subcolumns:
                subcolumn.pickLeafColumns( leafColumns )

    def getFormating( self, value ):
        formating = columnFormating(self.attributes)
        return formating.get( value )

    def processSorting( self ):
        self.formating.processSorting()

    def __repr__(self):
        return "<tableColumn(label={}, subcolumns={})>".format(self.label, [col.label for col in self.subcolumns])



class logToHtmlConvertor:

    def __init__(self):
        self.html = ""
        self.reset()

    def reset(self):
        self.metadata = {}
        self.maxLevel = 0
        self.leafColumns = []
        self.tableColumns = collections.OrderedDict()
        self.tableRows = []

    def processFile( self, logFileName, htmlFileName ):
        # init HTML text
        self.writeHtmlHeader()

        print("Processing file", logFileName)
        logFile = open( logFileName, 'r' )
        self.readFile(logFile)
        logFile.close()

        self.writeHtmlFooter()
        print("Writing output to", htmlFileName)
        htmlFile = open( htmlFileName, 'w' )
        htmlFile.write(self.html)
        htmlFile.close()

        self.reset()
        self.html = ""

    def readFile( self, logFile ):
        # read file by lines
        lines = logFile.readlines()

        # drop comments and blank lines
        lines = [line for line in lines if line.strip() and not line.startswith("#")]

        # drop anything before the first metadata block
        while len(lines) > 0 and not lines[0].startswith(":"):
            lines.pop(0)

        while len(lines) > 0:
            self.reset()
            metadata = []
            while len(lines) > 0 and lines[0].startswith(":"):
                metadata.append(lines.pop(0))
            self.parseMetadata(metadata)

            table = []
            while len(lines) > 0 and not lines[0].startswith(":"):
                table.append(lines.pop(0))
            self.parseTable(table)

            self.writeTable()

    def parseMetadata(self, lines):
        for line in lines:
            line = line[1:]
            key, value = line.split("=", 1)
            self.metadata[key.strip()] = value.strip()

    def parseTable(self, lines):
        header = []
        body = []
        while len(lines) > 0:
            while len(lines) > 0 and lines[0].startswith("!"):
                header.append(lines.pop(0))
            while len(lines) > 0 and not lines[0].startswith("!"):
                body.append(lines.pop(0))
            self.parseTableRow(header, body)
            header = []
            body = []

    def parseTableRow(self, header, body):
        columns = []
        for line in header:
            data = line.lstrip("!")
            level = len(line) - len(data)
            label = data.strip()
            #print " Inserting column on level ", level, " and label ", label
            if level > self.maxLevel:
                self.maxLevel = level;
            if level == 1:
                columns.append( tableColumn( 1, label ) )
            if level > 1:
                columns[ -1 ].insertSubcolumn( level, label )

        # merge columns of this block with the previously parsed columns
        self.mergeColumns(columns)

        # pick leaf columns (data will be added here)
        leafColumns = self.pickLeafColumns(columns)

        # elements of the table row corresponding to the header just parsed
        elements = [line.strip() for line in body]

        if len(elements) != len(leafColumns):
            raise Exception("Error in the table format: header has {} leaf columns, but the corresponding row has {} elements.".format(len(leafColumns), len(elements)))

        row = collections.OrderedDict()
        for element, column in zip(elements, leafColumns):
            path = tuple(column.path)
            row[path] = element
        self.tableRows.append(row)

    def pickLeafColumns(self, columns):
        leafColumns = []
        for column in columns:
            column.pickLeafColumns(leafColumns)
        return leafColumns

    def mergeColumns(self, columns):
        for col in columns:
            path = tuple(col.path)
            if path in self.tableColumns:
                # merge all column attributes
                self.tableColumns[path].attributes += " " + col.attributes
                # merge new subcolumns
                currentSubPaths = [tuple(col.path) for col in self.tableColumns[path].subcolumns]
                for subcol in col.subcolumns:
                    if tuple(subcol.path) not in currentSubPaths:
                        self.tableColumns[path].subcolumns.append(subcol)
            else:
                self.tableColumns[path] = col
            self.mergeColumns(col.subcolumns)

    def mergeRows(self):
        # sort table
        self.tableRows.sort(key=lambda row: list(row.values()))

        i = 0
        while i < len(self.tableRows) - 1:
            currentRow = self.tableRows[ i ]
            nextRow = self.tableRows[ i + 1 ]

            can_merge = True
            for key, value in nextRow.items():
                if key in currentRow and currentRow[key] != value:
                    can_merge = False
                    break
            if can_merge is True:
                currentRow.update(nextRow)
                self.tableRows.pop(i + 1)
            else:
                i += 1

        # TODO: check this
        # sort again (just in case, previous sorting might compare values from
        # different columns)
        self.tableRows.sort(key=lambda row: list(row.values()))

    def countSubcolumns( self ):
        for path, col in self.tableColumns.items():
            if len(path) == 1:
                col.countSubcolumns();

    def countHeight( self ):
        for path, col in self.tableColumns.items():
            if len(path) == 1:
                col.countHeight();

    def countRowspan( self ):
        for path, col in self.tableColumns.items():
            if len(path) == 1:
                col.countRowspan( self.maxLevel )

    def recomputeLevel( self ):
        for path, col in self.tableColumns.items():
            if len(path) == 1:
                col.recomputeLevel( 1 )

    def processSorting(self):
        for path, col in self.tableColumns.items():
            if len(path) == 1:
                col.processSorting()

    def writeTable(self):
        self.mergeRows()
        self.countSubcolumns()
        self.countHeight()
        self.countRowspan()
        self.recomputeLevel()
#        self.processSorting()

        # write metadata
        self.writeMetadata()

        self.html += "<table border=1>\n"

        # write header
        self.writeColumnsHeader()

        # write data
        firstLevelColumns = [column for path, column in self.tableColumns.items() if len(path) == 1]
        leafColumns = self.pickLeafColumns(firstLevelColumns)
        for row in self.tableRows:
            self.html += "    <tr>\n"
            # walk through leafColumns to ensure correct order
            for col in leafColumns:
                path = tuple(col.path)
                if path in row:
                    value = row[path]
                    formating = col.getFormating(value)
                    self.html += "        <td {}>{}</td>\n".format(formating, value)
                else:
                    self.html += "        <td></td>\n"
            self.html += "    </tr>\n"

        self.html += "</table>\n"

    def writeMetadata(self):
        self.html += "<h2>{}</h2>\n".format(self.metadata.get("title"))
        self.html += "<table border=1>\n"
        self.html += "<tbody>\n"
        for key in sorted(self.metadata.keys()):
            self.html += "    <tr><td>{}</td><td>{}</td></tr>\n".format(key, self.metadata[key])
        self.html += "</tbody>\n"
        self.html += "</table>\n"

    def writeColumnsHeader(self):
        level = 1
        while level <= self.maxLevel:
            self.html += "    <tr>\n"
            for path, column in self.tableColumns.items():
                self.html += column.getColumnHeader( level )
            self.html += "    </tr>\n"
            level += 1

    def writeHtmlHeader(self):
        self.html += "<html>\n"
        self.html += "<body>\n"

    def writeHtmlFooter(self):
        self.html += "</body>\n"
        self.html += "</html>\n"



arguments = sys.argv[ 1: ]
logFile = arguments[ 0 ]
htmlFile = arguments[ 1 ]
logConvertor = logToHtmlConvertor()
logConvertor.processFile( logFile, htmlFile )
