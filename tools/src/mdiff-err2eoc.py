#!/usr/bin/env python

import sys, string, math

arguments = sys. argv[1:]
latex_format = "no"
output_file_name = "eoc-table.txt"
input_files = []

i = 0
while i < len( arguments ):
   if arguments[ i ] == "--latex":
      latex_format = "yes"
      i = i + 1
      continue
   if arguments[ i ] == "--output-file":
      output_file_name = arguments[ i + 1 ]
      i = i + 2
      continue
   input_files. append( arguments[ i ] )
   i = i + 1

if latex_format == "yes":
   print "Writing to " + output_file_name + " in latex format"
else: 
   print "Writing to " + output_file_name + "."

h_list = []
l1_norm_list = []
l2_norm_list = []
max_norm_list = []
items = 0

for file_name in input_files:
   print "Processing file " + file_name
   file = open( file_name, "r" )
   
   l1_max = 0.0
   l_max_max = 0.0
   for line in file. readlines():
      if line[ 0:8 ] == "  Total:":
         data = string. split( line )
         h_list. append( float( data[ 1 ] ) )
         l1_norm_list. append( float( data[ 2 ] ) )
         l2_norm_list. append( float( data[ 3 ] ) )
         max_norm_list. append( float( data[ 4 ] ) )
         items = items + 1
         print line
   file. close()

h_width = 12
err_width = 15
file = open( output_file_name, "w" )
if latex_format == "no":
   file. write( " " + string. ljust( "h", h_width ) +
                string. rjust( "L1 Err.", err_width ) +
                string. rjust( "L1 EOC.", err_width ) +
                string. rjust( "L2 Err.", err_width ) +
                string. rjust( "L2 EOC.", err_width ) +
                string. rjust( "MAX Err.", err_width ) +
                string. rjust( "MAX EOC.\n", err_width ) )
   file. write( eval( "'-'*(h_width + 6 * err_width)" ) + "\n" )
if latex_format == "yes":
      file. write( "\\begin{tabular}{|r|l|l|l|l|l|l|}\\hline\n" )
      file. write( "\\raisebox{-1ex}[0ex]{$h$}& \n" )
      file. write( "\\multicolumn{2}{|c|}{\\raisebox{1ex}[3.5ex]{$\\left\| \\cdot \\right\\|_{L_1\\left(\\omega_h;\\left[0,T\\right]\\right)}^{h,\\tau}$}}& \n" )
      file. write( "\\multicolumn{2}{|c|}{\\raisebox{1ex}[3.5ex]{$\\left\| \\cdot \\right\\|_{L_2\\left(\\omega_h;\left[0,T\\right]\\right)}^{h,\\tau}$}}& \n" )
      file. write( "\\multicolumn{2}{|c|}{\\raisebox{1ex}[3.5ex]{$\\left\| \\cdot \\right\\|_{L_\\infty\\left(\\omega_h;\\left[0,T\\right]\\right)}^{h,\\tau}$}}\\\\ \\cline{2-7} \n" )
      file. write( " " + string. rjust( " ", h_width ) + "&" +
                string. rjust( "Error", err_width ) + "&" +
                string. rjust( "{\\bf EOC}", err_width ) + "&" +
                string. rjust( "Error", err_width ) + "&" +
                string. rjust( "{\\bf EOC}", err_width ) + "&" +
                string. rjust( "Error.", err_width ) + "&" +
                string. rjust( "{\\bf EOC}", err_width ) +
                "\\\\ \\hline \\hline \n")


i = 0
while i < items:
   if i == 0:
      if latex_format == "no":
         file. write( " " + string. ljust( str( h_list[ i ] ), h_width ) + 
                      string. rjust( "%.4g" % l1_norm_list[ i ], err_width ) + 
                      string. rjust( "%.4g" % l2_norm_list[ i ], 2 * err_width ) +
                      string. rjust( "%.4g" % max_norm_list[ i ], 2 * err_width ) + "\n" )
      if latex_format == "yes":
         file. write( " " + string. ljust( str( h_list[ i ] ), h_width ) + "&" +
                      string. rjust( "%.4g" % l1_norm_list[ i ], err_width ) + "&" + 
                      string. rjust( " ", err_width ) + "&"+ 
                      string. rjust( "%.4g" % l2_norm_list[ i ], err_width ) + "&" +
                      string. rjust( " ", err_width ) + "&" +
                      string. rjust( "%.4g" % max_norm_list[ i ], err_width ) + "&" +
                      string. rjust( " ", err_width ) + "\\\\\n" )
      i = i + 1;
      continue
   if h_list[ i ] == h_list[ i - 1 ]:
      print "Unable to count eoc since h[ " + \
      str( i ) + " ] = h[ " + str( i - 1 ) + \
      " ] = " + str( h_list[ i ] ) + ". \n"
      file. write( " eoc error:  h[ " + \
      str( i ) + " ] = h[ " + str( i - 1 ) + \
      " ] = " + str( h_list[ i ] ) + ". \n" )
   else:
      h_ratio = math. log( h_list[ i ] / h_list[ i - 1 ] )
      l1_ratio = math. log( l1_norm_list[ i ] / l1_norm_list[ i - 1 ] )
      l2_ratio = math. log( l2_norm_list[ i ] / l2_norm_list[ i - 1 ] )
      max_ratio = math. log( max_norm_list[ i ] / max_norm_list[ i - 1 ] )
      if latex_format == "no":
         file. write( " " + string. ljust( str( h_list[ i ] ), h_width ) +
                      string. rjust( "%.4g" % l1_norm_list[ i ], err_width ) +
                      string. rjust( "%.4g" % ( l1_ratio / h_ratio ), err_width ) +
                      string. rjust( "%.4g" % l2_norm_list[ i ], err_width ) +
                      string. rjust( "%.4g" % ( l2_ratio / h_ratio ), err_width ) +
                      string. rjust( "%.4g" % max_norm_list[ i ], err_width ) +
                      string. rjust( "%.4g" % ( max_ratio / h_ratio ), err_width ) + "\n" )
      if latex_format == "yes":
         file. write( " " + string. ljust( str( h_list[ i ] ), h_width ) + "&" +
                      string. rjust( "%.4g" % l1_norm_list[ i ], err_width ) + "&" +
                      string. rjust( "{\\bf " + "%.4g" % ( l1_ratio / h_ratio ) + "}", err_width ) + "&" +
                      string. rjust( "%.4g" % l2_norm_list[ i ], err_width ) + "&" +
                      string. rjust( "{\\bf " + "%.4g" % ( l2_ratio / h_ratio ) + "}", err_width ) + "&" +
                      string. rjust( "%.4g" % max_norm_list[ i ], err_width ) + "&" +
                      string. rjust( "{\\bf " + "%.4g" % ( max_ratio / h_ratio ) + "}", err_width ) + "\\\\\n" )
   i = i + 1

if latex_format == "yes":
   file. write( "\\hline \n" )
   file. write( "\\end{tabular} \n" )
