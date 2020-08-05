#! /usr/bin/env python3

import os
import re
import math
import pandas

from collections import defaultdict
from TNL.LogParser import LogParser

""" 
Sparse matrix formats as they appear in the log file.
"""
cpu_matrix_formats = [ 'CSR', 
                       'Ellpack', 'Ellpack Legacy',
                       'SlicedEllpack', 'SlicedEllpack Legacy',
                       'ChunkedEllpack', 'ChunkedEllpack Legacy',
                       'BiEllpack', 'BiEllpack Legacy' ]

gpu_matrix_formats = [ 'CSR Legacy Scalar', 'CSR Legacy Vector', 'CSR Legacy MultiVector',
                       'CSR Legacy Light', 'CSR Legacy Light2', 'CSR Legacy Light3', 'CSR Legacy Light4', 'CSR Legacy Light5', 'CSR Legacy Light6', 'CSR Legacy LightWithoutAtomic', 
                       'CSR Legacy Adaptive',
                       'Ellpack', 'Ellpack Legacy',
                       'SlicedEllpack', 'SlicedEllpack Legacy',
                       'ChunkedEllpack', 'ChunkedEllpack Legacy',
                       'BiEllpack', 'BiEllpack Legacy' ]

#pandas.options.display.float_format = "{:.2f}".format
pandas.options.display.float_format = "{:.2e}".format
pandas.options.display.width = 0    # auto-detect terminal width for formatting
pandas.options.display.max_rows = None

def slugify(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def parse_file(fname):
    parser = LogParser()
    for metadata, df in parser.readFile(fname):
        yield df

def calculate_efficiency(df, nodes_col_index, base_column=None):
    if base_column is None:
        base_column = df[df.columns[0]]
    eff_rows = []
    for i in df.index:
        row = df.loc[i]
        eff_row = row.copy()
        eff_idx = ("eff", *row.name[1:])
        base = base_column[i]
        for j in row.index:
            if isinstance(j, int):
                n = j
            else:
                n = j[nodes_col_index]
            eff_row[j] = base / row[j] / n
        eff_rows.append(eff_row)
    eff_df = pandas.DataFrame(eff_rows)
    eff_df.index = pandas.MultiIndex.from_tuples(eff_df.index)
    eff_df = eff_df.rename(index={"time": "eff"})
    return df.append(eff_df)

log_files = ["sparse-matrix-benchmark.log"]
print( "Parsing log file..." )

dfs = []
for f in log_files:
    for df in parse_file(f):
        dfs.append(df)

df = pandas.concat(dfs)

## Post-processing
print( "Postprocessing data frame..." )
# Drop norms of results differences
#df.drop(columns=['CSR Diff.Max','CSR Diff.L2'], axis=1, level=1, inplace=True )

# show matrix formats as columns
df = df.unstack()
df = df.reorder_levels([2, 0, 1], axis=1)
df.sort_index(axis=1, inplace=True)

# Drop CPU speedup
for cpu_format in cpu_matrix_formats:
   df.drop(columns=( cpu_format, 'CPU','speedup'), axis=1, inplace=True )

#print( "Exporting data frame to log.html..." )
#pandas.options.display.float_format = '{:,.4f}'.format
#df.to_html("log.html")

print( "Computing speed-up of formats...")
# Add speedup compared to CSR and cuSparse
for cpu_format in cpu_matrix_formats:
   if cpu_format != 'CSR':
      df[cpu_format, "CPU", "CSR speedup"] = df[cpu_format, "CPU", "time"] / df["CSR","CPU", "time"]

for gpu_format in gpu_matrix_formats:
   df[ gpu_format, "GPU", "cuSparse speedup"] = df[ gpu_format,"GPU", "time"] / df["cuSparse", "GPU", "time"]

# Add speedup compared to legacy formats
df["CSR",                   "GPU", "Legacy speedup"]   = df["CSR",                   "GPU", "time"] / df["CSR Legacy Scalar",    "GPU", "time"]
df["CSR",                   "CPU", "Legacy speedup"]   = df["CSR",                   "CPU", "time"] / df["CSR Legacy Scalar",    "CPU", "time"]
df["Ellpack",               "GPU", "Legacy speedup"]   = df["Ellpack",               "GPU", "time"] / df["Ellpack Legacy",       "GPU", "time"]
df["Ellpack",               "CPU", "Legacy speedup"]   = df["Ellpack",               "CPU", "time"] / df["Ellpack Legacy",       "CPU", "time"]
df["SlicedEllpack",         "GPU", "Legacy speedup"]   = df["SlicedEllpack",         "GPU", "time"] / df["SlicedEllpack Legacy", "GPU", "time"]
df["SlicedEllpack",         "CPU", "Legacy speedup"]   = df["SlicedEllpack",         "CPU", "time"] / df["SlicedEllpack Legacy", "CPU", "time"]
df["BiEllpack",             "GPU", "Legacy speedup"]   = df["BiEllpack",             "GPU", "time"] / df["BiEllpack Legacy",     "GPU", "time"]
df["BiEllpack",             "CPU", "Legacy speedup"]   = df["BiEllpack",             "CPU", "time"] / df["BiEllpack Legacy",     "CPU", "time"]

print( "Exporting data frame to log.html..." )
pandas.options.display.float_format = '{:,.4f}'.format
df.to_html("log.html")

# extract columns of reference formats on GPU
print( "Preparing data for graph analysis..." )
df['cuSparse-bandwidth'                        ] = df[ 'cuSparse','GPU','bandwidth']
for gpu_format in gpu_matrix_formats:
   df[ gpu_format + ' Bandwidth' ] = df[ gpu_format,'GPU','bandwidth']

# sort by cuSparse
df.sort_values(by=["cuSparse-bandwidth"],inplace=True,ascending=False)
cuSparse_list = df['cuSparse-bandwidth'].tolist()
cusparse_comparison = defaultdict( list )
for gpu_format in gpu_matrix_formats:
   cusparse_comparison[ gpu_format ] = df[ gpu_format, "GPU", "bandwidth" ].tolist()

# sort by Ellpack
df.sort_values(by=["Ellpack Bandwidth"],inplace=True,ascending=False)
ellpack_gpu_list = df["Ellpack", "GPU", "bandwidth"].tolist();
ellpack_legacy_gpu_list = df["Ellpack Legacy", "GPU", "bandwidth"].tolist();

# sort by SlicedEllpack
df.sort_values(by=["SlicedEllpack Bandwidth"],inplace=True,ascending=False)
sliced_ellpack_gpu_list = df["SlicedEllpack", "GPU", "bandwidth"].tolist();
sliced_ellpack_legacy_gpu_list = df["SlicedEllpack Legacy", "GPU", "bandwidth"].tolist();

# sort by ChunkedEllpack
df.sort_values(by=["ChunkedEllpack Bandwidth"],inplace=True,ascending=False)
chunked_ellpack_gpu_list = df["ChunkedEllpack", "GPU", "bandwidth"].tolist();
chunked_ellpack_legacy_gpu_list = df["ChunkedEllpack Legacy", "GPU", "bandwidth"].tolist();

# sort by BiEllpack
df.sort_values(by=["BiEllpack Bandwidth"],inplace=True,ascending=False)
bi_ellpack_gpu_list = df["BiEllpack", "GPU", "bandwidth"].tolist();
bi_ellpack_legacy_gpu_list = df["BiEllpack Legacy", "GPU", "bandwidth"].tolist();

print( "Writing gnuplot files..." )

for gpu_format in gpu_matrix_formats:
   filename = "cusparse-" + slugify( gpu_format ) + ".gplt"
   data = cusparse_comparison[ gpu_format ]
   print( "Writing to " + filename + "..." );
   out_file = open( filename, "w" )
   i = 0
   for x in cuSparse_list:
      if str( x ) != "nan":
         if ( str(cusparse_comparison[ gpu_format ][ i ] ) != "nan" ):
            out_file.write( f"{i+1} {x} {data[ i ]} \n" )
            i = i + 1;
   out_file.close()

ellpack_file = open( "ellpack.gplt", "w" )
i = 0;
for x in ellpack_gpu_list:
   if str( x ) != "nan":
      if str( ellpack_legacy_gpu_list[ i ] ) != "nan":
         ellpack_file.write( f"{i+1} {x} {ellpack_legacy_gpu_list[ i ]}\n" )
   i = i + 1
ellpack_file.close()

sliced_ellpack_file = open( "sliced-ellpack.gplt", "w" )
i = 0;
for x in sliced_ellpack_gpu_list:
   if str( x ) != "nan":
      if str( sliced_ellpack_legacy_gpu_list[ i ] ) != "nan":
         sliced_ellpack_file.write( f"{i+1} {x} {sliced_ellpack_legacy_gpu_list[ i ]}\n" )
   i = i + 1
sliced_ellpack_file.close()

chunked_ellpack_file = open( "chunked-ellpack.gplt", "w" )
i = 0;
for x in chunked_ellpack_gpu_list:
   if str( x ) != "nan":
      if str( chunked_ellpack_legacy_gpu_list[ i ] ) != "nan":
         chunked_ellpack_file.write( f"{i+1} {x} {chunked_ellpack_legacy_gpu_list[ i ]}\n" )
   i = i + 1
chunked_ellpack_file.close()

bi_ellpack_file = open( "bi-ellpack.gplt", "w" )
i = 0;
for x in bi_ellpack_gpu_list:
   if str( x ) != "nan":
      if str( bi_ellpack_legacy_gpu_list[ i ] ) != "nan":
         bi_ellpack_file.write( f"{i+1} {x} {bi_ellpack_legacy_gpu_list[ i ]}\n" )
   i = i + 1
bi_ellpack_file.close()

print( "Generating Gnuplot file..." )


gnuplot_file = open( "gnuplot.gplt", "w" )
gnuplot_file.write( r"""
set terminal postscript lw 3 20 color
set grid
set xlabel 'Matrix'
set xtics 250
set ylabel 'Bandwidth GB/sec'
""" )
for gpu_format in gpu_matrix_formats:
   filename = "cusparse-" + slugify( gpu_format ) + ".gplt"
   gnuplot_file.write( f"set output 'cusparse-vs-{slugify(gpu_format)}.eps' \n" )
   gnuplot_file.write( f"plot '{filename}' using 1:2 title '' with dots linewidth 2 lt rgb 'red', " )
   gnuplot_file.write( f" '{filename}' using 1:2 title 'cuSparse' with lines linewidth 0.5 lt rgb 'red', " )
   gnuplot_file.write( f" '{filename}' using 1:3 title '' with dots linewidth 2 lt rgb 'green', " )
   gnuplot_file.write( f" '{filename}' using 1:3 title '{gpu_format}' with lines linewidth 0.5 lt rgb 'green'  \n" )


gnuplot_file.write( r"""
set output 'ellpack-vs-ellpack-legacy.eps'                                                              
plot 'ellpack.gplt' using 1:2 title '' with dots linewidth 2 lt rgb 'red',                                      \
     'ellpack.gplt' using 1:2 title 'Ellpack' with lines linewidth 0.5 lt rgb 'red',                            \
     'ellpack.gplt' using 1:3 title '' with dots linewidth 2 lt rgb 'blue',                                     \
     'ellpack.gplt' using 1:3 title 'Ellpack Legacy' with lines linewidth 0.5 lt rgb 'blue'                
set output 'sliced-ellpack-vs-sliced-ellpack-legacy.eps'                                                
plot 'sliced-ellpack.gplt' using 1:2 title '' with dots linewidth 2 lt rgb 'red',                               \
     'sliced-ellpack.gplt' using 1:2 title 'SlicedEllpack' with lines linewidth 0.5 lt rgb 'red',               \
     'sliced-ellpack.gplt' using 1:3 title '' with dots linewidth 2 lt rgb 'blue',                              \
     'sliced-ellpack.gplt' using 1:3 title 'SlicedEllpack Legacy' with lines linewidth 0.5 lt rgb 'blue'   
set output 'chunked-ellpack-vs-chunked-ellpack-legacy.eps'                                                        
plot 'chunked-ellpack.gplt' using 1:2 title '' with dots linewidth 2 lt rgb 'red',                              \
     'chunked-ellpack.gplt' using 1:2 title 'ChunkedEllpack' with lines linewidth 0.5 lt rgb 'red',             \
     'chunked-ellpack.gplt' using 1:3 title '' with dots linewidth 2 lt rgb 'blue',                             \
     'chunked-ellpack.gplt' using 1:3 title 'ChunkedEllpack Legacy' with lines linewidth 0.5 lt rgb 'blue'
set output 'bi-ellpack-vs-bi-ellpack-legacy.eps'                                                        
plot 'bi-ellpack.gplt' using 1:2 title '' with dots linewidth 2 lt rgb 'red',                                   \
     'bi-ellpack.gplt' using 1:2 title 'BiEllpack' with lines linewidth 0.5 lt rgb 'red',                       \
     'bi-ellpack.gplt' using 1:3 title '' with dots linewidth 2 lt rgb 'blue',                                  \
     'bi-ellpack.gplt' using 1:3 title 'BiEllpack Legacy' with lines linewidth 0.5 lt rgb 'blue'
""")
gnuplot_file.close()

print( "Executing Gnuplot ..." )
os.system( "gnuplot gnuplot.gplt" )

print( "Converting files to PDF ..." )
for gpu_format in gpu_matrix_formats:
   filename = "cusparse-vs-" + slugify( gpu_format ) + ".eps"
   os.system( f"epstopdf --autorotate All {filename}" )

os.system( "epstopdf --autorotate All ellpack-vs-ellpack-legacy.eps" )
os.system( "epstopdf --autorotate All sliced-ellpack-vs-sliced-ellpack-legacy.eps" )
os.system( "epstopdf --autorotate All chunked-ellpack-vs-chunked-ellpack-legacy.eps" )
os.system( "epstopdf --autorotate All bi-ellpack-vs-bi-ellpack-legacy.eps" )

print( "Deleting temprary files..." )
#os.system( "rm cusparse.gplt" )
#os.system( "rm ellpack.gplt" )
#os.system( "rm sliced-ellpack.gplt" )
#os.system( "rm gnuplot.gplt" )
#os.system( "rm ellpack-vs-cusparse.eps" )
#os.system( "rm sliced-ellpack-vs-cusparse.eps" )
#os.system( "rm chunked-ellpack-vs-cusparse.eps" )
#os.system( "rm bi-ellpack-vs-cusparse.eps" )
#os.system( "rm ellpack-vs-ellpack-legacy.eps" )
#os.system( "rm sliced-ellpack-vs-sliced-ellpack-legacy.eps" )
