#! /usr/bin/env python3

import os
import re
import math
import pandas

from TNL.LogParser import LogParser

#pandas.options.display.float_format = "{:.2f}".format
pandas.options.display.float_format = "{:.2e}".format
pandas.options.display.width = 0    # auto-detect terminal width for formatting
pandas.options.display.max_rows = None

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
df.drop(columns=['CSR Diff.Max','CSR Diff.L2'], axis=1, level=1, inplace=True )

# show matrix formats as columns
df = df.unstack()
df = df.reorder_levels([2, 0, 1], axis=1)
df.sort_index(axis=1, inplace=True)

# Drop CPU speedup
df.drop(columns=('BiEllpack Legacy', 'CPU','speedup'), axis=1, inplace=True )
df.drop(columns=('CSR', 'CPU','speedup'), axis=1, inplace=True )
df.drop(columns=('CSR Legacy', 'CPU','speedup'), axis=1, inplace=True )
df.drop(columns=('ChunkedEllpack Legacy', 'CPU','speedup'), axis=1, inplace=True )
df.drop(columns=('Ellpack', 'CPU','speedup'), axis=1, inplace=True )
df.drop(columns=('Ellpack Legacy', 'CPU','speedup'), axis=1, inplace=True )
df.drop(columns=('SlicedEllpack', 'CPU','speedup'), axis=1, inplace=True )
df.drop(columns=('SlicedEllpack Legacy', 'CPU','speedup'), axis=1, inplace=True )
df.drop(columns=('cuSparse', 'CPU'), axis=1, inplace=True )

print( "Computing speed-up of formats...")
# Add speedup compared to CSR and cuSparse
df["BiEllpack Legacy",      "CPU", "CSR speedup"]      = df["BiEllpack Legacy",      "CPU", "time"] / df["CSR",      "CPU", "time"]
df["BiEllpack Legacy",      "GPU", "cuSparse speedup"] = df["BiEllpack Legacy",      "GPU", "time"] / df["cuSparse", "GPU", "time"]
df["CSR",                   "GPU", "cuSparse speedup"] = df["CSR",                   "GPU", "time"] / df["cuSparse", "GPU", "time"]
df["CSR Legacy",            "GPU", "cuSparse speedup"] = df["CSR Legacy",            "GPU", "time"] / df["cuSparse", "GPU", "time"]
df["ChunkedEllpack Legacy", "CPU", "CSR speedup"]      = df["ChunkedEllpack Legacy", "CPU", "time"] / df["CSR",      "CPU", "time"]
df["ChunkedEllpack Legacy", "GPU", "cuSparse speedup"] = df["ChunkedEllpack Legacy", "GPU", "time"] / df["cuSparse", "GPU", "time"]
df["Ellpack Legacy",        "CPU", "CSR speedup"]      = df["Ellpack Legacy",        "CPU", "time"] / df["CSR",      "CPU", "time"]
df["Ellpack Legacy",        "GPU", "cuSparse speedup"] = df["Ellpack Legacy",        "GPU", "time"] / df["cuSparse", "GPU", "time"]
df["Ellpack",               "CPU", "CSR speedup"]      = df["Ellpack",               "CPU", "time"] / df["CSR",      "CPU", "time"]
df["Ellpack",               "GPU", "cuSparse speedup"] = df["Ellpack",               "GPU", "time"] / df["cuSparse", "GPU", "time"]
df["SlicedEllpack Legacy",  "CPU", "CSR speedup"]      = df["SlicedEllpack Legacy",  "CPU", "time"] / df["CSR",      "CPU", "time"]
df["SlicedEllpack Legacy",  "GPU", "cuSparse speedup"] = df["SlicedEllpack Legacy",  "GPU", "time"] / df["cuSparse", "GPU", "time"]
df["SlicedEllpack",         "CPU", "CSR speedup"]      = df["SlicedEllpack",         "CPU", "time"] / df["CSR",      "CPU", "time"]
df["SlicedEllpack",         "GPU", "cuSparse speedup"] = df["SlicedEllpack",         "GPU", "time"] / df["cuSparse", "GPU", "time"]

# Add speedup compared to legacy formats
df["CSR",                   "GPU", "Legacy speedup"]   = df["CSR",                   "GPU", "time"] / df["CSR Legacy",           "GPU", "time"]
df["CSR",                   "CPU", "Legacy speedup"]   = df["CSR",                   "CPU", "time"] / df["CSR Legacy",           "CPU", "time"]
df["Ellpack",               "GPU", "Legacy speedup"]   = df["Ellpack",               "GPU", "time"] / df["Ellpack Legacy",       "GPU", "time"]
df["Ellpack",               "CPU", "Legacy speedup"]   = df["Ellpack",               "CPU", "time"] / df["Ellpack Legacy",       "CPU", "time"]
df["SlicedEllpack",         "GPU", "Legacy speedup"]   = df["SlicedEllpack",         "GPU", "time"] / df["SlicedEllpack Legacy", "GPU", "time"]
df["SlicedEllpack",         "CPU", "Legacy speedup"]   = df["SlicedEllpack",         "CPU", "time"] / df["SlicedEllpack Legacy", "CPU", "time"]

print( "Exporting data frame to log.html..." )
pandas.options.display.float_format = '{:,.4f}'.format
df.to_html("log.html")

# extract columns of reference formats on GPU
print( "Preparing data for graph analysis..." )
df['cuSparse-bandwidth']=df['cuSparse','GPU','bandwidth']
df['ellpack-bandwidth']=df['Ellpack','GPU','bandwidth']
df['sliced-ellpack-bandwidth']=df['SlicedEllpack','GPU','bandwidth']

# sort by cuSparse
df.sort_values(by=["cuSparse-bandwidth"],inplace=True,ascending=False)
cuSparse_list = df['cuSparse-bandwidth'].tolist()
cuSparse_ellpack_gpu_list = df["Ellpack", "GPU", "bandwidth"].tolist();
cuSparse_ellpack_legacy_gpu_list = df["Ellpack Legacy", "GPU", "bandwidth"].tolist();
cuSparse_sliced_ellpack_gpu_list = df["SlicedEllpack", "GPU", "bandwidth"].tolist();
cuSparse_sliced_ellpack_legacy_gpu_list = df["SlicedEllpack Legacy", "GPU", "bandwidth"].tolist();
cuSparse_chunked_ellpack_legacy_gpu_list = df["ChunkedEllpack Legacy", "GPU", "bandwidth"].tolist();
cuSparse_bi_ellpack_legacy_gpu_list = df["BiEllpack Legacy", "GPU", "bandwidth"].tolist();

# sort by Ellpack
df.sort_values(by=["ellpack-bandwidth"],inplace=True,ascending=False)
ellpack_gpu_list = df["Ellpack", "GPU", "bandwidth"].tolist();
ellpack_legacy_gpu_list = df["Ellpack Legacy", "GPU", "bandwidth"].tolist();

# sort by SlicedEllpack
df.sort_values(by=["sliced-ellpack-bandwidth"],inplace=True,ascending=False)
df.sort_values(by=["sliced-ellpack-bandwidth"],inplace=True,ascending=False)
sliced_ellpack_gpu_list = df["SlicedEllpack", "GPU", "bandwidth"].tolist();
sliced_ellpack_legacy_gpu_list = df["SlicedEllpack Legacy", "GPU", "bandwidth"].tolist();

print( "Writing gnuplot files..." )

cuSparse_file = open( "cusparse.gplt", "w" )
i = 0
for x in cuSparse_list:
   if str( x ) != "nan":
      if str( cuSparse_ellpack_gpu_list[ i ] ) != "nan" and str( cuSparse_ellpack_legacy_gpu_list[ i ] ) != "nan":
         cuSparse_file.write( f"{i+1} {x} " )
         cuSparse_file.write( f"{cuSparse_ellpack_gpu_list[ i ]} {cuSparse_ellpack_legacy_gpu_list[ i ]} " )
         cuSparse_file.write( f"{cuSparse_sliced_ellpack_gpu_list[ i ]} {cuSparse_sliced_ellpack_legacy_gpu_list[ i ]} " )
         cuSparse_file.write( f"{cuSparse_chunked_ellpack_legacy_gpu_list[ i ]} {cuSparse_bi_ellpack_legacy_gpu_list[ i ]}\n" )
   i = i + 1
cuSparse_file.close()

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
ellpack_file.close()

print( "Generating Gnuplot file..." )

gnuplot_file = open( "gnuplot.gplt", "w" )
# NOTE: """...""" allows multi-line strings, r"..." disables backslash-escaping (so a single \ is just a \ in the output)
gnuplot_file.write( r"""
set terminal postscript lw 3 20 color
set grid
set xlabel 'Matrix'
set xtics 250
set ylabel 'Bandwidth GB/sec'
set output 'ellpack-vs-cusparse.eps'
plot 'cusparse.gplt' using 1:2 title 'cuSparse' with lines linewidth 2 lt rgb 'red', \
     'cusparse.gplt' using 1:3 title 'Ellpack' with dots linewidth 2 lt rgb 'green', \
     'cusparse.gplt' using 1:4 title 'Ellpack Legacy' with dots linewidth 2 lt rgb 'blue'
""")
# TODO: formatting like ^
gnuplot_file.write( "set output 'sliced-ellpack-vs-cusparse.eps'\n" )
gnuplot_file.write( "plot 'cusparse.gplt' using 1:2 title 'cuSparse' with lines linewidth 2 lt rgb 'red', \\\n" )
gnuplot_file.write( "     'cusparse.gplt' using 1:5 title 'Sliced Ellpack' with dots linewidth 2 lt rgb 'green',\\\n" )
gnuplot_file.write( "     'cusparse.gplt' using 1:6 title 'Sliced Ellpack Legacy' with dots linewidth 2 lt rgb 'blue'\n" )
gnuplot_file.write( "set output 'chunked-ellpack-vs-cusparse.eps'\n" )
gnuplot_file.write( "plot 'cusparse.gplt' using 1:2 title 'cuSparse' with lines linewidth 2 lt rgb 'red', \\\n" )
#gnuplot_file.write( "     'cusparse.gplt' using 1:7 title 'Chunked Ellpack' with dots linewidth 2 lt rgb 'green',\\\n" )
gnuplot_file.write( "     'cusparse.gplt' using 1:7 title 'Chunked Ellpack Legacy' with dots linewidth 2 lt rgb 'blue'\n" )
gnuplot_file.write( "set output 'bi-ellpack-vs-cusparse.eps'\n" )
gnuplot_file.write( "plot 'cusparse.gplt' using 1:2 title 'cuSparse' with lines linewidth 2 lt rgb 'red', \\\n" )
#gnuplot_file.write( "     'cusparse.gplt' using 1:7 title 'BiEllpack' with dots linewidth 2 lt rgb 'green',\\\n" )
gnuplot_file.write( "     'cusparse.gplt' using 1:8 title 'BiEllpack Legacy' with dots linewidth 2 lt rgb 'blue'\n" )
gnuplot_file.write( "set output 'ellpack-vs-ellpack-legacy.eps'\n" )
gnuplot_file.write( "plot 'ellpack.gplt' using 1:2 title 'Ellpack' with lines linewidth 2 lt rgb 'red', \\\n" )
gnuplot_file.write( "     'ellpack.gplt' using 1:3 title 'Ellpack Legacy' with dots linewidth 2 lt rgb 'blue'\n" )
gnuplot_file.write( "set output 'sliced-ellpack-vs-sliced-ellpack-legacy.eps'\n" )
gnuplot_file.write( "plot 'ellpack.gplt' using 1:2 title 'Ellpack' with lines linewidth 2 lt rgb 'red', \\\n" )
gnuplot_file.write( "     'ellpack.gplt' using 1:3 title 'Ellpack Legacy' with dots linewidth 2 lt rgb 'blue'\n" )
gnuplot_file.close()

print( "Executing Gnuplot ..." )
os.system( "gnuplot gnuplot.gplt" )

print( "Converting files to PDF ..." )
os.system( "epstopdf --autorotate All ellpack-vs-cusparse.eps" )
os.system( "epstopdf --autorotate All sliced-ellpack-vs-cusparse.eps" )
os.system( "epstopdf --autorotate All chunked-ellpack-vs-cusparse.eps" )
os.system( "epstopdf --autorotate All bi-ellpack-vs-cusparse.eps" )
os.system( "epstopdf --autorotate All ellpack-vs-ellpack-legacy.eps" )
os.system( "epstopdf --autorotate All sliced-ellpack-vs-sliced-ellpack-legacy.eps" )

print( "Deleting temprary files..." )
os.system( "rm cusparse.gplt" )
os.system( "rm ellpack.gplt" )
os.system( "rm sliced-ellpack.gplt" )
os.system( "rm gnuplot.gplt" )
os.system( "rm ellpack-vs-cusparse.eps" )
os.system( "rm sliced-ellpack-vs-cusparse.eps" )
os.system( "rm chunked-ellpack-vs-cusparse.eps" )
os.system( "rm bi-ellpack-vs-cusparse.eps" )
os.system( "rm ellpack-vs-ellpack-legacy.eps" )
os.system( "rm sliced-ellpack-vs-sliced-ellpack-legacy.eps" )
