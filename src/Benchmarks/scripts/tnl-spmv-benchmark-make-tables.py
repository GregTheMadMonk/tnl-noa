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
#df.drop(columns=['CSR Diff.Max','CSR Diff.L2'], axis=1, level=1, inplace=True )

# show matrix formats as columns
df = df.unstack()
df = df.reorder_levels([2, 0, 1], axis=1)
df.sort_index(axis=1, inplace=True)

# Drop CPU speedup
df.drop(columns=('BiEllpack Legacy', 'CPU','speedup'), axis=1, inplace=True )
df.drop(columns=('BiEllpack', 'CPU','speedup'), axis=1, inplace=True )
df.drop(columns=('CSR', 'CPU','speedup'), axis=1, inplace=True )

#df.drop(columns=('CSR Legacy Adaptive', 'CPU','speedup'), axis=1, inplace=True )
#df.drop(columns=('CSR Legacy Light', 'CPU','speedup'), axis=1, inplace=True )
#df.drop(columns=('CSR Legacy LightWithoutAtomic', 'CPU','speedup'), axis=1, inplace=True )
#df.drop(columns=('CSR Legacy Scalar', 'CPU','speedup'), axis=1, inplace=True )
#df.drop(columns=('CSR Legacy Stream', 'CPU','speedup'), axis=1, inplace=True )
#df.drop(columns=('CSR Legacy Vector', 'CPU','speedup'), axis=1, inplace=True )
#df.drop(columns=('CSR Legacy MultiVector', 'CPU','speedup'), axis=1, inplace=True )

df.drop(columns=('ChunkedEllpack Legacy', 'CPU','speedup'), axis=1, inplace=True )
df.drop(columns=('Ellpack', 'CPU','speedup'), axis=1, inplace=True )
df.drop(columns=('Ellpack Legacy', 'CPU','speedup'), axis=1, inplace=True )
df.drop(columns=('SlicedEllpack', 'CPU','speedup'), axis=1, inplace=True )
df.drop(columns=('SlicedEllpack Legacy', 'CPU','speedup'), axis=1, inplace=True )
#df.drop(columns=('cuSparse', 'CPU'), axis=1, inplace=True )

#print( "Exporting data frame to log.html..." )
#pandas.options.display.float_format = '{:,.4f}'.format
#df.to_html("log.html")

print( "Computing speed-up of formats...")
# Add speedup compared to CSR and cuSparse

df["BiEllpack Legacy",              "CPU", "CSR speedup"]      = df["BiEllpack Legacy",              "CPU", "time"] / df["CSR",      "CPU", "time"]
df["BiEllpack Legacy",              "GPU", "cuSparse speedup"] = df["BiEllpack Legacy",              "GPU", "time"] / df["cuSparse", "GPU", "time"]
df["BiEllpack",                     "CPU", "CSR speedup"]      = df["BiEllpack",                     "CPU", "time"] / df["CSR",      "CPU", "time"]
df["BiEllpacky",                    "GPU", "cuSparse speedup"] = df["BiEllpack",                     "GPU", "time"] / df["cuSparse", "GPU", "time"]
df["CSR",                           "GPU", "cuSparse speedup"] = df["CSR",                           "GPU", "time"] / df["cuSparse", "GPU", "time"]
#df["CSR Legacy Adaptive",           "GPU", "cuSparse speedup"] = df["CSR Legacy Adaptive",           "GPU", "time"] / df["cuSparse", "GPU", "time"]
#df["CSR Legacy Light",              "GPU", "cuSparse speedup"] = df["CSR Legacy Light",              "GPU", "time"] / df["cuSparse", "GPU", "time"]
#df["CSR Legacy LightWithoutAtomic", "GPU", "cuSparse speedup"] = df["CSR Legacy LightWithoutAtomic", "GPU", "time"] / df["cuSparse", "GPU", "time"]
#df["CSR Legacy Scalar",             "GPU", "cuSparse speedup"] = df["CSR Legacy Scalar",             "GPU", "time"] / df["cuSparse", "GPU", "time"]
#df["CSR Legacy Vector",             "GPU", "cuSparse speedup"] = df["CSR Legacy Vector",             "GPU", "time"] / df["cuSparse", "GPU", "time"]
#df["CSR Legacy MultiVector",        "GPU", "cuSparse speedup"] = df["CSR Legacy MultiVector",        "GPU", "time"] / df["cuSparse", "GPU", "time"]
df["ChunkedEllpack Legacy",         "CPU", "CSR speedup"]      = df["ChunkedEllpack Legacy",         "CPU", "time"] / df["CSR",      "CPU", "time"]
df["ChunkedEllpack Legacy",         "GPU", "cuSparse speedup"] = df["ChunkedEllpack Legacy",         "GPU", "time"] / df["cuSparse", "GPU", "time"]
df["Ellpack Legacy",                "CPU", "CSR speedup"]      = df["Ellpack Legacy",                "CPU", "time"] / df["CSR",      "CPU", "time"]
df["Ellpack Legacy",                "GPU", "cuSparse speedup"] = df["Ellpack Legacy",                "GPU", "time"] / df["cuSparse", "GPU", "time"]
df["Ellpack",                       "CPU", "CSR speedup"]      = df["Ellpack",                       "CPU", "time"] / df["CSR",      "CPU", "time"]
df["Ellpack",                       "GPU", "cuSparse speedup"] = df["Ellpack",                       "GPU", "time"] / df["cuSparse", "GPU", "time"]
df["SlicedEllpack Legacy",          "CPU", "CSR speedup"]      = df["SlicedEllpack Legacy",          "CPU", "time"] / df["CSR",      "CPU", "time"]
df["SlicedEllpack Legacy",          "GPU", "cuSparse speedup"] = df["SlicedEllpack Legacy",          "GPU", "time"] / df["cuSparse", "GPU", "time"]
df["SlicedEllpack",                 "CPU", "CSR speedup"]      = df["SlicedEllpack",                 "CPU", "time"] / df["CSR",      "CPU", "time"]
df["SlicedEllpack",                 "GPU", "cuSparse speedup"] = df["SlicedEllpack",                 "GPU", "time"] / df["cuSparse", "GPU", "time"]

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
#df['csr-legacy-adaptive-bandwidth'             ] = df[ 'CSR Legacy Adaptive','GPU','bandwidth']
#df['csr-legacy-light-bandwidth'                ] = df[ 'CSR Legacy Light','GPU','bandwidth']
#df['csr-legacy-light-without-atomic-bandwidth' ] = df[ 'CSR Legacy LightWithoutAtomic','GPU','bandwidth']
#df['csr-legacy-scalar-bandwidth'               ] = df[ 'CSR Legacy Scalar','GPU','bandwidth']
#df['csr-legacy-vector-bandwidth'               ] = df[ 'CSR Legacy Vector','GPU','bandwidth']
#df['csr-legacy-multi-vector-bandwidth'         ] = df[ 'CSR Legacy MultiVector','GPU','bandwidth']
df['ellpack-bandwidth'                         ] = df[ 'Ellpack','GPU','bandwidth']
df['sliced-ellpack-bandwidth'                  ] = df[ 'SlicedEllpack','GPU','bandwidth']
df['chunked-ellpack-bandwidth'                 ] = df[ 'ChunkedEllpack','GPU','bandwidth']
df['bi-ellpack-bandwidth'                      ] = df[ 'BiEllpack','GPU','bandwidth']

# sort by cuSparse
df.sort_values(by=["cuSparse-bandwidth"],inplace=True,ascending=False)
cuSparse_list = df['cuSparse-bandwidth'].tolist()
#cuSparse_csr_legacy_adaptive_gpu_list               = df[ "CSR Legacy Adaptive", "GPU", "bandwidth"].tolist();
#cuSparse_csr_legacy_light_gpu_list                  = df[ "CSR Legacy Light", "GPU", "bandwidth"].tolist();
#cuSparse_csr_legacy_light_without_atomic_gpu_list   = df[ "CSR Legacy LightWithoutAtomic", "GPU", "bandwidth"].tolist();
#cuSparse_csr_legacy_scalar_gpu_list                 = df[ "CSR Legacy Scalar", "GPU", "bandwidth"].tolist();
#cuSparse_csr_legacy_vector_gpu_list                 = df[ "CSR Legacy Vector", "GPU", "bandwidth"].tolist();
#cuSparse_csr_legacy_multivector_gpu_list            = df[ "CSR Legacy MultiVector", "GPU", "bandwidth"].tolist();
cuSparse_ellpack_gpu_list                           = df[ "Ellpack", "GPU", "bandwidth"].tolist();
cuSparse_ellpack_legacy_gpu_list                    = df[ "Ellpack Legacy", "GPU", "bandwidth"].tolist();
cuSparse_sliced_ellpack_gpu_list                    = df[ "SlicedEllpack", "GPU", "bandwidth"].tolist();
cuSparse_sliced_ellpack_legacy_gpu_list             = df[ "SlicedEllpack Legacy", "GPU", "bandwidth"].tolist();
cuSparse_chunked_ellpack_legacy_gpu_list            = df[ "ChunkedEllpack Legacy", "GPU", "bandwidth"].tolist();
cuSparse_chunked_ellpack_gpu_list                   = df[ "ChunkedEllpack", "GPU", "bandwidth"].tolist();
cuSparse_bi_ellpack_legacy_gpu_list                 = df[ "BiEllpack Legacy", "GPU", "bandwidth"].tolist();
cuSparse_bi_ellpack_gpu_list                        = df[ "BiEllpack", "GPU", "bandwidth"].tolist();

# sort by Ellpack
df.sort_values(by=["ellpack-bandwidth"],inplace=True,ascending=False)
ellpack_gpu_list = df["Ellpack", "GPU", "bandwidth"].tolist();
ellpack_legacy_gpu_list = df["Ellpack Legacy", "GPU", "bandwidth"].tolist();

# sort by SlicedEllpack
df.sort_values(by=["sliced-ellpack-bandwidth"],inplace=True,ascending=False)
df.sort_values(by=["sliced-ellpack-bandwidth"],inplace=True,ascending=False)
sliced_ellpack_gpu_list = df["SlicedEllpack", "GPU", "bandwidth"].tolist();
sliced_ellpack_legacy_gpu_list = df["SlicedEllpack Legacy", "GPU", "bandwidth"].tolist();

# sort by ChunkedEllpack
df.sort_values(by=["chunked-ellpack-bandwidth"],inplace=True,ascending=False)
df.sort_values(by=["chunked-ellpack-bandwidth"],inplace=True,ascending=False)
chunked_ellpack_gpu_list = df["ChunkedEllpack", "GPU", "bandwidth"].tolist();
chunked_ellpack_legacy_gpu_list = df["ChunkedEllpack Legacy", "GPU", "bandwidth"].tolist();

# sort by BiEllpack
df.sort_values(by=["bi-ellpack-bandwidth"],inplace=True,ascending=False)
df.sort_values(by=["bi-ellpack-bandwidth"],inplace=True,ascending=False)
bi_ellpack_gpu_list = df["BiEllpack", "GPU", "bandwidth"].tolist();
bi_ellpack_legacy_gpu_list = df["BiEllpack Legacy", "GPU", "bandwidth"].tolist();

print( "Writing gnuplot files..." )

cuSparse_file = open( "cusparse.gplt", "w" )
i = 0
for x in cuSparse_list:
   if str( x ) != "nan":
      if ( #str( cuSparse_csr_legacy_adaptive_gpu_list[ i ] ) != "nan" and
         #str( cuSparse_csr_legacy_light_gpu_list[ i ] ) != "nan" and 
         #str( cuSparse_csr_legacy_light_without_atomic_gpu_list[ i ] ) != "nan" and 
         #str( cuSparse_csr_legacy_scalar_gpu_list[ i ] ) != "nan" and 
         #str( cuSparse_csr_legacy_vector_gpu_list[ i ] ) != "nan" and 
         #str( cuSparse_csr_legacy_multivector_gpu_list[ i ] ) != "nan" and 
         str( cuSparse_ellpack_gpu_list[ i ] ) != "nan" and 
         str( cuSparse_ellpack_legacy_gpu_list[ i ] ) != "nan" and
         str( cuSparse_sliced_ellpack_gpu_list[ i ] ) != "nan" and 
         str( cuSparse_sliced_ellpack_legacy_gpu_list[ i ] ) != "nan" and
         str( cuSparse_chunked_ellpack_gpu_list[ i ] ) != "nan" and 
         str( cuSparse_chunked_ellpack_legacy_gpu_list[ i ] ) != "nan" and
         str( cuSparse_bi_ellpack_gpu_list[ i ] ) != "nan" and 
         str( cuSparse_bi_ellpack_legacy_gpu_list[ i ] ) != "nan" ):
            cuSparse_file.write( f"{i+1} {x} " )                                                                                        # 1 2
            cuSparse_file.write( f"0 " ) #{cuSparse_csr_legacy_adaptive_gpu_list[ i ]} " )                                                     # 3
            cuSparse_file.write( f"0 " ) #{cuSparse_csr_legacy_light_gpu_list[ i ]} " )                                                        # 4
            cuSparse_file.write( f"0 " ) #{cuSparse_csr_legacy_light_without_atomic_gpu_list[ i ]} " )                                         # 5
            cuSparse_file.write( f"0 " ) #{cuSparse_csr_legacy_scalar_gpu_list[ i ]} " )                                                       # 6
            cuSparse_file.write( f"0 " ) #{cuSparse_csr_legacy_vector_gpu_list[ i ]} " )                                                       # 7
            cuSparse_file.write( f"0 " ) #{cuSparse_csr_legacy_multivector_gpu_list[ i ]} " )                                                  # 8
            cuSparse_file.write( f"{cuSparse_ellpack_gpu_list[ i ]} {cuSparse_ellpack_legacy_gpu_list[ i ]} " )                         # 9 10
            cuSparse_file.write( f"{cuSparse_sliced_ellpack_gpu_list[ i ]} {cuSparse_sliced_ellpack_legacy_gpu_list[ i ]} " )           # 11 12
            cuSparse_file.write( f"{cuSparse_chunked_ellpack_gpu_list[ i ]} {cuSparse_chunked_ellpack_legacy_gpu_list[ i ]} " )          # 13 14
            cuSparse_file.write( f"{cuSparse_bi_ellpack_gpu_list[ i ]} {cuSparse_bi_ellpack_legacy_gpu_list[ i ]}\n" )                  # 15 16
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
# NOTE: """...""" allows multi-line strings, r"..." disables backslash-escaping (so a single \ is just a \ in the output)
gnuplot_file.write( r"""
set terminal postscript lw 3 20 color
set grid
set xlabel 'Matrix'
set xtics 250
set ylabel 'Bandwidth GB/sec'
#set output 'csr-legacy-adaptive-vs-cusparse.eps'
#plot 'cusparse.gplt' using 1:2 title '' with dots linewidth 2 lt rgb 'red',                                     \
#     'cusparse.gplt' using 1:2 title 'cuSparse' with lines linewidth 0.5 lt rgb 'red',                          \
#     'cusparse.gplt' using 1:3 title '' with dots linewidth 2 lt rgb 'green',                                   \
#     'cusparse.gplt' using 1:3 title 'CSR Legacy Adaptive' with lines linewidth 0.5 lt rgb 'green',                    
#set output 'csr-legacy-light-vs-cusparse.eps'
#plot 'cusparse.gplt' using 1:2 title '' with dots linewidth 2 lt rgb 'red',                                     \
#     'cusparse.gplt' using 1:2 title 'cuSparse' with lines linewidth 0.5 lt rgb 'red',                          \
#     'cusparse.gplt' using 1:4 title '' with dots linewidth 2 lt rgb 'green',                                   \
#     'cusparse.gplt' using 1:4 title 'CSR Legacy Light' with lines linewidth 0.5 lt rgb 'green',                    
#set output 'csr-legacy-light-without-atomic-vs-cusparse.eps'
#plot 'cusparse.gplt' using 1:2 title '' with dots linewidth 2 lt rgb 'red',                                     \
#     'cusparse.gplt' using 1:2 title 'cuSparse' with lines linewidth 0.5 lt rgb 'red',                          \
#     'cusparse.gplt' using 1:5 title '' with dots linewidth 2 lt rgb 'green',                                   \
#     'cusparse.gplt' using 1:5 title 'CSR Legacy LightWithoutAtomic' with lines linewidth 0.5 lt rgb 'green',                    
#set output 'csr-legacy-scalar-vs-cusparse.eps'
#plot 'cusparse.gplt' using 1:2 title '' with dots linewidth 2 lt rgb 'red',                                     \
#     'cusparse.gplt' using 1:2 title 'cuSparse' with lines linewidth 0.5 lt rgb 'red',                          \
#     'cusparse.gplt' using 1:6 title '' with dots linewidth 2 lt rgb 'green',                                   \
#     'cusparse.gplt' using 1:6 title 'CSR Legacy Scalar' with lines linewidth 0.5 lt rgb 'green',                    
#set output 'csr-legacy-vector-vs-cusparse.eps'
#plot 'cusparse.gplt' using 1:2 title '' with dots linewidth 2 lt rgb 'red',                                     \
#     'cusparse.gplt' using 1:2 title 'cuSparse' with lines linewidth 0.5 lt rgb 'red',                          \
#     'cusparse.gplt' using 1:7 title '' with dots linewidth 2 lt rgb 'green',                                   \
#     'cusparse.gplt' using 1:7 title 'CSR Legacy Vector' with lines linewidth 0.5 lt rgb 'green',                    
#set output 'csr-legacy-multivector-vs-cusparse.eps'
#plot 'cusparse.gplt' using 1:2 title '' with dots linewidth 2 lt rgb 'red',                                     \
#     'cusparse.gplt' using 1:2 title 'cuSparse' with lines linewidth 0.5 lt rgb 'red',                          \
#     'cusparse.gplt' using 1:8 title '' with dots linewidth 2 lt rgb 'green',                                   \
#     'cusparse.gplt' using 1:8 title 'CSR Legacy MultiVector' with lines linewidth 0.5 lt rgb 'green',                    
set output 'ellpack-vs-cusparse.eps'
plot 'cusparse.gplt' using 1:2 title '' with dots linewidth 2 lt rgb 'red',                                     \
     'cusparse.gplt' using 1:2 title 'cuSparse' with lines linewidth 0.5 lt rgb 'red',                          \
     'cusparse.gplt' using 1:9 title '' with dots linewidth 2 lt rgb 'green',                                   \
     'cusparse.gplt' using 1:9 title 'Ellpack' with lines linewidth 0.5 lt rgb 'green',                         \
     'cusparse.gplt' using 1:10 title '' with dots linewidth 2 lt rgb 'blue',                                   \
     'cusparse.gplt' using 1:10 title 'Ellpack Legacy' with lines linewidth 0.5 lt rgb 'blue'               
set output 'sliced-ellpack-vs-cusparse.eps'                                                             
plot 'cusparse.gplt' using 1:2 title '' with dots linewidth 2 lt rgb 'red',                                     \
     'cusparse.gplt' using 1:2 title 'cuSparse' with lines linewidth 0.5 lt rgb 'red',                          \
     'cusparse.gplt' using 1:11 title '' with dots linewidth 2 lt rgb 'green',                                  \
     'cusparse.gplt' using 1:11 title 'Sliced Ellpack' with lines linewidth 0.5 lt rgb 'green',                 \
     'cusparse.gplt' using 1:12 title '' with dots linewidth 2 lt rgb 'blue',                                   \
     'cusparse.gplt' using 1:12 title 'Sliced Ellpack Legacy' with lines linewidth 0.5 lt rgb 'blue'        
set output 'chunked-ellpack-vs-cusparse.eps'                                                            
plot 'cusparse.gplt' using 1:2 title '' with dots linewidth 2 lt rgb 'red',                                     \
     'cusparse.gplt' using 1:2 title 'cuSparse' with lines linewidth 0.5 lt rgb 'red',                          \
     'cusparse.gplt' using 1:13 title '' with dots linewidth 2 lt rgb 'green',                                  \
     'cusparse.gplt' using 1:13 title 'Chunked Ellpack' with lines linewidth 0.5 lt rgb 'green',                \
     'cusparse.gplt' using 1:14 title '' with dots linewidth 2 lt rgb 'blue',                                   \
     'cusparse.gplt' using 1:14 title 'Chunked Ellpack Legacy' with lines linewidth 0.5 lt rgb 'blue'       
set output 'bi-ellpack-vs-cusparse.eps'                                                                 
plot 'cusparse.gplt' using 1:2 title '' with dots linewidth 2 lt rgb 'red',                                     \
     'cusparse.gplt' using 1:2 title 'cuSparse' with lines linewidth 0.5 lt rgb 'red',                          \
     'cusparse.gplt' using 1:15 title '' with dots linewidth 2 lt rgb 'green',                                  \
     'cusparse.gplt' using 1:15 title 'BiEllpack' with lines linewidth 0.5 lt rgb 'green',                      \
     'cusparse.gplt' using 1:16 title '' with dots linewidth 2 lt rgb 'blue',                                   \
     'cusparse.gplt' using 1:16 title 'BiEllpack Legacy' with lines linewidth 0.5 lt rgb 'blue'             
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
#os.system( "epstopdf --autorotate All csr-legacy-adaptive-vs-cusparse.eps" )
#os.system( "epstopdf --autorotate All csr-legacy-light-vs-cusparse.eps" )
#os.system( "epstopdf --autorotate All csr-legacy-light-without-atomic-vs-cusparse.eps" )
#os.system( "epstopdf --autorotate All csr-legacy-scalar-vs-cusparse.eps" )
#os.system( "epstopdf --autorotate All csr-legacy-vector-vs-cusparse.eps" )
#os.system( "epstopdf --autorotate All csr-legacy-multivector-vs-cusparse.eps" )
os.system( "epstopdf --autorotate All ellpack-vs-cusparse.eps" )
os.system( "epstopdf --autorotate All sliced-ellpack-vs-cusparse.eps" )
os.system( "epstopdf --autorotate All chunked-ellpack-vs-cusparse.eps" )
os.system( "epstopdf --autorotate All bi-ellpack-vs-cusparse.eps" )
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
