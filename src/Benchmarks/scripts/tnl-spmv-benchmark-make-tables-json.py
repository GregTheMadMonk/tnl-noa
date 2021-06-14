#!/usr/bin/python3

import os
import json
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import numpy as np

####
# Helper function
def slugify(s):
   s = str(s).strip().replace(' ', '_')
   return re.sub(r'(?u)[^-\w.]', '', s)

####
# Extract all formats
def get_formats( input_df ):
   matrixName = input_df.iloc[0]['matrix name']
   df_matrix = input_df.loc[input_df['matrix name'] == matrixName]
   formats = df_matrix.loc[:,'format'].values.tolist() # Get format names - TODO: the first benchmark might not have all of them
   formats = list(dict.fromkeys(formats))              # remove duplicates
   return formats

####
# Create multiindex for columns
def get_multiindex( input_df, formats ):
   level1 = [ 'Matrix name', 'rows', 'columns' ]
   level2 = [ '',            '',     ''        ]
   level3 = [ '',            '',     ''        ]
   level4 = [ '',            '',     ''        ]
   df_data = [[ ' ',' ',' ']]
   for format in formats:
      for device in ['CPU','GPU']:
         for data in ['bandwidth' ]: #,'time','speed-up','non-zeros','stddev','stddev/time','diff.max','diff.l2']:
            level1.append( format )
            level2.append( device )
            level3.append( data )
            level4.append( '' )
            df_data[ 0 ].append( ' ' )
      if not format in [ 'cusparse', 'CSR' ]:
         for speedup in [ 'cusparse', 'CSR CPU']:
            level1.append( format )
            level2.append( 'GPU' )
            level3.append( 'speed-up')
            level4.append( speedup )
            df_data[ 0 ].append( ' ' )
   multiColumns = pd.MultiIndex.from_arrays([ level1, level2, level3, level4 ] )
   return multiColumns, df_data

####
# Convert input table to better structured one
def convert_data_frame( input_df, multicolumns, df_data, max_rows = -1 ):
   frames = []
   in_idx = 0
   out_idx = 0
   max_out_idx = max_rows
   if max_out_idx == -1:
      max_out_idx = len(input_df.index)
   while in_idx < len(input_df.index) and out_idx < max_out_idx:
      matrixName = input_df.iloc[in_idx]['matrix name']
      df_matrix = input_df.loc[input_df['matrix name'] == matrixName]
      print( out_idx, ":", in_idx, "/", len(input_df.index), ":", matrixName )
      aux_df = pd.DataFrame( df_data, columns = multicolumns, index = [out_idx] )
      for index,row in df_matrix.iterrows():
         aux_df.iloc[0]['Matrix name'] = row['matrix name']
         aux_df.iloc[0]['rows']        = row['rows']
         aux_df.iloc[0]['columns']     = row['columns']
         current_format = row['format']
         current_device = row['device']
         #print( current_format + " / " + current_device )
         aux_df.iloc[0][(current_format,current_device,'bandwidth','')]   = pd.to_numeric(row['bandwidth'], errors='coerce')
         #aux_df.iloc[0][(current_format,current_device,'time')]        = row['time']
         #aux_df.iloc[0][(current_format,current_device,'speed-up')]    = row['speedup']
         #aux_df.iloc[0][(current_format,current_device,'non-zeros')]   = row['non-zeros']
         #aux_df.iloc[0][(current_format,current_device,'stddev')]      = row['stddev']
         #aux_df.iloc[0][(current_format,current_device,'stddev/time')] = row['stddev/time']
         #aux_df.iloc[0][(current_format,current_device,'diff.max')]    = row['CSR Diff.Max']
         #aux_df.iloc[0][(current_format,current_device,'diff.l2')]    = row['CSR Diff.L2']
      frames.append( aux_df )
      out_idx = out_idx + 1
      in_idx = in_idx + len(df_matrix.index)
   result = pd.concat( frames )
   return result

####
# Compute speed-up of particular formats compared to Cusparse on GPU and CSR on CPU
def compute_speedup( df, formats ):
   for format in formats:
      if not format in [ 'cusparse', 'CSR' ]:
         print( 'Adding speed-up for ', format )
         format_bdw_list = df[(format,'GPU','bandwidth')]
         cusparse_bdw_list = df[('cusparse','GPU','bandwidth')]
         csr_bdw_list = df[('CSR','CPU','bandwidth')]
         cusparse_speedup_list = []
         csr_speedup_list = []
         for ( format_bdw, cusparse_bdw, csr_bdw ) in zip( format_bdw_list, cusparse_bdw_list,csr_bdw_list ):
            try:
               cusparse_speedup_list.append( format_bdw / cusparse_bdw )
            except:
               cusparse_speedup_list.append('')
            try:
               csr_speedup_list.append( format_bdw / csr_bdw )
            except:
               csr_speedup_list.append('')
            #print( f'**{type(format_bdw)}** -- {type(5.2)}' )
            #if type(format_bdw) == "<class 'numpy.float64'>":
            #   print( f'##########{format_bdw / cusparse_bdw}' )
            #   cusparse_speedup_list.append( format_bdw / cusparse_bdw )
            #   csr_speedup_list.append( format_bdw / csr_bdw )
            #else:
            #   cusparse_speedup_list.append('')
            #   csr_speedup_list.append('')
         df[(format,'GPU','speed-up','cusparse')] = cusparse_speedup_list
         df[(format,'GPU','speed-up','CSR CPU')] = csr_speedup_list

####
# Comparison with Cusparse
def cusparse_comparison( df, formats, head_size=10 ):
   if not os.path.exists("Cusparse-bw"):
      os.mkdir("Cusparse-bw")
   df.sort_values(by=[('cusparse','GPU','bandwidth')],inplace=True,ascending=False)
   for format in formats:
      if not format in ['cusparse','CSR']:
         print( f"Writing comparison of {format} and Cusparse" )
         filtered_df = df.dropna( subset=[(format,'GPU','bandwidth','')] )
         t = np.arange(filtered_df[(format,'GPU','bandwidth')].size )
         fig, axs = plt.subplots( 2, 1 )
         axs[0].plot( t, filtered_df[(format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[0].plot( t, filtered_df[('cusparse','GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[0].legend( [ format, 'Cusparse' ], loc='upper right' )
         axs[0].set_ylabel( 'Bandwidth in GB/sec' )
         axs[1].set_yscale( 'log' )
         axs[1].plot( t, filtered_df[(format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[1].plot( t, filtered_df[('cusparse','GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[1].legend( [ format, 'Cusparse' ], loc='upper right' )
         axs[1].set_xlabel( 'Matrix ID - sorted w.r.t. Cusparse' )
         axs[1].set_ylabel( 'Bandwidth in GB/sec' )
         plt.savefig( f"Cusparse-bw/{format}.pdf" )
         plt.close(fig)
         head_df = filtered_df.head( head_size )
         for f in formats:
            if not f in ['cusparse','CSR',format]:
               #print( f"Droping {f}..." )
               head_df.drop( labels=f, axis='columns', level=0, inplace=True )
         head_df.to_html( f"Cusparse-bw/{format}-head.html" )

####
# Comparison with CSR on CPU
def csr_comparison( df, formats, head_size=10 ):
   if not os.path.exists("CSR-bw"):
      os.mkdir("CSR-bw")
   for format in formats:
      if not format in ['cusparse','CSR']:
         print( f"Writing comparison of {format} and CSR on CPU" )
         result.sort_values(by=[(format,'GPU','bandwidth')],inplace=True,ascending=False)
         fig, axs = plt.subplots( 2, 1 )
         t = np.arange(result[(format,'GPU','bandwidth')].size )
         axs[0].plot( t, result[(format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[0].plot( t, result[('CSR','CPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[0].legend( [ format, 'CSR on CPU' ], loc='upper right' )
         axs[1].set_yscale( 'log' )
         axs[1].plot( t, result[(format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[1].plot( t, result[('CSR','CPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[1].legend( [ format, 'CSR on CPU' ], loc='upper right' )
         axs[1].set_xlabel( f"Matrix ID - sorted w.r.t. {format}" )
         axs[1].set_ylabel( 'Bandwidth in GB/sec' )
         plt.savefig( f"CSR-bw/{format}.pdf")
         plt.close(fig)
         head_df = df.head( head_size )
         for f in formats:
            if not f in ['cusparse','CSR',format]:
               #print( f"Droping {f}..." )
               head_df.drop( labels=f, axis='columns', level=0, inplace=True )
         head_df.to_html( f"CSR-bw/{format}-head.html" )

####
# Comparison of Legacy formats
def legacy_formats_comparison( df, formats, head_size=10 ):
   if not os.path.exists("Legacy-bw"):
      os.mkdir("Legacy-bw")
   for ref_format, legacy_format in [ ('Ellpack', 'Ellpack Legacy'),
                                    ('SlicedEllpack', 'SlicedEllpack Legacy'),
                                    ('ChunkedEllpack', 'ChunkedEllpack Legacy'),
                                    ('BiEllpack', 'BiEllpack Legacy'),
                                    ('CSR< Adaptive >', 'CSR Legacy Adaptive'),
                                    ('CSR< Scalar >', 'CSR Legacy Scalar'),
                                    ('CSR< Vector >', 'CSR Legacy Vector') ]:
      if ref_format in formats and legacy_format in formats:
         print( f"Writing comparison of {ref_format} and {legacy_format}" )
         df.sort_values(by=[(ref_format,'GPU','bandwidth')],inplace=True,ascending=False)
         fig, axs = plt.subplots( 2, 1 )
         t = np.arange(df[(ref_format,'GPU','bandwidth')].size )
         axs[0].plot( t, df[(ref_format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[0].plot( t, df[(legacy_format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[0].legend( [ ref_format, legacy_format ], loc='upper right' )
         axs[1].set_yscale( 'log' )
         axs[1].plot( t, df[(ref_format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[1].plot( t, df[(legacy_format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[1].legend( [ ref_format, legacy_format ], loc='upper right' )
         axs[1].set_xlabel( f"Matrix ID - sorted w.r.t. {ref_format}" )
         axs[1].set_ylabel( 'Bandwidth in GB/sec' )
         plt.savefig( f"Legacy-bw/{ref_format}.pdf")
         plt.close(fig)
         head_df = df.head( head_size )
         for f in formats:
            if not f in ['cusparse','CSR',format]:
               #print( f"Droping {f}..." )
               head_df.drop( labels=f, axis='columns', level=0, inplace=True )
         head_df.to_html( f"Legacy-bw/{format}-head.html" )

####
# Comparison of speed-up w.r.t. Cusparse
def cusparse_speedup_comparison( df, formats, head_size=10 ):
   if not os.path.exists("Cusparse-speed-up"):
      os.mkdir("Cusparse-speed-up")
   for format in formats:
      if not format in ['cusparse','CSR']:
         print( f"Writing comparison of speed-up of {format} compared to Cusparse" )
         df['tmp'] = df[(format, 'GPU','bandwidth')]
         filtered_df=df.dropna(subset=[('tmp','','','')])
         filtered_df.sort_values(by=[(format,'GPU','speed-up','cusparse')],inplace=True,ascending=False)
         fig, axs = plt.subplots( 2, 1 )
         size = len(filtered_df[(format,'GPU','speed-up','cusparse')].index)
         t = np.arange( size )
         bar = np.full( size, 1 )
         axs[0].plot( t, filtered_df[(format,'GPU','speed-up','cusparse')], '-o', ms=1, lw=1 )
         axs[0].plot( t, bar, '-', ms=1, lw=1 )
         axs[0].legend( [ format, 'Cusparse' ], loc='upper right' )
         axs[1].set_yscale( 'log' )
         axs[1].plot( t, filtered_df[(format,'GPU','speed-up','cusparse')], '-o', ms=1, lw=1 )
         axs[1].plot( t, bar, '-', ms=1, lw=1 )
         axs[1].legend( [ format, 'Cusparse' ], loc='upper right' )
         axs[1].set_xlabel( f"Matrix ID - sorted w.r.t. {format}" )
         axs[1].set_ylabel( 'Bandwidth in GB/sec' )
         plt.savefig( f"Cusparse-speed-up/{format}.pdf")
         plt.close(fig)
         head_df = filtered_df.head( head_size )
         for f in formats:
            if not f in ['cusparse','CSR',format]:
               #print( f"Droping {f}..." )
               head_df.drop( labels=f, axis='columns', level=0, inplace=True )
         head_df.to_html( f"Cusparse-speed-up/{format}-head.html" )

####
# Parse input file
print( "Parsing input file...." )
with open('sparse-matrix-benchmark.log') as f:
    d = json.load(f)
input_df = json_normalize( d, record_path=['results'] )
#input_df.to_html( "orig-pandas.html" )

formats = get_formats( input_df )
multicolumns, df_data = get_multiindex( input_df, formats )

print( "Converting data..." )
result = convert_data_frame( input_df, multicolumns, df_data, 20000 )
compute_speedup( result, formats )

print( "Writting to HTML file..." )
result.to_html( 'output.html' )

result.replace( to_replace=' ',value=np.nan,inplace=True)

####
# Generate report = tables and figures
head_size = 10
cusparse_comparison( result, formats, head_size )
csr_comparison( result, formats, head_size )
legacy_formats_comparison( result, formats, head_size )
cusparse_speedup_comparison( result, formats, head_size )

