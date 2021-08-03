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

def latexFormatName( name ):
   name = name.replace('<','')
   name = name.replace('>','')
   return name

####
# Extract all formats
def get_formats( input_df ):
   matrixName = input_df.iloc[0]['matrix name']
   df_matrix = input_df.loc[input_df['matrix name'] == matrixName]
   formats = df_matrix.loc[:,'format'].values.tolist() # Get format names - TODO: the first benchmark might not have all of them
   formats = list(dict.fromkeys(formats))              # remove duplicates
   formats.append('TNL Best')
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
         if format == 'CSR< Light >':
            level1.append( format )
            level2.append( 'GPU' )
            level3.append( 'speed-up')
            level4.append( 'LightSpMV Vector' )
            df_data[ 0 ].append( ' ' )
         if format == 'TNL Best':
            level1.append( format )
            level2.append( 'GPU' )
            level3.append( 'format')
            level4.append( '' )
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
      best_bw = 0
      for index,row in df_matrix.iterrows():
         aux_df.iloc[0]['Matrix name'] = row['matrix name']
         aux_df.iloc[0]['rows']        = row['rows']
         aux_df.iloc[0]['columns']     = row['columns']
         current_format = row['format']
         current_device = row['device']
         #print( current_format + " / " + current_device )
         bw = pd.to_numeric(row['bandwidth'], errors='coerce')
         aux_df.iloc[0][(current_format,current_device,'bandwidth','')] = bw
         if( current_device == 'GPU' and
             not 'Binary' in current_format and
             not 'Symmetric' in current_format and
             not 'Legacy' in current_format and
             not 'cusparse' in current_format and
             not 'LightSpMV' in current_format and
             bw > best_bw ):
            best_bw = bw
            best_format = current_format
         if current_format == 'cusparse':
            cusparse_bw = bw
         #aux_df.iloc[0][(current_format,current_device,'time')]        = row['time']
         #aux_df.iloc[0][(current_format,current_device,'speed-up')]    = row['speedup']
         #aux_df.iloc[0][(current_format,current_device,'non-zeros')]   = row['non-zeros']
         #aux_df.iloc[0][(current_format,current_device,'stddev')]      = row['stddev']
         #aux_df.iloc[0][(current_format,current_device,'stddev/time')] = row['stddev/time']
         #aux_df.iloc[0][(current_format,current_device,'diff.max')]    = row['CSR Diff.Max']
         #aux_df.iloc[0][(current_format,current_device,'diff.l2')]    = row['CSR Diff.L2']
      aux_df.iloc[0][('TNL Best','GPU','bandwidth','')] = best_bw
      if best_bw > cusparse_bw:
         aux_df.iloc[0][('TNL Best','GPU','format','')] = best_format
      else:
         aux_df.iloc[0][('TNL Best','GPU','format','')] = 'cusparse'
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

   csr_light_bdw_list = df[('CSR< Light >','GPU','bandwidth')]
   light_spmv_bdw_list = df[('LightSpMV Vector','GPU','bandwidth')]

   csr_light_speedup_list = []
   for ( csr_light_bdw, light_spmv_bdw ) in zip(csr_light_bdw_list,light_spmv_bdw_list):
      try:
         csr_light_speedup_list.append( csr_light_bdw / light_spmv_bdw  )
      except:
         csr_light_speedup_list.append('')
   df[('CSR< Light >','GPU','speed-up','LightSpMV Vector')] = csr_light_speedup_list


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
         axs[1].legend( [ latexFormatName(format), 'Cusparse' ], loc='lower left' )
         axs[1].set_xlabel( 'Matrix ID - sorted w.r.t. Cusparse performance' )
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
         axs[0].legend( [ latexFormatName(format), 'CSR on CPU' ], loc='upper right' )
         axs[0].set_ylabel( 'Bandwidth in GB/sec' )
         axs[1].set_yscale( 'log' )
         axs[1].plot( t, result[(format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[1].plot( t, result[('CSR','CPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[1].legend( [ latexFormatName(format), 'CSR on CPU' ], loc='lower left' )
         axs[1].set_xlabel( f"Matrix ID - sorted w.r.t. {latexFormatName(format)} performance" )
         axs[1].set_ylabel( 'Bandwidth in GB/sec' )
         plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})
         # for Palatino and other serif fonts use:
         #plt.rcParams.update({
         #   "text.usetex": True,
         #   "font.family": "serif",
         #   "font.serif": ["Palatino"],
         #})
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
         axs[0].legend( [ latexFormatName(ref_format), latexFormatName(legacy_format) ], loc='upper right' )
         axs[0].set_ylabel( 'Bandwidth in GB/sec' )
         axs[1].set_yscale( 'log' )
         axs[1].plot( t, df[(ref_format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[1].plot( t, df[(legacy_format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[1].legend( [ latexFormatName(ref_format), latexFormatName(legacy_format) ], loc='lower left' )
         axs[1].set_xlabel( f"Matrix ID - sorted w.r.t. {latexFormatName(ref_format)}  performance" )
         axs[1].set_ylabel( 'Bandwidth in GB/sec' )
         plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})
         # for Palatino and other serif fonts use:
         #plt.rcParams.update({
         #   "text.usetex": True,
         #   "font.family": "serif",
         #   "font.serif": ["Palatino"],
         #})
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
         fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
         size = len(filtered_df[(format,'GPU','speed-up','cusparse')].index)
         t = np.arange( size )
         bar = np.full( size, 1 )
         axs.plot( t, filtered_df[(format,'GPU','speed-up','cusparse')], '-o', ms=1, lw=1 )
         axs.plot( t, bar, '-', ms=1, lw=1 )
         axs.legend( [ latexFormatName(format), 'Cusparse' ], loc='upper right' )
         axs.set_ylabel( 'Speedup' )
         axs.set_xlabel( f"Matrix ID - sorted w.r.t. {latexFormatName(format)} speed-up" )
         plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})
         # for Palatino and other serif fonts use:
         #plt.rcParams.update({
         #   "text.usetex": True,
         #   "font.family": "serif",
         #   "font.serif": ["Palatino"],
         #})
         plt.savefig( f"Cusparse-speed-up/{format}.pdf")
         plt.close(fig)

         fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
         axs.set_yscale( 'log' )
         axs.plot( t, filtered_df[(format,'GPU','speed-up','cusparse')], '-o', ms=1, lw=1 )
         axs.plot( t, bar, '-', ms=1, lw=1 )
         axs.legend( [ latexFormatName(format), 'Cusparse' ], loc='lower left' )
         axs.set_xlabel( f"Matrix ID - sorted w.r.t. {latexFormatName(format)} speed-up" )
         axs.set_ylabel( 'Speedup' )
         plt.savefig( f"Cusparse-speed-up/{format}-log.pdf")
         plt.close(fig)
         head_df = filtered_df.head( head_size )
         for f in formats:
            if not f in ['cusparse','CSR',format]:
               #print( f"Droping {f}..." )
               head_df.drop( labels=f, axis='columns', level=0, inplace=True )
         head_df.to_html( f"Cusparse-speed-up/{format}-head.html" )

####
# Comparison of speed-up w.r.t. LightSpMV
def csr_light_speedup_comparison( df, head_size=10 ):
   format = 'CSR< Light >'
   print( f"Writing comparison of speed-up of CSR Light compared to LightSPMV" )
   df['tmp'] = df[(format, 'GPU','bandwidth')]
   filtered_df=df.dropna(subset=[('tmp','','','')])
   filtered_df.sort_values(by=[(format,'GPU','speed-up','LightSpMV Vector')],inplace=True,ascending=False)
   fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
   size = len(filtered_df[(format,'GPU','speed-up','LightSpMV Vector')].index)
   t = np.arange( size )
   bar = np.full( size, 1 )
   axs.plot( t, filtered_df[(format,'GPU','speed-up','LightSpMV Vector')], '-o', ms=1, lw=1 )
   axs.plot( t, bar, '-', ms=1, lw=1 )
   axs.legend( [ latexFormatName(format), 'LightSpMV' ], loc='upper right' )
   axs.set_ylabel( 'Speedup' )
   axs.set_xlabel( f"Matrix ID - sorted w.r.t. {latexFormatName(format)} speed-up" )
   plt.rcParams.update({
      "text.usetex": True,
      "font.family": "sans-serif",
      "font.sans-serif": ["Helvetica"]})
   # for Palatino and other serif fonts use:
   #plt.rcParams.update({
   #   "text.usetex": True,
   #   "font.family": "serif",
   #   "font.serif": ["Palatino"],
   #})
   plt.savefig( f"LightSpMV-speed-up.pdf")
   plt.close(fig)

   fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
   axs.set_yscale( 'log' )
   axs.plot( t, filtered_df[(format,'GPU','speed-up','LightSpMV Vector')], '-o', ms=1, lw=1 )
   axs.plot( t, bar, '-', ms=1, lw=1 )
   axs.legend( [ latexFormatName(format), 'LightSpMV' ], loc='lower left' )
   axs.set_xlabel( f"Matrix ID - sorted w.r.t. {latexFormatName(format)} speed-up" )
   axs.set_ylabel( 'Speedup' )
   plt.savefig( f"LightSpMV-speed-up-log.pdf")
   plt.close(fig)
   head_df = filtered_df.head( head_size )
   for f in formats:
      if not f in ['cusparse','CSR',format]:
         #print( f"Droping {f}..." )
         head_df.drop( labels=f, axis='columns', level=0, inplace=True )
   head_df.to_html( f"LightSpMV-speed-up-head.html" )


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
result = convert_data_frame( input_df, multicolumns, df_data, 20 )
compute_speedup( result, formats )

result.replace( to_replace=' ',value=np.nan,inplace=True)

####
# Make data analysis
def processDf( df, formats, head_size = 10 ):
   print( "Writting to HTML file..." )
   df.to_html( f'output.html' )

   # Generate tables and figures
   cusparse_comparison( df, formats, head_size )
   csr_comparison( df, formats, head_size )
   legacy_formats_comparison( df, formats, head_size )
   cusparse_speedup_comparison( df, formats, head_size )
   csr_light_speedup_comparison( df, head_size )

   best = df[('TNL Best','GPU','format')].tolist()
   for format in formats:
      if( not 'Binary' in format and
          not 'Symmetric' in format and
          not 'Legacy' in format and
          not 'LightSpMV' in format and
          not 'TNL Best' in format ):
         cases = best.count(format)
         print( f'{format} is best in {cases} cases.')

head_size = 10
if not os.path.exists( 'general' ):
   os.mkdir( 'general' )
os.chdir( 'general' )
processDf( result, formats, head_size )
os.chdir( '..' )

#for rows_count in [ 10, 100, 1000, 10000, 100000, 1000000, 10000000 ]:
#   filtered_df = result[ result['rows'].astype('int32') <= rows_count ]
#   if not os.path.exists(f'rows-le-{rows_count}'):
#      os.mkdir( f'rows-le-{rows_count}')
#   os.chdir( f'rows-le-{rows_count}')
#   processDf( filtered_df, formats, head_size )
#   os.chdir( '..' )

#for rows_count in [ 10, 100, 1000, 10000, 100000, 1000000, 10000000 ]:
#   filtered_df = result[ result['rows'].astype('int32') >= rows_count ]
#   if not os.path.exists(f'rows-ge-{rows_count}'):
#      os.mkdir( f'rows-ge-{rows_count}')
#   os.chdir( f'rows-ge-{rows_count}')
#   processDf( filtered_df, formats, head_size )
#   os.chdir( '..' )
