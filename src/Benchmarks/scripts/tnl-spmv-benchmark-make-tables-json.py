#!/usr/bin/python3

import os
import json
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import numpy as np

#Latex fonst set-up

#plt.rcParams.update({
#   "text.usetex": True,
#   "font.family": "sans-serif",
#   "font.sans-serif": ["Helvetica"]})
#
# for Palatino and other serif fonts use:
#plt.rcParams.update({
#   "text.usetex": True,
#   "font.family": "serif",
#   "font.serif": ["Palatino"],
#})


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
      if 'Binary' in format:
         level1.append( format )
         level2.append( 'GPU' )
         level3.append( 'speed-up')
         level4.append( 'non-binary' )
         df_data[ 0 ].append( ' ' )
      if 'Symmetric' in format:
         level1.append( format )
         level2.append( 'GPU' )
         level3.append( 'speed-up')
         level4.append( 'non-symmetric' )
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
def convert_data_frame( input_df, multicolumns, df_data, begin_idx = 0, end_idx = -1 ):
   frames = []
   in_idx = 0
   out_idx = 0
   #max_out_idx = max_rows
   if end_idx == -1:
      end_idx = len(input_df.index)
   while in_idx < len(input_df.index) and out_idx < end_idx:
      matrixName = input_df.iloc[in_idx]['matrix name']
      df_matrix = input_df.loc[input_df['matrix name'] == matrixName]
      if out_idx >= begin_idx:
         print( f'{out_idx} : {in_idx} / {len(input_df.index)} : {matrixName}' )
      else:
         print( f'{out_idx} : {in_idx} / {len(input_df.index)} : {matrixName} - SKIP' )
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
             not 'Hybrid' in current_format and
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
      if out_idx >= begin_idx:
         frames.append( aux_df )
      out_idx = out_idx + 1
      in_idx = in_idx + len(df_matrix.index)
   result = pd.concat( frames )
   return result

####
# Compute speed-up of particular formats compared to Cusparse on GPU and CSR on CPU
def compute_cusparse_speedup( df, formats ):
   for device in [ 'CPU', 'GPU' ]:
      for format in formats:
         if not format in [ 'cusparse', 'CSR' ]:
            print( 'Adding speed-up for ', format )
            try:
               format_bdw_list = df[(format,device,'bandwidth')]
            except:
               continue
            cusparse_bdw_list = df[('cusparse','GPU','bandwidth')]
            csr_bdw_list = df[('CSR','CPU','bandwidth')]
            cusparse_speedup_list = []
            csr_speedup_list = []
            for( format_bdw, cusparse_bdw, csr_bdw ) in zip( format_bdw_list, cusparse_bdw_list,csr_bdw_list ):
               if( device == 'GPU' ):
                  try:
                     cusparse_speedup_list.append( format_bdw / cusparse_bdw )
                  except:
                     cusparse_speedup_list.append(float('nan'))
               try:
                  csr_speedup_list.append( format_bdw / csr_bdw )
               except:
                  csr_speedup_list.append(float('nan'))
            if( device == 'GPU' ):
               df[(format,'GPU','speed-up','cusparse')] = cusparse_speedup_list
            df[(format,device,'speed-up','CSR CPU')] = csr_speedup_list

####
# Compute speedup of Light CSR
def compute_csr_light_speedup( df ):
   csr_light_bdw_list = df[('CSR< Light >','GPU','bandwidth')]
   light_spmv_bdw_list = df[('LightSpMV Vector','GPU','bandwidth')]

   csr_light_speedup_list = []
   for ( csr_light_bdw, light_spmv_bdw ) in zip(csr_light_bdw_list,light_spmv_bdw_list):
      try:
         csr_light_speedup_list.append( csr_light_bdw / light_spmv_bdw  )
      except:
         csr_light_speedup_list.append(float('nan'))
   df[('CSR< Light >','GPU','speed-up','LightSpMV Vector')] = csr_light_speedup_list

####
# Compute speed-up of binary formats
def compute_binary_speedup( df, formats ):
   for format in formats:
      if 'Binary' in format:
         non_binary_format = format.replace( 'Binary ', '' )
         print( f'Adding speed-up of {format} vs {non_binary_format}' )
         format_bdw_list = df[(format,'GPU','bandwidth')]
         non_binary_bdw_list = df[(non_binary_format,'GPU','bandwidth')]
         binary_speedup_list = []
         for ( format_bdw, non_binary_bdw ) in zip( format_bdw_list, non_binary_bdw_list ):
            try:
               binary_speedup_list.append( format_bdw / non_binary_bdw )
            except:
               binary_speedup_list.append( float('nan'))
         df[(format,'GPU','speed-up','non-binary')] = binary_speedup_list

####
# Compute speed-up of symmetric formats
def compute_symmetric_speedup( df, formats ):
   for format in formats:
      if 'Symmetric' in format:
         non_symmetric_format = format.replace( 'Symmetric ', '' )
         print( f'Adding speed-up of {format} vs {non_symmetric_format}' )
         format_bdw_list = df[(format,'GPU','bandwidth')]
         non_symmetric_bdw_list = df[(non_symmetric_format,'GPU','bandwidth')]
         symmetric_speedup_list = []
         for ( format_bdw, non_symmetric_bdw ) in zip( format_bdw_list, non_symmetric_bdw_list ):
            try:
               symmetric_speedup_list.append( format_bdw / non_symmetric_bdw )
            except:
               symmetric_speedup_list.append(float('nan'))
         df[(format,'GPU','speed-up','non-symmetric')] = symmetric_speedup_list

def compute_speedup( df, formats ):
   compute_cusparse_speedup( df, formats )
   compute_csr_light_speedup( df )
   compute_binary_speedup( df, formats )
   compute_symmetric_speedup( df, formats )

###
# Draw several profiles into one figure
def draw_profiles( formats, profiles, xlabel, ylabel, filename, style=[] ):
   fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
   latexNames = []
   size = 1
   for format in formats:
      t = np.arange(profiles[format].size )
      axs.plot( t, profiles[format], '-o', ms=1, lw=1 )
      size = len( profiles[format] )
      latexNames.append( latexFormatName( format ) )
   if 'draw-bar' in style:
      #print( f'size = {size}' )
      bar = np.full( size, 1 )
      axs.plot( t, bar, '-', ms=1, lw=1.5 )

   axs.legend( latexNames, loc='upper right' )
   axs.set_xlabel( xlabel )
   axs.set_ylabel( ylabel )
   axs.set_yscale( 'log' )
   plt.rcParams.update({
      "text.usetex": True,
      "font.family": "sans-serif",
      "font.sans-serif": ["Helvetica"]})
   plt.savefig( filename )
   plt.close(fig)


####
# Effective BW profile
def effective_bw_profile( df, formats, head_size=10 ):
   if not os.path.exists("BW-profile"):
      os.mkdir("BW-profile")
   profiles = {}
   for format in formats:
      print( f"Writing BW profile of {format}" )
      fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
      t = np.arange(df[(format,'GPU','bandwidth')].size )
      if format == 'CSR':
         df.sort_values(by=[(format,'CPU','bandwidth')],inplace=True,ascending=False)
         profiles[format] = df[(format,'CPU','bandwidth')].copy()
         axs.plot( t, df[(format,'CPU','bandwidth')], '-o', ms=1, lw=1 )
      else:
         df.sort_values(by=[(format,'GPU','bandwidth')],inplace=True,ascending=False)
         profiles[format] = df[(format,'GPU','bandwidth')].copy()
         axs.plot( t, df[(format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
      axs.legend( [ latexFormatName(format), 'CSR on CPU' ], loc='upper right' )
      axs.set_ylabel( 'Bandwidth in GB/sec' )
      plt.rcParams.update({
         "text.usetex": True,
         "font.family": "sans-serif",
         "font.sans-serif": ["Helvetica"]})
      plt.savefig( f"BW-profile/{format}.pdf")
      plt.close(fig)
      fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
      axs.set_yscale( 'log' )
      axs.plot( t, result[(format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
      axs.legend( [ latexFormatName(format), 'CSR on CPU' ], loc='lower left' )
      axs.set_xlabel( f"Matrix ID - sorted w.r.t. {latexFormatName(format)} performance" )
      axs.set_ylabel( 'Bandwidth in GB/sec' )
      plt.rcParams.update({
         "text.usetex": True,
         "font.family": "sans-serif",
         "font.sans-serif": ["Helvetica"]})
      plt.savefig( f"BW-profile/{format}-log.pdf")
      plt.close(fig)
      copy_df = df.copy()
      for f in formats:
         if not f in ['cusparse','CSR',format]:
            copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
      copy_df.to_html( f"BW-profile/{format}.html" )

   # Draw ellpack formats profiles
   current_formats = []
   xlabel = "Matrix ID - sorted by particular formats effective BW"
   ylabel = "Bandwidth in GB/sec"
   for format in formats:
      if( ( 'Ellpack' in format and not 'Binary' in format and not 'Legacy' in format ) or
          format == 'CSR' or
          format == 'cusparse' ):
         current_formats.append( format )
   draw_profiles( current_formats, profiles, xlabel, ylabel, "ellpack-profiles-bw.pdf" )

   # Draw CSR formats profiles
   current_formats.clear()
   for format in formats:
      if( ( 'CSR' in format and not 'Binary' in format and not 'Legacy' in format and not 'Hybrid' in format ) or
          format == 'cusparse' ):
         current_formats.append( format )
   draw_profiles( current_formats, profiles, xlabel, ylabel, "csr-profiles-bw.pdf" )


####
# Comparison with Cusparse
def cusparse_comparison( df, formats, head_size=10 ):
   if not os.path.exists("Cusparse-bw"):
      os.mkdir("Cusparse-bw")
   ascend_df = df.copy()
   df.sort_values(by=[('cusparse','GPU','bandwidth')],inplace=True,ascending=False)
   ascend_df.sort_values(by=[('cusparse','GPU','bandwidth')],inplace=True,ascending=True)
   for format in formats:
      if not format in ['cusparse','CSR']:
         print( f"Writing comparison of {format} and Cusparse" )
         filtered_df = df.dropna( subset=[(format,'GPU','bandwidth','')] )
         filtered_ascend_df = ascend_df.dropna( subset=[(format,'GPU','bandwidth','')] )
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
         copy_df = df.copy()
         for f in formats:
            if not f in ['cusparse','CSR',format]:
               copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
         copy_df.to_html( f"Cusparse-bw/{format}.html" )

####
# Comparison with CSR on CPU
def csr_comparison( df, formats, head_size=10 ):
   if not os.path.exists("CSR-bw"):
      os.mkdir("CSR-bw")
   for device in [ 'CPU', 'GPU' ]:
      for format in formats:
         if not format in ['cusparse','CSR']:
            print( f"Writing comparison of {format} and CSR on CPU" )
            try:
               df.sort_values(by=[(format,device,'bandwidth')],inplace=True,ascending=False)
            except:
               continue
            fig, axs = plt.subplots( 2, 1 )
            t = np.arange(df[(format,device,'bandwidth')].size )
            axs[0].plot( t, df[(format,device,'bandwidth')], '-o', ms=1, lw=1 )
            axs[0].plot( t, df[('CSR','CPU','bandwidth')], '-o', ms=1, lw=1 )
            axs[0].legend( [ latexFormatName(format), 'CSR on CPU' ], loc='upper right' )
            axs[0].set_ylabel( 'Bandwidth in GB/sec' )
            axs[1].set_yscale( 'log' )
            axs[1].plot( t, result[(format,device,'bandwidth')], '-o', ms=1, lw=1 )
            axs[1].plot( t, result[('CSR','CPU','bandwidth')], '-o', ms=1, lw=1 )
            axs[1].legend( [ latexFormatName(format), 'CSR on CPU' ], loc='lower left' )
            axs[1].set_xlabel( f"Matrix ID - sorted w.r.t. {latexFormatName(format)} performance" )
            axs[1].set_ylabel( 'Bandwidth in GB/sec' )
            plt.rcParams.update({
               "text.usetex": True,
               "font.family": "sans-serif",
               "font.sans-serif": ["Helvetica"]})
            plt.savefig( f"CSR-bw/{format}-{device}.pdf")
            plt.close(fig)
            copy_df = df.copy()
            for f in formats:
               if not f in ['cusparse','CSR',format]:
                  copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
            copy_df.to_html( f"CSR-bw/{format}-{device}.html" )

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
         ascend_df = df.copy()
         df.sort_values(by=[(ref_format,'GPU','bandwidth')],inplace=True,ascending=False)
         ascend_df.sort_values(by=[(ref_format,'GPU','bandwidth')],inplace=True,ascending=True)
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
         plt.savefig( f"Legacy-bw/{ref_format}.pdf")
         plt.close(fig)
         copy_df = df.copy()
         for f in formats:
            if not f in ['cusparse','CSR',format]:
               copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
         copy_df.to_html( f"Legacy-bw/{format}.html" )

####
# Comparison of speed-up w.r.t. CSR
def csr_speedup_comparison( df, formats, head_size=10 ):
   if not os.path.exists("CSR-speed-up"):
      os.mkdir("CSR-speed-up")
   for device in ['CPU', 'GPU']:
      profiles = {}
      for format in formats:
         if not format in ['cusparse','CSR']:
            print( f"Writing comparison of speed-up of {format} compared to CSR" )
            df['tmp'] = df[(format, device,'bandwidth')]
            filtered_df=df.dropna(subset=[('tmp','','','')])
            try:
               filtered_df.sort_values(by=[(format,device,'speed-up','CSR CPU')],inplace=True,ascending=False)
            except:
               continue
            profiles[format] = filtered_df[(format,device,'speed-up','CSR CPU')].copy()
            fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
            size = len(filtered_df[(format,device,'speed-up','CSR CPU')].index)
            t = np.arange( size )
            bar = np.full( size, 1 )
            axs.plot( t, filtered_df[(format,device,'speed-up','CSR CPU')], '-o', ms=1, lw=1 )
            axs.plot( t, bar, '-', ms=1, lw=1 )
            axs.legend( [ latexFormatName(format), 'CSR CPU' ], loc='upper right' )
            axs.set_ylabel( 'Speedup' )
            axs.set_xlabel( f"Matrix ID - sorted w.r.t. {latexFormatName(format)} speed-up" )
            plt.rcParams.update({
               "text.usetex": True,
               "font.family": "sans-serif",
               "font.sans-serif": ["Helvetica"]})
            plt.savefig( f"CSR-speed-up/{format}.pdf")
            plt.close(fig)

            fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
            axs.set_yscale( 'log' )
            axs.plot( t, filtered_df[(format,device,'speed-up','CSR CPU')], '-o', ms=1, lw=1 )
            axs.plot( t, bar, '-', ms=1, lw=1 )
            axs.legend( [ latexFormatName(format), 'CSR' ], loc='lower left' )
            axs.set_xlabel( f"Matrix ID - sorted w.r.t. {latexFormatName(format)} speed-up" )
            axs.set_ylabel( 'Speedup' )
            plt.savefig( f"CSR-speed-up/{format}-{device}-log.pdf")
            plt.close(fig)
            copy_df = df.copy()
            for f in formats:
               if not f in ['cusparse','CSR',format]:
                  copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
            copy_df.to_html( f"CSR-speed-up/{format}-{device}.html" )


####
# Comparison of speed-up w.r.t. Cusparse
def cusparse_speedup_comparison( df, formats, head_size=10 ):
   if not os.path.exists("Cusparse-speed-up"):
      os.mkdir("Cusparse-speed-up")
   profiles = {}
   for format in formats:
      if not format in ['cusparse','CSR']:
         print( f"Writing comparison of speed-up of {format} compared to Cusparse" )
         df['tmp'] = df[(format, 'GPU','bandwidth')]
         filtered_df=df.dropna(subset=[('tmp','','','')])
         filtered_df.sort_values(by=[(format,'GPU','speed-up','cusparse')],inplace=True,ascending=False)
         profiles[format] = filtered_df[(format,'GPU','speed-up','cusparse')].copy()
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
         copy_df = df.copy()
         for f in formats:
            if not f in ['cusparse','CSR',format]:
               copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
         copy_df.to_html( f"Cusparse-speed-up/{format}.html" )

   # Draw Ellpack formats profiles
   xlabel = "Matrix ID - sorted particular by formats speedup compared to Cusparse"
   ylabel = "Speedup"
   current_formats = []
   for format in formats:
      if( 'Ellpack' in format and not 'Binary' in format and not 'Legacy' in format ):
         current_formats.append( format )
   draw_profiles( current_formats, profiles, xlabel, ylabel, "ellpack-profiles-cusparse-speedup.pdf", "draw-bar" )

   # Draw CSR formats profiles
   current_formats.clear()
   for format in formats:
      if( 'CSR' in format and not 'Binary' in format and not 'Legacy' in format and not 'Hybrid' in format and format != 'CSR' ):
         current_formats.append( format )
   draw_profiles( current_formats, profiles, xlabel, ylabel, "csr-profiles-cusparse-speedup.pdf", "draw-bar" )

####
# Comparison of binary matrices
def binary_matrices_comparison( df, formats, head_size = 10 ):
   if not os.path.exists("Binary-speed-up"):
      os.mkdir("Binary-speed-up")
   for format in formats:
      if 'Binary' in format:
         non_binary_format = format.replace('Binary ','')
         print( f"Writing comparison of speed-up of {format} vs {non_binary_format}" )
         #df['tmp'] = df[(format, 'GPU','speed-up','non-binary')]
         filtered_df=df.dropna(subset=[(format, 'GPU','speed-up','non-binary')]) #('tmp','','','')])
         #print( f"{format} -> {filtered_df[(format,'GPU','speed-up','non-binary')]}" )
         ascend_df = filtered_df.copy()
         filtered_df.sort_values(by=[(format,'GPU','speed-up','non-binary')],inplace=True,ascending=False)
         ascend_df.sort_values(by=[(format,'GPU','speed-up','non-binary')],inplace=True,ascending=True)
         fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
         size = len(filtered_df[(format,'GPU','speed-up','non-binary')].index)
         t = np.arange( size )
         bar = np.full( size, 1 )
         axs.plot( t, filtered_df[(format,'GPU','speed-up','non-binary')], '-o', ms=1, lw=1 )
         axs.plot( t, bar, '-', ms=1, lw=1 )
         axs.legend( [ latexFormatName(format), latexFormatName(non_binary_format) ], loc='upper right' )
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
         plt.savefig( f"Binary-speed-up/{format}.pdf")
         plt.close(fig)

         fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
         axs.set_yscale( 'log' )
         axs.plot( t, filtered_df[(format,'GPU','speed-up','non-binary')], '-o', ms=1, lw=1 )
         axs.plot( t, bar, '-', ms=1, lw=1 )
         axs.legend( [ latexFormatName(format), latexFormatName(non_binary_format) ], loc='upper right' )
         axs.set_xlabel( f"Matrix ID - sorted w.r.t. {latexFormatName(format)} speed-up" )
         axs.set_ylabel( 'Speedup' )
         plt.savefig( f"Binary-speed-up/{format}-log.pdf")
         plt.close(fig)
         #head_df = filtered_df.head( head_size )
         #bottom_df = ascend_df.head( head_size )
         copy_df = df.copy()
         for f in formats:
            if not f in ['cusparse','CSR',format,non_binary_format]:
               #print( f"Droping {f}..." )
               #head_df.drop( labels=f, axis='columns', level=0, inplace=True )
               copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
         #head_df.to_html( f"Binary-speed-up/{format}-head.html" )
         copy_df.to_html( f"Binary-speed-up/{format}.html" )

####
# Comparison of symmetric matrices
def symmetric_matrices_comparison( df, formats, head_size = 10 ):
   if not os.path.exists("Symmetric-speed-up"):
      os.mkdir("Symmetric-speed-up")
   for format in formats:
      if 'Symmetric' in format:
         non_symmetric_format = format.replace('Symmetric ','')
         print( f"Writing comparison of speed-up of {format} vs {non_symmetric_format}" )
         #df['tmp'] = df[(format, 'GPU','speed-up','non-symmetric')]
         filtered_df=df.dropna(subset=[(format, 'GPU','speed-up','non-symmetric')]) #('tmp','','','')])
         ascend_df = filtered_df.copy()
         #print( f"{format} -> {filtered_df[(format,'GPU','speed-up','non-symmetric')]}" )
         filtered_df.sort_values(by=[(format,'GPU','speed-up','non-symmetric')],inplace=True,ascending=False)
         ascend_df.sort_values(by=[(format,'GPU','speed-up','non-symmetric')],inplace=True,ascending=True)
         fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
         size = len(filtered_df[(format,'GPU','speed-up','non-symmetric')].index)
         t = np.arange( size )
         bar = np.full( size, 1 )
         axs.plot( t, filtered_df[(format,'GPU','speed-up','non-symmetric')], '-o', ms=1, lw=1 )
         axs.plot( t, bar, '-', ms=1, lw=1 )
         axs.legend( [ latexFormatName(format), latexFormatName(non_symmetric_format) ], loc='upper right' )
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
         plt.savefig( f"Symmetric-speed-up/{format}.pdf")
         plt.close(fig)

         fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
         axs.set_yscale( 'log' )
         axs.plot( t, filtered_df[(format,'GPU','speed-up','non-symmetric')], '-o', ms=1, lw=1 )
         axs.plot( t, bar, '-', ms=1, lw=1 )
         axs.legend( [ latexFormatName(format), latexFormatName(non_symmetric_format) ], loc='lower left' )
         axs.set_xlabel( f"Matrix ID - sorted w.r.t. {latexFormatName(format)} speed-up" )
         axs.set_ylabel( 'Speedup' )
         plt.savefig( f"Symmetric-speed-up/{format}-log.pdf")
         plt.close(fig)
         #head_df = filtered_df.head( head_size )
         #bottom_df = ascend_df.head( head_size )
         copy_df = df.copy()
         for f in formats:
            if not f in ['cusparse','CSR',format,non_symmetric_format]:
               #print( f"Droping {f}..." )
               #head_df.drop( labels=f, axis='columns', level=0, inplace=True )
               copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
         #head_df.to_html( f"Symmetric-speed-up/{format}-head.html" )
         copy_df.to_html( f"Symmetric-speed-up/{format}.html" )

####
# Comparison of speed-up w.r.t. LightSpMV
def csr_light_speedup_comparison( df, head_size=10 ):
   format = 'CSR< Light >'
   print( f"Writing comparison of speed-up of CSR Light compared to LightSPMV" )
   df['tmp'] = df[(format, 'GPU','bandwidth')]
   filtered_df=df.dropna(subset=[('tmp','','','')])
   ascend_df = filtered_df.copy()
   filtered_df.sort_values(by=[(format,'GPU','speed-up','LightSpMV Vector')],inplace=True,ascending=False)
   ascend_df.sort_values(by=[(format,'GPU','speed-up','LightSpMV Vector')],inplace=True,ascending=True)
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
   #head_df = filtered_df.head( head_size )
   #bottom_df = ascend_df.head( head_size )
   copy_df = df.copy()
   for f in formats:
      if not f in ['cusparse','CSR',format]:
         #print( f"Droping {f}..." )
         #head_df.drop( labels=f, axis='columns', level=0, inplace=True )
         copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
   #head_df.to_html( f"LightSpMV-speed-up-head.html" )
   copy_df.to_html( f"LightSpMV-speed-up-bottom.html" )


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
result = convert_data_frame( input_df, multicolumns, df_data, 0, 10000 )
compute_speedup( result, formats )

result.replace( to_replace=' ',value=np.nan,inplace=True)

####
# Make data analysis
def processDf( df, formats, head_size = 10 ):
   print( "Writting to HTML file..." )
   df.to_html( f'output.html' )

   # Generate tables and figures
   effective_bw_profile( df, formats, head_size )
   cusparse_comparison( df, formats, head_size )
   csr_comparison( df, formats, head_size )
   legacy_formats_comparison( df, formats, head_size )
   csr_speedup_comparison( df, formats, head_size )
   cusparse_speedup_comparison( df, formats, head_size )
   binary_matrices_comparison( df, formats, head_size )
   symmetric_matrices_comparison( df, formats, head_size )
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

head_size = 25
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
