#!/usr/bin/python3

import json
import pandas as pd
from pandas.io.json import json_normalize


def slugify(s):
   s = str(s).strip().replace(' ', '_')
   return re.sub(r'(?u)[^-\w.]', '', s)

####
# Parse input file
print( "Parsing input file...." )
with open('sparse-matrix-benchmark.log') as f:
    d = json.load(f)
input_df = json_normalize( d, record_path=['results'] )
#input_df.to_html( "orig-pandas.html" )

####
# Create multiindex for columns

# Get format names - TODO: the first benchmark might not have all of them
matrixName = input_df.iloc[0]['matrix name']
df_matrix = input_df.loc[input_df['matrix name'] == matrixName]
formats = df_matrix.loc[:,'format']
level1 = [ 'Matrix name', 'rows', 'columns' ]
level2 = [ '',            '',     ''        ]
level3 = [ '',            '',     ''        ]
df_data = [[ ' ',' ',' ']]
for format in formats:
   for device in ['CPU','GPU']:
      for data in ['bandwidth' ]: #,'time','speed-up','non-zeros','stddev','stddev/time','diff.max','diff.l2']:
         level1.append( format )
         level2.append( device )
         level3.append( data )
         df_data[ 0 ].append( ' ' )
multiColumns = pd.MultiIndex.from_arrays([ level1, level2, level3 ] )
frames = []

in_idx = 0
out_idx = 0
max_out_idx = 50
print( "Converting data..." )
while in_idx < len(input_df.index) and out_idx < max_out_idx:
   matrixName = input_df.iloc[in_idx]['matrix name']
   df_matrix = input_df.loc[input_df['matrix name'] == matrixName]
   print( out_idx, ":", in_idx, "/", len(input_df.index), ":", matrixName )
   aux_df = pd.DataFrame( df_data, columns = multiColumns, index = [out_idx] )
   for index,row in df_matrix.iterrows():
      aux_df.iloc[0]['Matrix name'] = row['matrix name']
      aux_df.iloc[0]['rows']        = row['rows']
      aux_df.iloc[0]['columns']     = row['columns']
      current_format = row['format']
      current_device = row['device']
      #print( current_format + " / " + current_device )
      aux_df.iloc[0][(current_format,current_device,'bandwidth')]   = row['bandwidth']
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

print( "Merging data into one frame..." )
result = pd.concat( frames )

print( "Setting data types..." )
for format in formats:
   for device in ['CPU','GPU']:
      #df['eps'] = pd.to_numeric(df['eps'], errors='coerce')
      print(result[(format,device,'bandwidth')].toList())
      result[(format,device,'bandwidth')] = pd.to_numeric( result[(format,device,'bandwidth')], errors='coerce' )
      #result[(format,device,'time')].astype('float64')
      #result[(format,device,'speed-up')].astype('float64')
      #result[(format,device,'non-zeros')].astype('int64')
      #result[(format,device,'stddev')].astype('float64')
      #result[(format,device,'stddev/time')].astype('float64')
      #result[(format,device,'diff.max')].astype('float64')
      #result[(format,device,'diff.l2')].astype('float64')

print( "Writting to HTML file..." )
result.to_html( 'output.html' )



#result.sort_values(by=[('cusparse','GPU','bandwidth')],inplace=True,ascending=False)
#for format in formats:
#   cusparse_bw = result[('cusparse','GPU','bandwidth')].toList()
#   format_bw = result[(format,'GPU','bandwidth')].toList()
#

#for format in formats:
#   result.sort_values(by=[(format,'GPU','bandwidth')],inplace=True,ascending=False)
