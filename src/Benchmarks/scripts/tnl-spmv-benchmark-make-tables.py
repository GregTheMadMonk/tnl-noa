#! /usr/bin/env python3

import os
import re

import pandas

from TNL.LogParser import LogParser

#pandas.options.display.float_format = "{:.2f}".format
pandas.options.display.float_format = "{:.2e}".format
pandas.options.display.width = 0    # auto-detect terminal width for formatting
pandas.options.display.max_rows = None

def parse_file(fname):
    parser = LogParser()
    for metadata, df in parser.readFile(fname):

        # drop useless columns
        #df.drop(("CPU", "bandwidth"), axis=1, inplace=True)
        #df.drop(("GPU", "bandwidth"), axis=1, inplace=True)
        #df.drop(("CPU", "speedup"), axis=1, inplace=True)
        #df.drop(("GPU", "speedup"), axis=1, inplace=True)

        # parse the matrix name
        #matrix = metadata["title"].split("(")[1].split(")")[0]
        #matrix = "/".join(matrix.split("/")[-3:])

        # put the matrix name into the dataframe
        #df["matrix"] = matrix
        #idx = ["matrix", "rows", "columns", "max elements per row", "operation"]
        #df = df.reset_index().set_index(idx)
        #df.sort_index(inplace=True)

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

log_files = ["log-files/sparse-matrix-benchmark.log"]

dfs = []
for f in log_files:
    for df in parse_file(f):
        dfs.append(df)

df = pandas.concat(dfs)

## Post-processing

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

pandas.options.display.float_format = '{:,.4f}'.format
df.to_html("log.html")
#print( df )


# compute speedup between CGS and CWY
#df["CGS/CWY ratio", "CPU", "time"] = df["CGS-GMRES (Jacobi)", "CPU", "time"] / df["CWY-GMRES (Jacobi)", "CPU", "time"]
#df["CGS/CWY ratio", "GPU", "time"] = df["CGS-GMRES (Jacobi)", "GPU", "time"] / df["CWY-GMRES (Jacobi)", "GPU", "time"]


# print rows where CWY is faster
#print()
#print("Matrices for which CWY was faster:")
#print(df.loc[df["CGS/CWY ratio", "GPU", "time"] >= 1])
