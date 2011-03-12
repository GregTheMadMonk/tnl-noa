/***************************************************************************
                          sparse-matrix-benchmark.cpp  -  description
                             -------------------
    begin                : Jul 27, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "sparse-matrix-benchmark-def.h"

#include "sparse-matrix-benchmark.h"
#include <string.h>
#include <core/tnlObject.h>
#include <core/tnlFile.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;

   if( conf_desc. ParseConfigDescription( CONFIG_DESCRIPTION_FILE ) != 0 )
      return 1;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return 1;
   }
   tnlString str_input_file = parameters. GetParameter< tnlString >( "input-file" );
   tnlString str_input_mtx_file = parameters. GetParameter< tnlString >( "input-mtx-file" );
   tnlString log_file_name = parameters. GetParameter< tnlString >( "log-file" );
   double stop_time = parameters. GetParameter< double >( "stop-time" );
   int verbose = parameters. GetParameter< int >( "verbose");
   char* input_file = str_input_file. getString();

   tnlFile binaryFile;
   if( ! binaryFile. open( str_input_file, tnlReadMode ) )
   {
      cerr << "I am not able to open the file " << str_input_file << "." << endl;
      return 1;
   }
   tnlString object_type;
   if( ! getObjectType( binaryFile, object_type ) )
   {
      cerr << "Unknown object ... SKIPPING!" << endl;
      return EXIT_FAILURE;
   }
   if( verbose )
      cout << object_type << " detected ... " << endl;
   binaryFile. close();

   int spmv_csr_iter( 0 ),
            spmv_hyb_iter( 0 ),
            spmv_coacsr_iter[ 6 ],
            spmv_cuda_coacsr_iter[ 6 ],
            spmv_cuda_rgcsr_iter[ 10 ],
            spmv_ellpack_iter( 0 ),
            spmv_fast_csr_iter( 0 ),
            spmv_coa_fast_csr_iter[ 6 ],
            spmv_cuda_coa_fast_csr_iter[ 6 ],
            coa_fast_csr_max_cs_dict_size[ 6 ];
   double spmv_csr_gflops( 0.0 ),
            spmv_hyb_gflops( 0.0 ),
            spmv_coacsr_gflops[ 6 ],
            spmv_cuda_coacsr_gflops[ 6 ],
            spmv_cuda_rgcsr_gflops[ 10 ],
            spmv_ellpack_gflops( 0.0 ),
            spmv_fast_csr_gflops( 0.0 ),
            spmv_coa_fast_csr_gflops[ 6 ],
            spmv_cuda_coa_fast_csr_gflops[ 6 ];
   double coa_csr_artificial_zeros[ 7 ],
   ellpack_artificial_zeros( 0.0 ),
   fast_csr_compression( 0.0 ),
   coa_fast_csr_compression[ 5 ];
   int size, nonzero_elements;

   if( object_type == "tnlCSRMatrix< float, tnlHost >")
      benchmarkMatrix< float >( str_input_file,
               str_input_mtx_file,
               verbose,
               stop_time,
               size,
               nonzero_elements,
               spmv_csr_iter,
               spmv_hyb_iter,
               spmv_coacsr_iter,
               spmv_cuda_coacsr_iter,
               spmv_cuda_rgcsr_iter,
               spmv_fast_csr_iter,
               spmv_coa_fast_csr_iter,
               spmv_cuda_coa_fast_csr_iter,
               spmv_ellpack_iter,
               spmv_csr_gflops,
               spmv_hyb_gflops,
               spmv_coacsr_gflops,
               spmv_cuda_coacsr_gflops,
               spmv_cuda_rgcsr_gflops,
               spmv_fast_csr_gflops,
               spmv_coa_fast_csr_gflops,
               spmv_cuda_coa_fast_csr_gflops,
               spmv_ellpack_gflops,
               coa_csr_artificial_zeros,
               ellpack_artificial_zeros,
               fast_csr_compression,
               coa_fast_csr_compression,
               coa_fast_csr_max_cs_dict_size );


   if( object_type == "tnlCSRMatrix< double, tnlHost >" )
   {
#if CUDA_ARCH > 12
      benchmarkMatrix< double >( str_input_file,
               str_input_mtx_file,
               verbose,
               stop_time,
               size,
               nonzero_elements,
               spmv_csr_iter,
               spmv_hyb_iter,
               spmv_coacsr_iter,
               spmv_cuda_coacsr_iter,
               spmv_cuda_rgcsr_iter,
               spmv_fast_csr_iter,
               spmv_coa_fast_csr_iter,
               spmv_cuda_coa_fast_csr_iter,
               spmv_ellpack_iter,
               spmv_csr_gflops,
               spmv_hyb_gflops,
               spmv_coacsr_gflops,
               spmv_cuda_coacsr_gflops,
               spmv_cuda_rgcsr_gflops,
               spmv_fast_csr_gflops,
               spmv_coa_fast_csr_gflops,
               spmv_cuda_coa_fast_csr_gflops,
               spmv_ellpack_gflops,
               coa_csr_artificial_zeros,
               ellpack_artificial_zeros,
               fast_csr_compression,
               coa_fast_csr_compression,
               coa_fast_csr_max_cs_dict_size );
#else
   cerr << "Skipping double precision test because this CUDA device does not support the double precision." << endl;
#endif
   }
   //binaryFile. close();


   fstream log_file;
   if( log_file_name )
   {
      log_file. open( log_file_name. getString(), ios :: out | ios :: app );
      if( ! log_file )
      {
         cerr << "Unable to open log file " << log_file_name << " for appending logs." << endl;
         return EXIT_FAILURE;
      }
      cout << "Writing to log file " << log_file_name << "..." << endl;
      double all_elements = ( double ) size;
      all_elements *= all_elements;
      log_file << "| " << left << setw( 98 ) << input_file << right
               << "| " << setw( 7 ) << right << size
               << " | " << setw( 10 ) << right << nonzero_elements
               << " | " << setw( 8 ) << right << setprecision( 2 ) << ( double ) nonzero_elements / all_elements << " %"
               << " | " << setw( 6 ) << setprecision( 2 ) << spmv_csr_gflops
               << " | " << setw( 6 ) << setprecision( 2 ) << spmv_hyb_gflops
               << " | " << setw( 8 ) << setprecision( 2 ) << spmv_hyb_gflops / spmv_csr_gflops;
      for( int i = 0; i < 6; i ++ )
      {
         log_file << " | " << setw( 10 ) << setprecision( 2 ) << fixed << coa_csr_artificial_zeros[ i ] << " %"
                  << " | " << setw( 6 ) << setprecision( 2 ) << spmv_coacsr_gflops[ i ]
                  << " | " << setw( 8 ) << setprecision( 2 ) << spmv_coacsr_gflops[ i ] / spmv_csr_gflops
                  << " | " << setw( 6 ) << setprecision( 2 ) << spmv_cuda_coacsr_gflops[ i ]
                  << " | " << setw( 8 ) << setprecision( 2 ) << spmv_cuda_coacsr_gflops[ i ] / spmv_csr_gflops;
      }
      for( int i = 0; i < 8; i ++ )
      {
         log_file << " | " << setw( 6 ) << setprecision( 2 ) << spmv_cuda_rgcsr_gflops[ i ]
                  << " | " << setw( 8 ) << setprecision( 2 ) << spmv_cuda_rgcsr_gflops[ i ] / spmv_csr_gflops;
      }


      log_file << " | " << setw( 5 ) << setprecision( 2 ) << fixed << fast_csr_compression << " %"
               << " | " << setw( 6 ) << setprecision( 2 ) << spmv_fast_csr_gflops;
      for( int i = 0; i < 5; i ++ )
      {
         log_file << " | " << setw( 12 ) << setprecision( 2 ) << coa_fast_csr_max_cs_dict_size[ i ]
                  << " | " << setw( 6 ) << setprecision( 2 ) << spmv_coa_fast_csr_gflops[ i ]
                  << " | " << setw( 8 ) << setprecision( 2 ) << spmv_coa_fast_csr_gflops[ i ] / spmv_csr_gflops
                  << " | " << setw( 6 ) << setprecision( 2 ) << spmv_cuda_coa_fast_csr_gflops[ i ]
                  << " | " << setw( 8 ) << setprecision( 2 ) << spmv_cuda_coa_fast_csr_gflops[ i ] / spmv_csr_gflops;
      }
      log_file << " |" << endl;
   }
   return EXIT_SUCCESS;
}
