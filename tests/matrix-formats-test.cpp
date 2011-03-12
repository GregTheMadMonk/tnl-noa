/***************************************************************************
                          matrix-formats-test.cpp  -  description
                             -------------------
    begin                : Jul 23, 2010
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

#include "matrix-formats-test-def.h"

#include "matrix-formats-test.h"
#include <string.h>
#include <core/tnlObject.h>
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
   tnlString input_file = parameters. GetParameter< tnlString >( "input-file" );
   bool have_mtx_file( false );
   tnlString input_mtx_file;
   //if( parameters. CheckParameter( "input-mtx-file ") )
	   input_mtx_file = parameters. GetParameter< tnlString >( "input-mtx-file" );
   tnlString log_file_name = parameters. GetParameter< tnlString >( "log-file" );
   int verbose = parameters. GetParameter< int >( "verbose");

    tnlString object_type;
    if( ! getObjectType( input_file. getString(), object_type ) )
    {
       cerr << "Unknown object ... SKIPPING!" << endl;
       return EXIT_FAILURE;
    }
    if( verbose )
    	cout << object_type << " detected ... " << endl;

    bool test_result;
    bool have_full_matrix( false );
    bool test_full_csr( false );
    bool test_coa_csr( false );
    bool test_cuda_coa_csr( false );
    bool test_fast_csr( false );
    bool test_coa_fast_csr( false );
    bool test_cuda_coa_fast_csr( false );
    bool test_ellpack( false );

    if( object_type == "tnlCSRMatrix< float, tnlHost >")
    	test_result = testMatrixFormats< float >( input_file,
    		          	                     input_mtx_file,
						        		            verbose,
								                  have_full_matrix,
								                  test_full_csr,
								                  test_coa_csr,
								                  test_cuda_coa_csr,
								                  test_fast_csr,
								                  test_coa_fast_csr,
								                  test_cuda_coa_fast_csr,
      								              test_ellpack );

    if( object_type == "tnlCSRMatrix< double, tnlHost >")
	   test_result = testMatrixFormats< double >( input_file,
		           			        			  input_mtx_file,
						          	        	  verbose,
								                  have_full_matrix,
								                  test_full_csr,
								                  test_coa_csr,
								                  test_cuda_coa_csr,
								                  test_fast_csr,
								                  test_coa_fast_csr,
								                  test_cuda_coa_fast_csr,
								                  test_ellpack );


   fstream log_file;
   if( log_file_name )
   {
	   log_file. open( log_file_name. getString(), ios :: out | ios :: app );
	   if( ! log_file )
	   {
		   cerr << "Unable to open log file " << log_file_name << " for appending logs." << endl;
		   return EXIT_FAILURE;
	   }
	   log_file << "| " << left << setw( 98 ) << input_mtx_file << right;

	   if( have_full_matrix )
	   {
		   log_file << "|" << setw( 12 ) << "YES ";
		   if( test_full_csr )
			   log_file << "|" << setw( 12 ) << "YES ";
		   else
			   log_file << "|" << setw( 12 ) << "NO ";
	   }
	   else
		   log_file << "|" << setw( 12 ) << "NO "
		            << "|" << setw( 12 ) << "N/A ";

	   if( test_coa_csr )
		   log_file << "|" << setw( 12 ) << "YES ";
	   else
		   log_file << "|" << setw( 12 ) << "NO ";

	   if( test_cuda_coa_csr )
		   log_file << "|" << setw( 12 ) << "YES ";
	   else
		   log_file << "|" << setw( 12 ) << "NO ";


	   if( test_fast_csr )
	      log_file << "|" << setw( 12 ) << "YES ";
	   else
	      log_file << "|" << setw( 12 ) << "NO ";

	   if( test_coa_fast_csr )
		   log_file << "|" << setw( 12 ) << "YES ";
	   else
		   log_file << "|" << setw( 12 ) << "NO ";

	   /*if( test_ellpack )
	      log_file << "|" << setw( 12 ) << "YES ";
	   else
	      log_file << "|" << setw( 12 ) << "NO ";*/

	   log_file << " |" << endl;
	   log_file. close();
   }
}
