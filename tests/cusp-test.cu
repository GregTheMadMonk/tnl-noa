/***************************************************************************
                          cusp-test.cu  -  description
                             -------------------
    begin                : Oct 3, 2010
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

#include <cusp-test.h>

bool cuspSpMVTest( const char* mtx_file_name,
                   double& time,
                   int nonzero_elements,
                   int& spmv_hyb_iter,
                   double& spmv_hyb_gflops,
                   tnlLongVector< float >& hyb_result )
{
   return cuspSpMVTestStarter( mtx_file_name,
                               time,
                               nonzero_elements,
                               spmv_hyb_iter,
                               spmv_hyb_gflops,
                               hyb_result );
}

bool cuspSpMVTest( const char* mtx_file_name,
                   double& time,
                   int nonzero_elements,
                   int& spmv_hyb_iter,
                   double& spmv_hyb_gflops,
                   tnlLongVector< double >& hyb_result )
{
   return cuspSpMVTestStarter( mtx_file_name,
                               time,
                               nonzero_elements,
                               spmv_hyb_iter,
                               spmv_hyb_gflops,
                               hyb_result );
}