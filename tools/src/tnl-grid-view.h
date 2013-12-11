/***************************************************************************
                          tnl-grid-view.h  -  description
                             -------------------
    begin                : Feb 11, 2010
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

#ifndef tnlGridOldVIEW_H_
#define tnlGridOldVIEW_H_

#include <config/tnlParameterContainer.h>
#include <core/tnlCurve.h>
#include <matrices/tnlCSRMatrix.h>
#include <matrices/tnlRgCSRMatrix.h>
#include <matrices/tnlAdaptiveRgCSRMatrix.h>
#include <legacy/mesh/tnlGridOld.h>
#include <fstream>

using namespace std;

template< typename Real, typename Device, typename Index >
bool ProcesstnlGridOld2D( const tnlString& file_name,
                       const tnlParameterContainer& parameters,
                       int file_index,
                       const tnlString& output_file_name,
                       const tnlString& output_file_format )
{
   int verbose = parameters. GetParameter< int >( "verbose");
   tnlGridOld< 2, Real, Device, Index > u( "u" ), resizedU( "resizedU" );
   if( ! u. load( file_name ) )
   {
      cout << " unable to restore the data " << endl;
      return false;
   }

   tnlGridOld< 2, Real, Device, Index >* output_u;

   int output_x_size( 0 ), output_y_size( 0 );
   parameters. GetParameter< int >( "output-x-size", output_x_size );
   parameters. GetParameter< int >( "output-y-size", output_y_size );
   Real scale = parameters. GetParameter< double >( "scale" );
   if( ! output_x_size && ! output_y_size && scale == ( Real ) 1.0 )
      output_u = &u;
   else
   {
      if( ! output_x_size ) output_x_size = u. getDimensions(). x();
      if( ! output_y_size ) output_y_size = u. getDimensions(). y();
      output_u = &resizedU;

      resizedU. setDimensions( tnlTuple< 2, Index >( output_x_size, output_y_size ) );
      resizedU. setDomain( u. getDomainLowerCorner(), u. getDomainUpperCorner() );

      const Real& hx = output_u -> getSpaceSteps(). x();
      const Real& hy = output_u -> getSpaceSteps(). y();
      for( Index i = 0; i < output_x_size; i ++ )
         for( Index j = 0; j < output_y_size; j ++ )
         {
            const Real x = output_u -> getDomainLowerCorner(). x() + Real( i ) * hx;
            const Real y = output_u -> getDomainUpperCorner(). y() + Real( j ) * hy;
            output_u -> setElement( i, j, scale * u. getValue( x, y ) );
         }
   }

   if( verbose )
      cout << " writing ... " << output_file_name;

   tnlList< Real > level_lines;
   parameters. GetParameter< tnlList< Real > >( "level-lines", level_lines );
   if( ! level_lines. isEmpty() )
   {
      tnlCurve< tnlTuple< 2, Real > > crv( "tnl-grid-view:curve" );
      int j;
      for( j = 0; j < level_lines. getSize(); j ++ )
         if( ! getLevelSetCurve( * output_u, crv, level_lines[ j ] ) )
         {
            cerr << "Unable to identify the level line " << level_lines[ j ] << endl;
            if( output_u != &u ) delete output_u;
            return false;
         }
      if( strcmp( output_file_name. getString() + output_file_name. getLength() - 4, ".tnl" ) == 0 )
      {
         if( ! crv. save( output_file_name ) )
         {
            cerr << " ... FAILED " << endl;
         }
      }
      else Write( crv, output_file_name. getString(), output_file_format. getString() );

   }
   else
   {
      if( ! output_u -> draw( output_file_name, output_file_format ) )
      {
         cerr << " ... FAILED " << endl;
      }
   }
   if( verbose )
      cout << " OK " << endl;
   return true;
}

template< typename Real, typename Device, typename Index >
bool ProcesstnlGridOld3D( const tnlString& file_name,
                       const tnlParameterContainer& parameters,
                       int file_index,
                       const tnlString& output_file_name,
                       const tnlString& output_file_format )
{
   int verbose = parameters. GetParameter< int >( "verbose");
   tnlGridOld< 3, Real, Device, Index > u( "u"), resizedU( "resizedU" );
   if( ! u. load( file_name ) )
   {
      cout << " unable to restore the data " << endl;
      return false;
   }

   tnlGridOld< 3, Real, Device, Index >* output_u;

   int output_x_size( 0 ), output_y_size( 0 ), output_z_size( 0 );
   parameters. GetParameter< int >( "output-x-size", output_x_size );
   parameters. GetParameter< int >( "output-y-size", output_y_size );
   parameters. GetParameter< int >( "output-y-size", output_z_size );
   Real scale = parameters. GetParameter< Real >( "scale" );
   if( ! output_x_size && ! output_y_size && ! output_z_size && scale == Real( 1.0 ) )
      output_u = &u;
   else
   {
      if( ! output_x_size ) output_x_size = u. getDimensions()[ tnlX ];
      if( ! output_y_size ) output_y_size = u. getDimensions()[ tnlY ];
      if( ! output_z_size ) output_z_size = u. getDimensions()[ tnlZ ];
      output_u = &resizedU;

      resizedU. setDimensions( tnlTuple< 3, Index >( output_x_size, output_y_size, output_z_size ) );
      resizedU. setDomain( u. getDomainLowerCorner(), u. getDomainUpperCorner() );

      const Real& hx = output_u -> getSpaceSteps(). x();
      const Real& hy = output_u -> getSpaceSteps(). y();
      const Real& hz = output_u -> getSpaceSteps(). z();

      for( Index i = 0; i < output_x_size; i ++ )
         for( Index j = 0; j < output_y_size; j ++ )
            for( Index k = 0; j < output_y_size; k ++ )
            {
               const Real x = output_u -> getDomainLowerCorner(). x() + Real( i ) * hx;
               const Real y = output_u -> getDomainLowerCorner(). y() + Real( j ) * hy;
               const Real z = output_u -> getDomainLowerCorner(). z() + Real( k ) * hz;
               output_u -> setElement( i, j, k, scale * u. getValue( x, y, z ) );
            }
   }

   if( verbose )
      cout << " writing " << output_file_name << " ... ";
   if( ! output_u -> draw( output_file_name, output_file_format ) )
   {
      cerr << " unable to write to " << output_file_name << endl;
   }
   else
      if( verbose )
         cout << " ... OK " << endl;
   return true;
}

template< typename Real >
bool drawCSRMatrix( const tnlString& output_file_name,
                    const tnlString& output_file_format )
{

}

template< typename Real >
bool ProcessCSRMatrix( const tnlString& file_name,
                       const tnlParameterContainer& parameters,
                       int file_index,
                       const tnlString& output_file_name,
                       const tnlString& output_file_format )
{
   /*int verbose = parameters. GetParameter< int >( "verbose");
   const tnlString matrixFormat = parameters. GetParameter< tnlString >( "matrix-format" );
   const int groupSize = parameters. GetParameter< int >( "matrix-group-size" );
   const int desiredChunkSize = parameters. GetParameter< int >( "desired-matrix-chunk-size" );
   const int cudaBlockSize = parameters. GetParameter< int >( "cuda-block-size" );
   bool sortMatrix = parameters. GetParameter< bool >( "sort-matrix" );
   tnlCSRMatrix< Real > matrix;//( "tnl-view:matrix" );
   tnlCSRMatrix< Real >* inputMatrix( &matrix );
   tnlFile file;
   if( ! file. open( file_name, tnlReadMode ) )
      return false;
   if( ! matrix. load( file ) )
   {
      cout << " unable to restore the data " << endl;
      file. close();
      return false;
   }
   file. close();
   tnlCSRMatrix< Real > sortedMatrix; //( "tnl-view:sortedMatrix" );
   if( sortMatrix )
   {
      if( verbose )
         cout << "Sorting the matrix rows..." << endl;

      tnlVector< int, tnlHost > rowPermutation( "rowPermutation" );
      matrix. sortRowsDecreasingly( rowPermutation );
      sortedMatrix. reorderRows( rowPermutation, matrix );
      inputMatrix = & sortedMatrix;
   }

   if( verbose )
      cout << " writing " << output_file_name << " ... " << endl;
   fstream outFile;
   outFile. open( output_file_name. getString(), ios :: out );
   if( matrixFormat == "" || matrixFormat == "csr" )
   {
      if( ! inputMatrix -> draw( outFile, output_file_format. getString(), inputMatrix, verbose ) )
         cerr << " unable to write to " << output_file_name << endl;
      else
         if( verbose )
            cout << " ... OK " << endl;
   }
   if( matrixFormat == "rg-csr" )
   {
      tnlRgCSRMatrix< Real > rgCsrMatrix( "rgCsrMatrix" );
      rgCsrMatrix. tuneFormat( groupSize );
      if( verbose )
         cout << "Converting CSR format to Row-grouped CSR ..." << endl;
      if( ! rgCsrMatrix. copyFrom( *inputMatrix ) )
         cerr << "I am not able to convert the CSR matrix to Row-grouped CSR format." << endl;
      else
         if( ! rgCsrMatrix. draw( outFile, output_file_format. getString(), inputMatrix, verbose ) )
            cerr << " unable to write to " << output_file_name << endl;
         else
            if( verbose )
               cout << " ... OK " << endl;
   }
   if( matrixFormat == "arg-csr" )
   {
      tnlAdaptiveRgCSRMatrix< Real > adaptiveRgCsrMatrix( "adaptiveRgCsrMatrix" );
      adaptiveRgCsrMatrix. tuneFormat( desiredChunkSize, cudaBlockSize );
      if( verbose )
         cout << "Converting CSR format to Row-grouped CSR ..." << endl;
      if( ! adaptiveRgCsrMatrix. copyFrom( * inputMatrix ) )
         cerr << "I am not able to convert the CSR matrix to Row-grouped CSR format." << endl;
      else
         if( ! adaptiveRgCsrMatrix. draw( outFile, output_file_format. getString(), inputMatrix, verbose ) )
            cerr << " unable to write to " << output_file_name << endl;
         else
            if( verbose )
               cout << " ... OK " << endl;
   }
   outFile. close();*/
   return true;
}



#endif /* tnlGridOldVIEW_H_ */
