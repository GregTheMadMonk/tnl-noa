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

#ifndef TNLGRIDVIEW_H_
#define TNLGRIDVIEW_H_

#include <core/tnlParameterContainer.h>
#include <diff/tnlGrid2D.h>
#include <diff/tnlGrid3D.h>
#include <core/tnlCurve.h>
#include <fstream>

using namespace std;

template< typename REAL >
bool ProcesstnlGrid2D( const tnlString& file_name,
                       const tnlParameterContainer& parameters,
                       int file_index,
                       const tnlString& output_file_name,
                       const tnlString& output_file_format )
{
   int verbose = parameters. GetParameter< int >( "verbose");
   tnlGrid2D< REAL > u;
   fstream file;
   file. open( file_name. Data(), ios :: in | ios :: binary );
   if( ! u. Load( file ) )
   {
      cout << " unable to restore the data " << endl;
      file. close();
      return false;
   }
   file. close();

   tnlGrid2D< REAL >* output_u;

   int output_x_size( 0 ), output_y_size( 0 );
   parameters. GetParameter< int >( "output-x-size", output_x_size );
   parameters. GetParameter< int >( "output-y-size", output_y_size );
   REAL scale = parameters. GetParameter< double >( "scale" );
   if( ! output_x_size && ! output_y_size && scale == 1.0 )
      output_u = &u;
   else
   {
      if( ! output_x_size ) output_x_size = u. GetXSize();
      if( ! output_y_size ) output_y_size = u. GetYSize();

      output_u = new tnlGrid2D< REAL >( "output-u",
                                        output_x_size,
                                        output_y_size,
                                        u. GetAx(),
                                        u. GetBx(),
                                        u. GetAy(),
                                        u. GetBy() );

      const REAL& hx = output_u -> GetHx();
      const REAL& hy = output_u -> GetHy();
      int i, j;
      for( i = 0; i < output_x_size; i ++ )
         for( j = 0; j < output_y_size; j ++ )
         {
            const REAL x = output_u -> GetAx() + i * hx;
            const REAL y = output_u -> GetAy() + j * hy;
            ( *output_u )( i, j ) = scale * u. Value( x, y );
         }
   }

   if( verbose )
      cout << " writing ... " << output_file_name;

   tnlList< REAL > level_lines;
   parameters. GetParameter< tnlList< REAL > >( "level-lines", level_lines );
   if( ! level_lines. IsEmpty() )
   {
      tnlCurve< tnlVector< 2, REAL > > crv;
      int j;
      for( j = 0; j < level_lines. Size(); j ++ )
         if( ! GetLevelSetCurve( * output_u, crv, level_lines[ j ] ) )
         {
            cerr << "Unable to identify the level line " << level_lines[ j ] << endl;
            if( output_u != &u ) delete output_u;
            return false;
         }
      if( ! Write( crv, output_file_name. Data(), output_file_format. Data() ) )
      {
         cerr << " ... FAILED " << endl;
      }
   }
   else
   {
      if( ! Draw( *output_u, output_file_name. Data(), output_file_format. Data() ) )
      {
         cerr << " ... FAILED " << endl;
      }
   }
   level_lines. EraseAll();
   if( output_u != &u ) delete output_u;
   if( verbose )
      cout << " OK " << endl;
}

template< typename REAL >
bool ProcesstnlGrid3D( const tnlString& file_name,
                       const tnlParameterContainer& parameters,
                       int file_index,
                       const tnlString& output_file_name,
                       const tnlString& output_file_format )
{
   int verbose = parameters. GetParameter< int >( "verbose");
   tnlGrid3D< REAL > u;
   fstream file;
   file. open( file_name. Data(), ios :: in | ios :: binary );
   if( ! u. Load( file ) )
   {
      cout << " unable to restore the data " << endl;
      file. close();
      return false;
   }
   file. close();

   tnlGrid3D< REAL >* output_u;

   int output_x_size( 0 ), output_y_size( 0 ), output_z_size( 0 );
   parameters. GetParameter< int >( "output-x-size", output_x_size );
   parameters. GetParameter< int >( "output-y-size", output_y_size );
   parameters. GetParameter< int >( "output-y-size", output_z_size );
   REAL scale = parameters. GetParameter< REAL >( "scale" );
   if( ! output_x_size && ! output_y_size && ! output_z_size && scale == 1.0 )
      output_u = &u;
   else
   {
      if( ! output_x_size ) output_x_size = u. GetXSize();
      if( ! output_y_size ) output_y_size = u. GetYSize();
      if( ! output_z_size ) output_z_size = u. GetZSize();

      output_u = new tnlGrid3D< REAL >( "output-u",
                                        output_x_size,
                                        output_y_size,
                                        output_z_size,
                                        u. GetAx(),
                                        u. GetBx(),
                                        u. GetAy(),
                                        u. GetBy(),
                                        u. GetAz(),
                                        u. GetBz() );
      const REAL& hx = output_u -> GetHx();
      const REAL& hy = output_u -> GetHy();
      const REAL& hz = output_u -> GetHz();
      int i, j, k;
      for( i = 0; i < output_x_size; i ++ )
         for( j = 0; j < output_y_size; j ++ )
            for( k = 0; j < output_y_size; k ++ )
            {
               const REAL x = output_u -> GetAx() + i * hx;
               const REAL y = output_u -> GetAy() + j * hy;
               const REAL z = output_u -> GetAz() + k * hz;
               ( *output_u )( i, j, k ) = scale * u. Value( x, y, z );
            }
   }

   if( verbose )
      cout << " writing " << output_file_name << " ... ";
   if( ! Draw( *output_u, output_file_name. Data(), output_file_format. Data() ) )
   {
      cerr << " unable to write to " << output_file_name << endl;
   }
   else
      if( verbose )
         cout << " ... OK " << endl;
   if( output_u != &u ) delete output_u;
}



#endif /* TNLGRIDVIEW_H_ */
