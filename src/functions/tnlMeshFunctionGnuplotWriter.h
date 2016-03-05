/***************************************************************************
                          tnlMeshFunctionGnuplotWriter.h  -  description
                             -------------------
    begin                : Jan 28, 2016
    copyright            : (C) 2016 by oberhuber
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

#ifndef TNLMESHFUNCTIONGNUPLOTWRITER_H
#define	TNLMESHFUNCTIONGNUPLOTWRITER_H

#include<mesh/tnlGrid.h>

template< typename, int, typename > class tnlMeshFunction;

template< typename MeshFunction >
class tnlMeshFunctionGnuplotWriter
{
   public:

      static bool write( const MeshFunction& function,
                         ostream& str );
};

/***
 * 1D grids cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
class tnlMeshFunctionGnuplotWriter< tnlMeshFunction< tnlGrid< 1, MeshReal, Device, MeshIndex >, 1, Real > >
{
   public:
      typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef tnlMeshFunction< MeshType, 1, RealType > MeshFunctionType;

      static bool write( const MeshFunctionType& function,
                         ostream& str );
};

/***
 * 1D grids vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
class tnlMeshFunctionGnuplotWriter< tnlMeshFunction< tnlGrid< 1, MeshReal, Device, MeshIndex >, 0, Real > >
{
   public:
      typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef tnlMeshFunction< MeshType, 0, RealType > MeshFunctionType;

      static bool write( const MeshFunctionType& function,
                         ostream& str );
};

/***
 * 2D grids cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
class tnlMeshFunctionGnuplotWriter< tnlMeshFunction< tnlGrid< 2, MeshReal, Device, MeshIndex >, 2, Real > >
{
   public:
      typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef tnlMeshFunction< MeshType, 2, RealType > MeshFunctionType;

      static bool write( const MeshFunctionType& function,
                         ostream& str );
};

/***
 * 2D grids faces
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
class tnlMeshFunctionGnuplotWriter< tnlMeshFunction< tnlGrid< 2, MeshReal, Device, MeshIndex >, 1, Real > >
{
   public:
      typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef tnlMeshFunction< MeshType, 1, RealType > MeshFunctionType;

      static bool write( const MeshFunctionType& function,
                         ostream& str );
};


/***
 * 2D grids vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
class tnlMeshFunctionGnuplotWriter< tnlMeshFunction< tnlGrid< 2, MeshReal, Device, MeshIndex >, 0, Real > >
{
   public:
      typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef tnlMeshFunction< MeshType, 0, RealType > MeshFunctionType;

      static bool write( const MeshFunctionType& function,
                         ostream& str );
};


#endif	/* TNLMESHFUNCTIONGNUPLOTWRITER_H */

