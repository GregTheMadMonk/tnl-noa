/***************************************************************************
                          tnlGridTraversal.h  -  description
                             -------------------
    begin                : Jul 28, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLGRIDTRAVERSAL_H_
#define TNLGRIDTRAVERSAL_H_

#include <mesh/tnlTraversal.h>

template< typename Real,
          typename Index,
          typename UserData >
class tnlTraversal< tnlGrid< 1, Real, tnlHost, Index >, UserData >
{
   public:
      typedef tnlGrid< 1, Real, tnlHost, Index > GridType;
      typedef Real RealType;
      typedef tnlHost DeviceType;
      typedef Index IndexType;
      typedef UserData UserDataType;

      static void processBoundaryCells( const GridType& grid,
                                        const UserDataType& userData );

      static void processInteriorCells( const GridType& grid,
                                        const UserDataType& userData );

      static void processAllCells( const GridType& grid,
                                   const UserDataType& userData );

      static void processBoundaryVertices( const GridType& grid,
                                           const UserDataType& userData );

      static void processInteriorVertices( const GridType& grid,
                                           const UserDataType& userData );

      static void processAllVertices( const GridType& grid,
                                      const UserDataType& userData );

};

template< typename Real,
          typename Index,
          typename UserData >
class tnlTraversal< tnlGrid< 1, Real, tnlCuda, Index >, UserData >
{
   public:
      typedef tnlGrid< 1, Real, tnlCuda, Index > GridType;
      typedef Real RealType;
      typedef tnlCuda DeviceType;
      typedef Index IndexType;
      typedef UserData UserDataType;

      static void processBoundaryCells( const GridType& grid,
                                        const UserDataType& userData );

      static void processInteriorCells( const GridType& grid,
                                        const UserDataType& userData );

      static void processAllCells( const GridType& grid,
                                   const UserDataType& userData );

      static void processBoundaryVertices( const GridType& grid,
                                           const UserDataType& userData );

      static void processInteriorVertices( const GridType& grid,
                                           const UserDataType& userData );

      static void processAllVertices( const GridType& grid,
                                      const UserDataType& userData );

};





#endif /* TNLGRIDTRAVERSAL_H_ */
