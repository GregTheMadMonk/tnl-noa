/***************************************************************************
                          tnlMeshFunctionNormGetter.h  -  description
                             -------------------
    begin                : Jan 5, 2016
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

#ifndef TNLMESHFUNCTIONNORMGETTER_H
#define	TNLMESHFUNCTIONNORMGETTER_H

template< typename MeshFunction,
          typename Mesh = typename MeshFunction::MeshType >
class tnlMeshFunctionNormGetter
{
};

template< typename MeshFunction,
          int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex >
class tnlMeshFunctionNormGetter< MeshFunction, tnlGrid< Dimensions, MeshReal, Device, MeshIndex > >
{
   public:
      
      typedef MeshFunction MeshFunctionType;
      typedef tnlGrid< Dimensions, MeshReal, Device, MeshIndex > GridType;
      typedef MeshReal MeshRealType;
      typedef Device DeviceType;
      typedef MeshIndex MeshIndexType;
      typedef typename MeshFunctionType::RealType RealType;
      
      static RealType getNorm( const MeshFunction& function,
                               const RealType& p )
      {
         if( p == 1.0 )
            return function.getMesh().getCellMeasure() * function.getData().lpNorm( 1.0 );
         if( p == 2.0 )
            return sqrt( function.getMesh().getCellMeasure() ) * function.getData().lpNorm( 2.0 );
         return pow( function.getMesh().getCellMeasure(), 1.0 / p ) * function.getData().lpNorm( p );
      }     
};

#endif	/* TNLMESHFUNCTIONNORMGETTER_H */

