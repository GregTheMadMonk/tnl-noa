/***************************************************************************
                          tnlForwardFiniteDifference.h  -  description
                             -------------------
    begin                : Jan 9, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
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

#ifndef TNLFORWARDFINITEDIFFERENCE_H
#define	TNLFORWARDFINITEDIFFERENCE_H

#include <mesh/tnlGrid.h>
#include <operators/fdm/tnlFiniteDifferences.h>
#include <operators/fdm/tnlExactDifference.h>
#include <operators/tnlOperator.h>

template< typename Mesh,
          int Xdifference = 0,
          int YDifference = 0,
          int ZDifference = 0,
          typename RealType = typename Mesh::RealType,
          typename IndexType = typename Mesh::IndexType >
class tnlForwardFiniteDifference
{    
};

template< int Dimensions,
          typename MeshReal,
          typename MeshDevice,
          typename MeshIndex,
          int XDifference,
          int YDifference,
          int ZDifference,
          typename Real,
          typename Index >
class tnlForwardFiniteDifference< tnlGrid< Dimensions, MeshReal, MeshDevice, MeshIndex >, XDifference, YDifference, ZDifference, Real, Index >
: public tnlOperator< tnlGrid< Dimensions, MeshReal, MeshDevice, MeshIndex >,
                      MeshInteriorDomain, Dimensions, Dimensions, Real, Index >
{
   public:
      
      typedef tnlGrid< Dimensions, MeshReal, MeshDevice, MeshIndex > MeshType;
      typedef Real RealType;
      typedef MeshDevice DeviceType;
      typedef Index IndexType;
      typedef tnlExactDifference< Dimensions, XDifference, YDifference, ZDifference > ExactOperatorType;
      
      static constexpr int getMeshDimensions() { return Dimensions; }
      
      static tnlString getType()
      {
         return tnlString( "tnlForwardFiniteDifference< " ) +
            MeshType::getType() + ", " +
            tnlString( XDifference ) + ", " +
            tnlString( YDifference ) + ", " +
            tnlString( ZDifference ) + ", " +
            ::getType< RealType >() + ", " +
            ::getType< IndexType >() + " >";
      }

      
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      inline Real operator()( const MeshFunction& u,
                              const MeshEntity& entity,
                              const RealType& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntitiesDimensions() == Dimensions,
            "Finite differences can be evaluate only on mesh cells, i.e. the dimensions count of the mesh entities of mesh function must be the same as mesh dimensions count." );
         const int XDirection = 1 * ( XDifference != 0 );
         const int YDirection = 1 * ( YDifference != 0 );
         const int ZDirection = 1 * ( ZDifference != 0 );

         return tnlFiniteDifferences<
            MeshType,
            Real,
            Index,
            XDifference,
            YDifference,
            ZDifference,
            XDirection,
            YDirection,
            ZDirection >::getValue( u, entity );

      }
};

#endif	/* TNLFORWARDFINITEDIFFERENCE_H */

