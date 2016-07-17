/***************************************************************************
                          tnlMultidiagonalMatrixSetter.h  -  description
                             -------------------
    begin                : Jan 2, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <mesh/tnlGrid.h>

namespace TNL {

template< typename MeshType >
class tnlMultidiagonalMatrixSetter
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex >
class tnlMultidiagonalMatrixSetter< tnlGrid< 1, MeshReal, Device, MeshIndex > >
{
   public:
      typedef MeshReal MeshRealType;
      typedef Device DeviceType;
      typedef MeshIndex MeshIndexType;
      typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      enum { Dimensions = 1 };

      template< typename Real, typename Index >
      static bool setupMatrix( const MeshType& mesh,
                               tnlMultidiagonalMatrix< Real, Device, Index >& matrix,
                               int stencilSize = 1,
                               bool crossStencil = false );

};

template< typename MeshReal,
          typename Device,
          typename MeshIndex >
class tnlMultidiagonalMatrixSetter< tnlGrid< 2, MeshReal, Device, MeshIndex > >
{
   public:
      typedef MeshReal MeshRealType;
      typedef Device DeviceType;
      typedef MeshIndex MeshIndexType;
      typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      enum { Dimensions = 2 };

      template< typename Real, typename Index >
      static bool setupMatrix( const MeshType& mesh,
                               tnlMultidiagonalMatrix< Real, Device, Index >& matrix,
                               int stencilSize = 1,
                               bool crossStencil = false );
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex >
class tnlMultidiagonalMatrixSetter< tnlGrid< 3, MeshReal, Device, MeshIndex > >
{
   public:
      typedef MeshReal MeshRealType;
      typedef Device DeviceType;
      typedef MeshIndex MeshIndexType;
      typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      enum { Dimensions = 3 };

      template< typename Real, typename Index >
      static bool setupMatrix( const MeshType& mesh,
                               tnlMultidiagonalMatrix< Real, Device, Index >& matrix,
                               int stencilSize = 1,
                               bool crossStencil = false );
};

} // namespace TNL

#include <matrices/tnlMultidiagonalMatrixSetter_impl.h>
