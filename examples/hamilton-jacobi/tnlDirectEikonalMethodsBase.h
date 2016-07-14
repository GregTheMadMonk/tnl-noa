/* 
 * File:   tnlDirectEikonalMethodsBase.h
 * Author: oberhuber
 *
 * Created on July 14, 2016, 3:17 PM
 */

#pragma once 

#include <mesh/tnlGrid.h>
#include <functions/tnlMeshFunction.h>

template< typename Mesh >
class tnlDirectEikonalMethodsBase
{   
};

template< typename Real,
          typename Device,
          typename Index >
class tnlDirectEikonalMethodsBase< tnlGrid< 1, Real, Device, Index > >
{
   public:
      
      typedef tnlGrid< 1, Real, Device, Index > MeshType;
      typedef Real RealType;
      typedef Device DevcieType;
      typedef Index IndexType;
      
      template< typename MeshFunction >
      void initInterface( const MeshFunction& input,
                          MeshFunction& output );
      
      template< typename MeshFunction, typename MeshEntity >
      void updateCell( MeshFunction& u,
                       const MeshEntity& cell );
      
};


template< typename Real,
          typename Device,
          typename Index >
class tnlDirectEikonalMethodsBase< tnlGrid< 2, Real, Device, Index > >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > MeshType;
      typedef Real RealType;
      typedef Device DevcieType;
      typedef Index IndexType;
      typedef tnlMeshFunction< MeshType > MeshFunctionType;
      
      template< typename MeshFunction >
      void initInterface( const MeshFunction& input,
                          MeshFunction& output );
      
      template< typename MeshFunction, typename MeshEntity >
      void updateCell( MeshFunction& u,
                       const MeshEntity& cell );
};

template< typename Real,
          typename Device,
          typename Index >
class tnlDirectEikonalMethodsBase< tnlGrid< 3, Real, Device, Index > >
{
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > MeshType;
      typedef Real RealType;
      typedef Device DevcieType;
      typedef Index IndexType;
      typedef tnlMeshFunction< MeshType > MeshFunctionType;
      
      template< typename MeshFunction >
      void initInterface( const MeshFunction& input,
                          MeshFunction& output );
      
      template< typename MeshFunction, typename MeshEntity >
      void updateCell( MeshFunction& u,
                       const MeshEntity& cell );      
};

#include "tnlDirectEikonalMethodsBase_impl.h"
