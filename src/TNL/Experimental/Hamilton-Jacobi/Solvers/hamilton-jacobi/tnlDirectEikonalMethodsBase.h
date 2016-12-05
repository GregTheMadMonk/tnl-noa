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
      typedef tnlMeshFunction< MeshType > MeshFunctionType;
      typedef tnlMeshFunction< MeshType, 1, bool > InterfaceMapType;
      
      void initInterface( const MeshFunctionType& input,
                          MeshFunctionType& output,
                          InterfaceMapType& interfaceMap );
      
      template< typename MeshEntity >
      void updateCell( MeshFunctionType& u,
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
      typedef tnlMeshFunction< MeshType, 2, bool > InterfaceMapType;

      void initInterface( const MeshFunctionType& input,
                          MeshFunctionType& output,
                          InterfaceMapType& interfaceMap );
      
      template< typename MeshEntity >
      void updateCell( MeshFunctionType& u,
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
      typedef tnlMeshFunction< MeshType, 3, bool > InterfaceMapType;

      void initInterface( const MeshFunctionType& input,
                          MeshFunctionType& output,
                          InterfaceMapType& interfaceMap );
      
      template< typename MeshEntity >
      void updateCell( MeshFunctionType& u,
                       const MeshEntity& cell );      
};

#include "tnlDirectEikonalMethodsBase_impl.h"
