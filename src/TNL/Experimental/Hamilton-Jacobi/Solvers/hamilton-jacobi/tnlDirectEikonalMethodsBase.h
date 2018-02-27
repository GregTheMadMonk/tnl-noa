/* 
 * File:   tnlDirectEikonalMethodsBase.h
 * Author: oberhuber
 *
 * Created on July 14, 2016, 3:17 PM
 */

#pragma once 

#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/MeshFunction.h>

using namespace TNL;

template< typename Mesh >
class tnlDirectEikonalMethodsBase
{   
};

template< typename Real,
          typename Device,
          typename Index >
class tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > >
{
   public:
      
      typedef Meshes::Grid< 1, Real, Device, Index > MeshType;
      typedef Real RealType;
      typedef Device DevcieType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      typedef Functions::MeshFunction< MeshType, 1, bool > InterfaceMapType;
      
      void initInterface( const MeshFunctionType& input,
                          MeshFunctionType& output,
                          InterfaceMapType& interfaceMap );
      
      template< typename MeshEntity >
      void updateCell( MeshFunctionType& u,
                       const MeshEntity& cell);
      
};


template< typename Real,
          typename Device,
          typename Index >
class tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >
{
   public:
      
      typedef Meshes::Grid< 2, Real, Device, Index > MeshType;
      typedef Real RealType;
      typedef Device DevcieType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      typedef Functions::MeshFunction< MeshType, 2, bool > InterfaceMapType;

      void initInterface( const MeshFunctionType& input,
                          MeshFunctionType& output,
                          InterfaceMapType& interfaceMap );
      
      template< typename MeshEntity >
      void updateCell( MeshFunctionType& u,
                       const MeshEntity& cell,
                       double velocity = 1.0 );
};

template< typename Real,
          typename Device,
          typename Index >
class tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >
{
   public:
      
      typedef Meshes::Grid< 3, Real, Device, Index > MeshType;
      typedef Real RealType;
      typedef Device DevcieType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      typedef Functions::MeshFunction< MeshType, 3, bool > InterfaceMapType;

      void initInterface( const MeshFunctionType& input,
                          MeshFunctionType& output,
                          InterfaceMapType& interfaceMap );
      
      template< typename MeshEntity >
      void updateCell( MeshFunctionType& u,
                       const MeshEntity& cell,
                       double velocity = 1.0);
      
      Real sort( Real a, Real b, Real c,
                 const RealType& ha,
                 const RealType& hb,
                 const RealType& hc ); 
};

template < typename T1, typename T2 >
T1 meet2DCondition( T1 a, T1 b, const T2 ha, const T2 hb, const T1 value, double v = 1);

template < typename T1 >
void sortMinims( T1 pom[] );

#include "tnlDirectEikonalMethodsBase_impl.h"
