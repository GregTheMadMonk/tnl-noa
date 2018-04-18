/* 
 * File:   tnlDirectEikonalMethodsBase.h
 * Author: oberhuber
 *
 * Created on July 14, 2016, 3:17 PM
 */

#pragma once 

#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Devices/Cuda.h>

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
      using MeshFunctionPointer = SharedPointer< MeshFunctionType >;
      using InterfaceMapPointer = SharedPointer< InterfaceMapType >;
      
      void initInterface( const MeshFunctionPointer& input,
                          MeshFunctionPointer& output,
                          InterfaceMapPointer& interfaceMap );
      
      template< typename MeshEntity >
      __cuda_callable__ void updateCell( MeshFunctionType& u,
                                         const MeshEntity& cell );
      
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
      using MeshFunctionPointer = SharedPointer< MeshFunctionType >;
      using InterfaceMapPointer = SharedPointer< InterfaceMapType >;      

      void initInterface( const MeshFunctionPointer& input,
                          MeshFunctionPointer& output,
                          InterfaceMapPointer& interfaceMap );
      
      template< typename MeshEntity >
      __cuda_callable__ void updateCell( MeshFunctionType& u,
                                         const MeshEntity& cell,
                                         const RealType velocity = 1.0 );
   protected:
       
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
      using MeshFunctionPointer = SharedPointer< MeshFunctionType >;
      using InterfaceMapPointer = SharedPointer< InterfaceMapType >;      

      void initInterface( const MeshFunctionPointer& input,
                          MeshFunctionPointer& output,
                          InterfaceMapPointer& interfaceMap );
      
      template< typename MeshEntity >
      __cuda_callable__ void updateCell( MeshFunctionType& u,
                                         const MeshEntity& cell,
                                         const RealType velocity = 1.0);
      
      /*Real sort( Real a, Real b, Real c,
                 const RealType& ha,
                 const RealType& hb,
                 const RealType& hc ); */
};

template < typename T1, typename T2 >
T1 meet2DCondition( T1 a, T1 b, const T2 ha, const T2 hb, const T1 value, double v = 1);

template < typename T1 >
__cuda_callable__ void sortMinims( T1 pom[] );


#ifdef HAVE_CUDA
template < typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index >, 2, bool >& interfaceMap,
                                      Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& aux );

template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller( const Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& input, 
                                Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& output,
                                Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index >, 2, bool >& interfaceMap );

//__global__ void CudaInitCaller( const Functions::MeshFunction< Meshes::Grid< 2, double, TNL::Devices::Cuda, int > >& input );
#endif

#include "tnlDirectEikonalMethodsBase_impl.h"
