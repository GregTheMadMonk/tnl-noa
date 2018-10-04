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
      using MeshFunctionPointer = Pointers::SharedPointer< MeshFunctionType >;
      using InterfaceMapPointer = Pointers::SharedPointer< InterfaceMapType >;
      
      void initInterface( const MeshFunctionPointer& input,
                          MeshFunctionPointer& output,
                          InterfaceMapPointer& interfaceMap );
      
      template< typename MeshEntity >
      __cuda_callable__ void updateCell( MeshFunctionType& u,
                                         const MeshEntity& cell,
                                         const RealType velocity = 1.0  );
      
      __cuda_callable__ bool updateCell( volatile Real sArray[18],
                                         int thri, const Real h,
                                         const Real velocity = 1.0 );
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
      using MeshFunctionPointer = Pointers::SharedPointer< MeshFunctionType >;
      using InterfaceMapPointer = Pointers::SharedPointer< InterfaceMapType >;      

      void initInterface( const MeshFunctionPointer& input,
                          MeshFunctionPointer& output,
                          InterfaceMapPointer& interfaceMap );
      
      template< typename MeshEntity >
      __cuda_callable__ void updateCell( MeshFunctionType& u,
                                         const MeshEntity& cell,
                                         const RealType velocity = 1.0 );
      
      __cuda_callable__ bool updateCell( volatile Real sArray[18][18],
                                         int thri, int thrj, const Real hx, const Real hy,
                                         const Real velocity = 1.0 );
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
      using MeshFunctionPointer = Pointers::SharedPointer< MeshFunctionType >;
      using InterfaceMapPointer = Pointers::SharedPointer< InterfaceMapType >;      

      void initInterface( const MeshFunctionPointer& input,
                          MeshFunctionPointer& output,
                          InterfaceMapPointer& interfaceMap );
      
      template< typename MeshEntity >
      __cuda_callable__ void updateCell( MeshFunctionType& u,
                                         const MeshEntity& cell,
                                         const RealType velocity = 1.0);
      
      __cuda_callable__ bool updateCell( volatile Real sArray[10][10][10],
                                         int thri, int thrj, int thrk, const Real hx, const Real hy, const Real hz,
                                         const Real velocity = 1.0 );
};

template < typename T1, typename T2 >
T1 meet2DCondition( T1 a, T1 b, const T2 ha, const T2 hb, const T1 value, double v = 1);

template < typename T1 >
__cuda_callable__ void sortMinims( T1 pom[] );


#ifdef HAVE_CUDA
template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller( const Functions::MeshFunction< Meshes::Grid< 1, Real, Device, Index > >& input, 
                                Functions::MeshFunction< Meshes::Grid< 1, Real, Device, Index > >& output,
                                Functions::MeshFunction< Meshes::Grid< 1, Real, Device, Index >, 1, bool >& interfaceMap  );

template < typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > > ptr,
                                      const Functions::MeshFunction< Meshes::Grid< 1, Real, Device, Index >, 1, bool >& interfaceMap,
                                      Functions::MeshFunction< Meshes::Grid< 1, Real, Device, Index > >& aux,
                                      bool *BlockIterDevice );

template < typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > > ptr,
                                      const Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index >, 2, bool >& interfaceMap,
                                      Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& aux,
                                      int *BlockIterDevice, int oddEvenBlock);
__global__ void CudaParallelReduc( int *BlockIterDevice, int *dBlock, int nBlocks );

/*template < typename Real, typename Device, typename Index >
__global__ void aux1( Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& aux, Real *dAux, int a );*/

template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller( const Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& input, 
                                Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& output,
                                Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index >, 2, bool >& interfaceMap );

template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller3d( const Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& input, 
                                  Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& output,
                                  Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index >, 3, bool >& interfaceMap );

template < typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > > ptr,
                                      const Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index >, 3, bool >& interfaceMap,
                                      Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& aux,
                                      int *BlockIterDevice );
#endif

#include "tnlDirectEikonalMethodsBase_impl.h"
