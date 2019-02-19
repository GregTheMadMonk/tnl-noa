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
    typedef TNL::Containers::Array< int, Device, IndexType > ArrayContainer;
    using MeshFunctionPointer = Pointers::SharedPointer< MeshFunctionType >;
    using InterfaceMapPointer = Pointers::SharedPointer< InterfaceMapType >;
    
    void initInterface( const MeshFunctionPointer& input,
            MeshFunctionPointer& output,
            InterfaceMapPointer& interfaceMap );
    
    template< typename MeshEntity >
    __cuda_callable__ bool updateCell( MeshFunctionType& u,
            const MeshEntity& cell,
            const RealType velocity = 1.0 );
    
    template< int sizeSArray >
    __cuda_callable__ bool updateCell( volatile Real *sArray,
            int thri, int thrj, const Real hx, const Real hy,
            const Real velocity = 1.0 );
    
    template< int sizeSArray >
    void updateBlocks( const InterfaceMapType& interfaceMap,
            MeshFunctionType& aux,
            MeshFunctionType& helpFunc,
            ArrayContainer& BlockIterHost, int numThreadsPerBlock/*, Real **sArray*/ );
    
    void getNeighbours( ArrayContainer& BlockIterHost, int numBlockX, int numBlockY  );
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
    typedef TNL::Containers::Array< int, Device, IndexType > ArrayContainer;
    using MeshFunctionPointer = Pointers::SharedPointer< MeshFunctionType >;
    using InterfaceMapPointer = Pointers::SharedPointer< InterfaceMapType >;      
    
    void initInterface( const MeshFunctionPointer& input,
            MeshFunctionPointer& output,
            InterfaceMapPointer& interfaceMap );
    
    template< typename MeshEntity >
    __cuda_callable__ void updateCell( MeshFunctionType& u,
            const MeshEntity& cell,
            const RealType velocity = 1.0);
    
    template< int sizeSArray >
    void updateBlocks( const InterfaceMapType& interfaceMap,
            const MeshFunctionType& aux,
            MeshFunctionType& helpFunc,
            ArrayContainer& BlockIterHost, int numThreadsPerBlock/*, Real **sArray*/ );
    
    void getNeighbours( ArrayContainer& BlockIterHost, int numBlockX, int numBlockY, int numBlockZ );
    
    template< int sizeSArray >
    __cuda_callable__ bool updateCell3D( volatile Real *sArray,
            int thri, int thrj, int thrk, const Real hx, const Real hy, const Real hz,
            const Real velocity = 1.0 );
};

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

template < int sizeSArray, typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > > ptr,
        const Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index >, 2, bool >& interfaceMap,
        const Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& aux,
        Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& helpFunc,
        TNL::Containers::Array< int, Devices::Cuda, Index > BlockIterDevice,
        Containers::StaticVector< 2, Index > vLower, Containers::StaticVector< 2, Index > vUpper, int k,int oddEvenBlock =0);

template< typename Real, typename Device, typename Index >
__global__ void DeepCopy( const Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& aux,
        Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& helpFunc );

template < typename Index >
__global__ void CudaParallelReduc( TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterDevice,
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > dBlock, int nBlocks );

template < typename Index >
__global__ void GetNeighbours( TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterDevice,
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterPom, int numBlockX, int numBlockY );

template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller( const Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& input, 
        Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& output,
        Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index >, 2, bool >& interfaceMap,
        Containers::StaticVector< 2, Index > vLower, Containers::StaticVector< 2, Index > vUpper );

template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller3d( const Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& input, 
        Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& output,
        Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index >, 3, bool >& interfaceMap );

template < int sizeSArray, typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > > ptr,
        const Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index >, 3, bool >& interfaceMap,
        const Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& aux,
        Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& helpFunc,
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterDevice );

template < typename Index >
__global__ void GetNeighbours3D( TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterDevice,
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterPom,
        int numBlockX, int numBlockY, int numBlockZ );
#endif

#include "tnlDirectEikonalMethodsBase_impl.h"
