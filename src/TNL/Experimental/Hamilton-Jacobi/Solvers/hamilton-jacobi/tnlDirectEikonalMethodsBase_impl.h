 /* 
 * File:   tnlDirectEikonalMethodsBase_impl.h
 * Author: oberhuber
 *
 * Created on July 14, 2016, 3:22 PM
 */

#pragma once

#include <limits>

#include <iostream>
#include "tnlFastSweepingMethod.h"

template< typename Real,
          typename Device,
          typename Index >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > >::
initInterface( const MeshFunctionPointer& _input,
               MeshFunctionPointer& _output,
               InterfaceMapPointer& _interfaceMap  )
{
    if( std::is_same< Device, Devices::Cuda >::value )
    {
#ifdef HAVE_CUDA
        const MeshType& mesh = _input->getMesh();
        
        const int cudaBlockSize( 16 );
        int numBlocksX = Devices::Cuda::getNumberOfBlocks( mesh.getDimensions().x(), cudaBlockSize );
        dim3 blockSize( cudaBlockSize );
        dim3 gridSize( numBlocksX );
        Devices::Cuda::synchronizeDevice();
        CudaInitCaller<<< gridSize, blockSize >>>( _input.template getData< Device >(),
                                                   _output.template modifyData< Device >(),
                                                   _interfaceMap.template modifyData< Device >() );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
#endif
    }
    if( std::is_same< Device, Devices::Host >::value )
    {
        const MeshType& mesh = _input->getMesh();
        typedef typename MeshType::Cell Cell;
        const MeshFunctionType& input = _input.getData();
        MeshFunctionType& output = _output.modifyData();
        InterfaceMapType& interfaceMap = _interfaceMap.modifyData();
        Cell cell( mesh );
        for( cell.getCoordinates().x() = 0;
            cell.getCoordinates().x() < mesh.getDimensions().x();
            cell.getCoordinates().x() ++ )
           {
               cell.refresh();
               output[ cell.getIndex() ] =
               input( cell ) >= 0 ? std::numeric_limits< RealType >::max() :
                                  -std::numeric_limits< RealType >::max();
               interfaceMap[ cell.getIndex() ] = false;
           }
        
        
        const RealType& h = mesh.getSpaceSteps().x();
        for( cell.getCoordinates().x() = 0;
             cell.getCoordinates().x() < mesh.getDimensions().x() - 1;
             cell.getCoordinates().x() ++ )
        {
           cell.refresh();
           const RealType& c = input( cell );      
           if( ! cell.isBoundaryEntity()  )
           {
              const auto& neighbors = cell.getNeighborEntities();
              Real pom = 0;
              //const IndexType& c = cell.getIndex();
              const IndexType e = neighbors.template getEntityIndex<  1 >();
              if( c * input[ e ] <= 0 )
              {
                pom = TNL::sign( c )*( h * c )/( c - input[ e ]);
                if( TNL::abs( output[ cell.getIndex() ] ) > TNL::abs( pom ) )
                    output[ cell.getIndex() ] = pom;

                pom = pom - TNL::sign( c )*h; //output[ e ] = (hx * c)/( c - input[ e ]) - hx;
                if( TNL::abs( output[ e ] ) > TNL::abs( pom ) )
                    output[ e ] = pom; 

                interfaceMap[ cell.getIndex() ] = true;
                interfaceMap[ e ] = true;
              }
           }
        }
    }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshEntity >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
            const MeshEntity& cell, 
            const RealType v )
{
    const auto& neighborEntities = cell.template getNeighborEntities< 1 >();
    const MeshType& mesh = cell.getMesh();
    const RealType& h = mesh.getSpaceSteps().x();
    const RealType value = u( cell );
    RealType a, tmp = std::numeric_limits< RealType >::max();

    if( cell.getCoordinates().x() == 0 )
       a = u[ neighborEntities.template getEntityIndex< 1 >() ];
    else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
       a = u[ neighborEntities.template getEntityIndex< -1 >() ];
    else
    {
       a = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< -1 >() ],
                           u[ neighborEntities.template getEntityIndex<  1 >() ] );
    }

    if( fabs( a ) == std::numeric_limits< RealType >::max() )
      return;
   
    tmp = a + TNL::sign( value ) * h/v;
    
    u[ cell.getIndex() ] = argAbsMin( value, tmp );
}

template< typename Real,
          typename Device,
          typename Index >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
initInterface( const MeshFunctionPointer& _input,
               MeshFunctionPointer& _output,
               InterfaceMapPointer& _interfaceMap )
{
            
    if( std::is_same< Device, Devices::Cuda >::value )
    {
#ifdef HAVE_CUDA
        const MeshType& mesh = _input->getMesh();
        
        const int cudaBlockSize( 16 );
        int numBlocksX = Devices::Cuda::getNumberOfBlocks( mesh.getDimensions().x(), cudaBlockSize );
        int numBlocksY = Devices::Cuda::getNumberOfBlocks( mesh.getDimensions().y(), cudaBlockSize );
        dim3 blockSize( cudaBlockSize, cudaBlockSize );
        dim3 gridSize( numBlocksX, numBlocksY );
        Devices::Cuda::synchronizeDevice();
        CudaInitCaller<<< gridSize, blockSize >>>( _input.template getData< Device >(),
                                                   _output.template modifyData< Device >(),
                                                   _interfaceMap.template modifyData< Device >() );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
#endif
    }
    if( std::is_same< Device, Devices::Host >::value )
    {
        MeshFunctionType input = _input.getData();
        
        /*double A[320][320];
        std::ifstream fileInit("/home/maty/Downloads/initData.txt");

        for (int i = 0; i < 320; i++)
            for (int j = 0; j < 320; j++)
                fileInit >> A[i][j];
        fileInit.close();
        for (int i = 0; i < 320; i++)
            for (int j = 0; j < 320; j++)
                input[i*320 + j] = A[i][j];*/
        
        
         MeshFunctionType& output = _output.modifyData();
         InterfaceMapType& interfaceMap = _interfaceMap.modifyData();
        const MeshType& mesh = input.getMesh();
        typedef typename MeshType::Cell Cell;
        Cell cell( mesh );
        for( cell.getCoordinates().y() = 0;
             cell.getCoordinates().y() < mesh.getDimensions().y();
             cell.getCoordinates().y() ++ )
            for( cell.getCoordinates().x() = 0;
                 cell.getCoordinates().x() < mesh.getDimensions().x();
                 cell.getCoordinates().x() ++ )
                {
                    cell.refresh();
                    output[ cell.getIndex() ] =
                    input( cell ) >= 0 ? std::numeric_limits< RealType >::max() :
                                       - std::numeric_limits< RealType >::max();
                    interfaceMap[ cell.getIndex() ] = false;
                }

       const RealType& hx = mesh.getSpaceSteps().x();
       const RealType& hy = mesh.getSpaceSteps().y();     
       for( cell.getCoordinates().y() = 0;
            cell.getCoordinates().y() < mesh.getDimensions().y();
            cell.getCoordinates().y() ++ )
          for( cell.getCoordinates().x() = 0;
               cell.getCoordinates().x() < mesh.getDimensions().x();
               cell.getCoordinates().x() ++ )
          {
             cell.refresh();
             const RealType& c = input( cell );
             if( ! cell.isBoundaryEntity()  )
             {
                auto neighbors = cell.getNeighborEntities();
                Real pom = 0;
                const IndexType e = neighbors.template getEntityIndex<  1,  0 >();
                const IndexType n = neighbors.template getEntityIndex<  0,  1 >();
                //Try init with exact data:
                /*if( c * input[ n ] <= 0 )
                {
                    output[ cell.getIndex() ] = c;
                    output[ n ] = input[ n ];
                    interfaceMap[ cell.getIndex() ] = true;
                    interfaceMap[ n ] = true;
                }   
                if( c * input[ e ] <= 0 )
                {   
                    output[ cell.getIndex() ] = c;
                    output[ e ] = input[ e ];
                    interfaceMap[ cell.getIndex() ] = true;
                    interfaceMap[ e ] = true;
                }*/
                if( c * input[ n ] <= 0 )
                {
                    /*if( c >= 0 )
                    {*/
                        pom = TNL::sign( c )*( hy * c )/( c - input[ n ]);
                        if( TNL::abs( output[ cell.getIndex() ] ) > TNL::abs( pom ) ) 
                            output[ cell.getIndex() ] = pom;
                        pom = pom - TNL::sign( c )*hy;
                        if( TNL::abs( output[ n ] ) > TNL::abs( pom ) )
                            output[ n ] = pom; //( hy * c )/( c - input[ n ]) - hy;
                    /*}else
                    {
                        pom = - ( hy * c )/( c - input[ n ]);
                        if( output[ cell.getIndex() ] < pom )
                            output[ cell.getIndex() ] = pom;
                        if( output[ n ] > hy + pom )
                            output[ n ] = hy + pom; //hy - ( hy * c )/( c - input[ n ]);
                    }*/
                    interfaceMap[ cell.getIndex() ] = true;
                    interfaceMap[ n ] = true;
                }
                if( c * input[ e ] <= 0 )
                {
                    /*if( c >= 0 )
                    {*/
                        pom = TNL::sign( c )*( hx * c )/( c - input[ e ]);
                        if( TNL::abs( output[ cell.getIndex() ] ) > TNL::abs( pom ) )
                            output[ cell.getIndex() ] = pom;

                        pom = pom - TNL::sign( c )*hx; //output[ e ] = (hx * c)/( c - input[ e ]) - hx;
                        if( TNL::abs( output[ e ] ) > TNL::abs( pom ) )
                            output[ e ] = pom; 
                    /*}else
                    {
                        pom = - (hx * c)/( c - input[ e ]);
                        if( output[ cell.getIndex() ] < pom )
                            output[ cell.getIndex() ] = pom;

                        pom = pom + hx; //output[ e ] = hx - (hx * c)/( c - input[ e ]);
                        if( output[ e ] > pom )
                            output[ e ] = pom;
                    }*/
                    interfaceMap[ cell.getIndex() ] = true;
                    interfaceMap[ e ] = true;
                }
             }
          }
      }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshEntity >
__cuda_callable__
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
            const MeshEntity& cell,   
            const RealType v)
{
   const auto& neighborEntities = cell.template getNeighborEntities< 2 >();
   const MeshType& mesh = cell.getMesh();
   const RealType& hx = mesh.getSpaceSteps().x();
   const RealType& hy = mesh.getSpaceSteps().y();
   const RealType value = u( cell );
   RealType a, b, tmp = std::numeric_limits< RealType >::max();
   
   if( cell.getCoordinates().x() == 0 )
      a = u[ neighborEntities.template getEntityIndex< 1,  0 >() ];
   else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
      a = u[ neighborEntities.template getEntityIndex< -1,  0 >() ];
   else
   {
      a = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< -1,  0 >() ],
                          u[ neighborEntities.template getEntityIndex<  1,  0 >() ] );
   }

   if( cell.getCoordinates().y() == 0 )
      b = u[ neighborEntities.template getEntityIndex< 0,  1 >()];
   else if( cell.getCoordinates().y() == mesh.getDimensions().y() - 1 )
      b = u[ neighborEntities.template getEntityIndex< 0,  -1 >() ];
   else
   {
      b = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< 0,  -1 >() ],
                          u[ neighborEntities.template getEntityIndex< 0,   1 >() ] );
   }
   if( fabs( a ) == std::numeric_limits< RealType >::max() && 
       fabs( b ) == std::numeric_limits< RealType >::max() )
      return;
   /*if( fabs( a ) == TypeInfo< Real >::getMaxValue() ||
       fabs( b ) == TypeInfo< Real >::getMaxValue() ||
       fabs( a - b ) >= TNL::sqrt( (hx * hx + hy * hy)/v ) )
   {
      tmp = 
        fabs( a ) >= fabs( b ) ? b + TNL::sign( value ) * hy :
                                 a + TNL::sign( value ) * hx;
   }*/
   /*if( fabs( a ) != TypeInfo< Real >::getMaxValue() &&
       fabs( b ) != TypeInfo< Real >::getMaxValue() &&
       fabs( a - b ) < TNL::sqrt( (hx * hx + hy * hy)/v ) )
   {
       tmp = ( hx * hx * b + hy * hy * a + 
            sign( value ) * hx * hy * TNL::sqrt( ( hx * hx + hy * hy )/v - 
            ( a - b ) * ( a - b ) ) )/( hx * hx + hy * hy );
       u[ cell.getIndex() ] =  tmp;
   }
   else
   {
       tmp = 
          fabs( a ) > fabs( b ) ? b + TNL::sign( value ) * hy/v :
                                   a + TNL::sign( value ) * hx/v;
       u[ cell.getIndex() ] = argAbsMin( value, tmp );
       //tmp = TypeInfo< RealType >::getMaxValue();
   }*/
    RealType pom[6] = { a, b, std::numeric_limits< RealType >::max(), (RealType)hx, (RealType)hy, 0.0 };
    sortMinims( pom );
    tmp = pom[ 0 ] + TNL::sign( value ) * pom[ 3 ]/v;
    
                                
    if( fabs( tmp ) < fabs( pom[ 1 ] ) ) 
        u[ cell.getIndex() ] = argAbsMin( value, tmp );
    else
    {
        tmp = ( pom[ 3 ] * pom[ 3 ] * pom[ 1 ] + pom[ 4 ] * pom[ 4 ] * pom[ 0 ] + 
            TNL::sign( value ) * pom[ 3 ] * pom[ 4 ] * TNL::sqrt( ( pom[ 3 ] * pom[ 3 ] +  pom[ 4 ] *  pom[ 4 ] )/( v * v ) - 
            ( pom[ 1 ] - pom[ 0 ] ) * ( pom[ 1 ] - pom[ 0 ] ) ) )/( pom[ 3 ] * pom[ 3 ] + pom[ 4 ] * pom[ 4 ] );
        u[ cell.getIndex() ] = argAbsMin( value, tmp );
    }
}

template< typename Real,
          typename Device,
          typename Index >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
initInterface( const MeshFunctionPointer& _input,
               MeshFunctionPointer& _output,
               InterfaceMapPointer& _interfaceMap  )
{
    if( std::is_same< Device, Devices::Cuda >::value )
    {
#ifdef HAVE_CUDA
        const MeshType& mesh = _input->getMesh();
        
        const int cudaBlockSize( 8 );
        int numBlocksX = Devices::Cuda::getNumberOfBlocks( mesh.getDimensions().x(), cudaBlockSize );
        int numBlocksY = Devices::Cuda::getNumberOfBlocks( mesh.getDimensions().y(), cudaBlockSize );
        int numBlocksZ = Devices::Cuda::getNumberOfBlocks( mesh.getDimensions().z(), cudaBlockSize );
        if( cudaBlockSize * cudaBlockSize * cudaBlockSize > 1024 || numBlocksX > 1024 || numBlocksY > 1024 || numBlocksZ > 64 )
            std::cout << "Invalid kernel call. Dimensions of grid are max: [1024,1024,64], and maximum threads per block are 1024!" << std::endl;
        dim3 blockSize( cudaBlockSize, cudaBlockSize, cudaBlockSize );
        dim3 gridSize( numBlocksX, numBlocksY, numBlocksZ );
        Devices::Cuda::synchronizeDevice();
        CudaInitCaller3d<<< gridSize, blockSize >>>( _input.template getData< Device >(),
                                                     _output.template modifyData< Device >(),
                                                     _interfaceMap.template modifyData< Device >() );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
#endif
    }
    if( std::is_same< Device, Devices::Host >::value )
    {
        const MeshFunctionType& input =  _input.getData();
        MeshFunctionType& output =  _output.modifyData();
        InterfaceMapType& interfaceMap =  _interfaceMap.modifyData();
        const MeshType& mesh = input.getMesh();
        typedef typename MeshType::Cell Cell;
        Cell cell( mesh );
        for( cell.getCoordinates().z() = 0;
             cell.getCoordinates().z() < mesh.getDimensions().z();
             cell.getCoordinates().z() ++ )
             for( cell.getCoordinates().y() = 0;
                  cell.getCoordinates().y() < mesh.getDimensions().y();
                  cell.getCoordinates().y() ++ )
                 for( cell.getCoordinates().x() = 0;
                      cell.getCoordinates().x() < mesh.getDimensions().x();
                      cell.getCoordinates().x() ++ )
                 {
                     cell.refresh();
                     output[ cell.getIndex() ] =
                     input( cell ) > 0 ? std::numeric_limits< RealType >::max() :
                                        - std::numeric_limits< RealType >::max();
                     interfaceMap[ cell.getIndex() ] = false;
                 }

        const RealType& hx = mesh.getSpaceSteps().x();
        const RealType& hy = mesh.getSpaceSteps().y();
        const RealType& hz = mesh.getSpaceSteps().z();
        for( cell.getCoordinates().z() = 0;
             cell.getCoordinates().z() < mesh.getDimensions().z();
             cell.getCoordinates().z() ++ )   
           for( cell.getCoordinates().y() = 0;
                cell.getCoordinates().y() < mesh.getDimensions().y();
                cell.getCoordinates().y() ++ )
              for( cell.getCoordinates().x() = 0;
                   cell.getCoordinates().x() < mesh.getDimensions().x();
                   cell.getCoordinates().x() ++ )
              {
                 cell.refresh();
                 const RealType& c = input( cell );
                 if( ! cell.isBoundaryEntity() )
                 {
                    auto neighbors = cell.getNeighborEntities();
                    Real pom = 0;
                    const IndexType e = neighbors.template getEntityIndex<  1,  0,  0 >();
                    const IndexType n = neighbors.template getEntityIndex<  0,  1,  0 >();
                    const IndexType t = neighbors.template getEntityIndex<  0,  0,  1 >();
                    //Try exact initiation
                    /*const IndexType w = neighbors.template getEntityIndex< -1,  0,  0 >();
                    const IndexType s = neighbors.template getEntityIndex<  0, -1,  0 >();
                    const IndexType b = neighbors.template getEntityIndex<  0,  0, -1 >();
                    if( c * input[ e ] <= 0 )
                    {
                       output[ cell.getIndex() ] = c;
                       output[ e ] = input[ e ];
                       interfaceMap[ e ] = true;   
                       interfaceMap[ cell.getIndex() ] = true;
                    }
                    else if( c * input[ n ] <= 0 )
                    {
                       output[ cell.getIndex() ] = c;
                       output[ n ] = input[ n ];
                       interfaceMap[ n ] = true;   
                       interfaceMap[ cell.getIndex() ] = true;
                    }
                    else if( c * input[ t ] <= 0 )
                    {
                       output[ cell.getIndex() ] = c;
                       output[ t ] = input[ t ];
                       interfaceMap[ t ] = true;   
                       interfaceMap[ cell.getIndex() ] = true;
                    }*/
                    if( c * input[ n ] <= 0 )
                    {
                        if( c >= 0 )
                        {
                        pom = ( hy * c )/( c - input[ n ]);
                        if( output[ cell.getIndex() ] > pom ) 
                            output[ cell.getIndex() ] = pom;

                        if ( output[ n ] < pom - hy)
                             output[ n ] = pom - hy; // ( hy * c )/( c - input[ n ]) - hy;

                        }else
                        {
                          pom = - ( hy * c )/( c - input[ n ]);
                          if( output[ cell.getIndex() ] < pom )
                              output[ cell.getIndex() ] = pom;
                          if( output[ n ] > hy + pom )
                              output[ n ] = hy + pom; //hy - ( hy * c )/( c - input[ n ]);

                        }
                    interfaceMap[ cell.getIndex() ] = true;
                    interfaceMap[ n ] = true;
                    }
                    if( c * input[ e ] <= 0 )
                    {
                        if( c >= 0 )
                        {
                            pom = ( hx * c )/( c - input[ e ]);
                            if( output[ cell.getIndex() ] > pom )
                                output[ cell.getIndex() ] = pom;

                            pom = pom - hx; //output[ e ] = (hx * c)/( c - input[ e ]) - hx;
                            if( output[ e ] < pom )
                                output[ e ] = pom;      

                        }else
                        {
                            pom = - (hx * c)/( c - input[ e ]);
                            if( output[ cell.getIndex() ] < pom )
                                output[ cell.getIndex() ] = pom;

                            pom = pom + hx; //output[ e ] = hx - (hx * c)/( c - input[ e ]);
                            if( output[ e ] > pom )
                                output[ e ] = pom;
                        }
                    interfaceMap[ cell.getIndex() ] = true;
                    interfaceMap[ e ] = true;
                    }
                    if( c * input[ t ] <= 0 )
                    {
                        if( c >= 0 )
                        {
                            pom = ( hz * c )/( c - input[ t ]);
                            if( output[ cell.getIndex() ] > pom )
                                output[ cell.getIndex() ] = pom;

                            pom = pom - hz; //output[ e ] = (hx * c)/( c - input[ e ]) - hx;
                            if( output[ t ] < pom )
                                output[ t ] = pom; 

                        }else
                        {
                            pom = - (hz * c)/( c - input[ t ]);
                            if( output[ cell.getIndex() ] < pom )
                                output[ cell.getIndex() ] = pom;

                            pom = pom + hz; //output[ e ] = hx - (hx * c)/( c - input[ e ]);
                            if( output[ t ] > pom )
                                output[ t ] = pom;

                        }
                    interfaceMap[ cell.getIndex() ] = true;
                    interfaceMap[ t ] = true;
                    }    
                 }
                 /*output[ cell.getIndex() ] =
                    c > 0 ? TypeInfo< RealType >::getMaxValue() :
                           -TypeInfo< RealType >::getMaxValue();
                 interfaceMap[ cell.getIndex() ] = false;*/ //is on line 245
              }
    }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshEntity >
__cuda_callable__
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
            const MeshEntity& cell, 
            const RealType v )
{
   const auto& neighborEntities = cell.template getNeighborEntities< 3 >();
   const MeshType& mesh = cell.getMesh();
  
   const RealType& hx = mesh.getSpaceSteps().x();
   const RealType& hy = mesh.getSpaceSteps().y();
   const RealType& hz = mesh.getSpaceSteps().z();
   const RealType value = u( cell );
   //std::cout << value << std::endl;
   RealType a, b, c, tmp = std::numeric_limits< RealType >::max();
   
   
   if( cell.getCoordinates().x() == 0 )
      a = u[ neighborEntities.template getEntityIndex< 1, 0, 0 >() ];
   else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
      a = u[ neighborEntities.template getEntityIndex< -1, 0, 0 >() ];
   else
   {
      a = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< -1, 0, 0 >() ],
                        u[ neighborEntities.template getEntityIndex< 1, 0, 0 >() ] );
   }
   if( cell.getCoordinates().y() == 0 )
      b = u[ neighborEntities.template getEntityIndex< 0, 1, 0 >() ];
   else if( cell.getCoordinates().y() == mesh.getDimensions().y() - 1 )
      b = u[ neighborEntities.template getEntityIndex< 0, -1, 0 >() ];
   else
   {
      b = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< 0, -1, 0 >() ],
                        u[ neighborEntities.template getEntityIndex< 0, 1, 0 >() ] );
   }if( cell.getCoordinates().z() == 0 )
      c = u[ neighborEntities.template getEntityIndex< 0, 0, 1 >() ];
   else if( cell.getCoordinates().z() == mesh.getDimensions().z() - 1 )
      c = u[ neighborEntities.template getEntityIndex< 0, 0, -1 >() ];
   else
   {
      c = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< 0, 0, -1 >() ],
                         u[ neighborEntities.template getEntityIndex< 0, 0, 1 >() ] );
   }
   if( fabs( a ) == std::numeric_limits< RealType >::max() && 
       fabs( b ) == std::numeric_limits< RealType >::max() &&
       fabs( c ) == std::numeric_limits< RealType >::max() )
      return;
   
   
       /*if( fabs( a ) != TypeInfo< Real >::getMaxValue() &&
           fabs( b ) != TypeInfo< Real >::getMaxValue() &&
           fabs( a - b ) >= TNL::sqrt( (hx * hx + hy * hy)/v ) )
       {
           tmp = ( hx * hx * a + hy * hy * b + 
                sign( value ) * hx * hy * sqrt( ( hx * hx + hy * hy )/v - 
                ( a - b ) * ( a - b ) ) )/( hx * hx + hy * hy );
       }
       if( fabs( a ) != TypeInfo< Real >::getMaxValue() &&
           fabs( c ) != TypeInfo< Real >::getMaxValue() &&
           fabs( a - c ) >= TNL::sqrt( (hx * hx + hz * hz)/v ) )
       {
           tmp = ( hx * hx * a + hz * hz * c + 
                sign( value ) * hx * hz * sqrt( ( hx * hx + hz * hz )/v - 
                ( a - c ) * ( a - c ) ) )/( hx * hx + hz * hz );
       }
       if( fabs( b ) != TypeInfo< Real >::getMaxValue() &&
           fabs( c ) != TypeInfo< Real >::getMaxValue() &&
           fabs( b - c ) >= TNL::sqrt( (hy * hy + hz * hz)/v ) )
       {
           tmp = ( hy * hy * b + hz * hz * c + 
                sign( value ) * hy * hz * sqrt( ( hy * hy + hz * hz )/v - 
                ( b - c ) * ( b - c ) ) )/( hy * hy + hz * hz );
       }*/
    RealType pom[6] = { a, b, c, (RealType)hx, (RealType)hy, (RealType)hz};
    sortMinims( pom );   
    tmp = pom[ 0 ] + TNL::sign( value ) * pom[ 3 ];
    if( fabs( tmp ) < fabs( pom[ 1 ] ) )
    {
        u[ cell.getIndex() ] = argAbsMin( value, tmp ); 
    }
    else
    {
        tmp = ( pom[ 3 ] * pom[ 3 ] * pom[ 1 ] + pom[ 4 ] * pom[ 4 ] * pom[ 0 ] + 
            TNL::sign( value ) * pom[ 3 ] * pom[ 4 ] * TNL::sqrt( ( pom[ 3 ] * pom[ 3 ] +  pom[ 4 ] *  pom[ 4 ] )/( v * v ) - 
            ( pom[ 1 ] - pom[ 0 ] ) * ( pom[ 1 ] - pom[ 0 ] ) ) )/( pom[ 3 ] * pom[ 3 ] + pom[ 4 ] * pom[ 4 ] );
        if( fabs( tmp ) < fabs( pom[ 2 ]) ) 
        {
            u[ cell.getIndex() ] = argAbsMin( value, tmp );
        }
        else
        {
            tmp = ( hy * hy * hz * hz * a + hx * hx * hz * hz * b + hx * hx * hy * hy * c +
                TNL::sign( value ) * hx * hy * hz * TNL::sqrt( ( hx * hx * hz * hz + hy * hy * hz * hz + hx * hx * hy * hy)/( v * v ) - 
                hz * hz * ( a - b ) * ( a - b ) - hy * hy * ( a - c ) * ( a - c ) -
                hx * hx * ( b - c ) * ( b - c ) ) )/( hx * hx * hy * hy + hy * hy * hz * hz + hz * hz * hx *hx );
            u[ cell.getIndex() ] = argAbsMin( value, tmp );
        }
    }
}

template < typename T1, typename T2 >
T1 meet2DCondition( T1 a, T1 b, const T2 ha, const T2 hb, const T1 value, double v)
{
   T1 tmp;
   if( fabs( a ) != std::numeric_limits< T1 >::max &&
       fabs( b ) != std::numeric_limits< T1 >::max &&
       fabs( a - b ) < ha/v )//TNL::sqrt( (ha * ha + hb * hb)/2 )/v )
   {
      tmp = ( ha * ha * b + hb * hb * a + 
            TNL::sign( value ) * ha * hb * TNL::sqrt( ( ha * ha + hb * hb )/( v * v ) - 
            ( a - b ) * ( a - b ) ) )/( ha * ha + hb * hb );
   }
   else
   {
       tmp = std::numeric_limits< T1 >::max;
   }
   
   return tmp;
}

template < typename T1 >
__cuda_callable__ void sortMinims( T1 pom[] )
{
    T1 tmp[6] = {0.0,0.0,0.0,0.0,0.0,0.0}; 
    if( fabs(pom[0]) <= fabs(pom[1]) && fabs(pom[1]) <= fabs(pom[2])){
        tmp[0] = pom[0]; tmp[1] = pom[1]; tmp[2] = pom[2];
        tmp[3] = pom[3]; tmp[4] = pom[4]; tmp[5] = pom[5];
        
    }
    else if( fabs(pom[0]) <= fabs(pom[2]) && fabs(pom[2]) <= fabs(pom[1]) ){
        tmp[0] = pom[0]; tmp[1] = pom[2]; tmp[2] = pom[1];
        tmp[3] = pom[3]; tmp[4] = pom[5]; tmp[5] = pom[4];
    }
    else if( fabs(pom[1]) <= fabs(pom[0]) && fabs(pom[0]) <= fabs(pom[2]) ){
        tmp[0] = pom[1]; tmp[1] = pom[0]; tmp[2] = pom[2];
        tmp[3] = pom[4]; tmp[4] = pom[3]; tmp[5] = pom[5];
    }
    else if( fabs(pom[1]) <= fabs(pom[2]) && fabs(pom[2]) <= fabs(pom[0]) ){
        tmp[0] = pom[1]; tmp[1] = pom[2]; tmp[2] = pom[0];
        tmp[3] = pom[4]; tmp[4] = pom[5]; tmp[5] = pom[3];
    }
    else if( fabs(pom[2]) <= fabs(pom[0]) && fabs(pom[0]) <= fabs(pom[1]) ){
        tmp[0] = pom[2]; tmp[1] = pom[0]; tmp[2] = pom[1];
        tmp[3] = pom[5]; tmp[4] = pom[3]; tmp[5] = pom[4];
    }
    else if( fabs(pom[2]) <= fabs(pom[1]) && fabs(pom[1]) <= fabs(pom[0]) ){
        tmp[0] = pom[2]; tmp[1] = pom[1]; tmp[2] = pom[0];
        tmp[3] = pom[5]; tmp[4] = pom[4]; tmp[5] = pom[3];
    }
    
    for( int i = 0; i < 6; i++ )
    {
        pom[ i ] = tmp[ i ];
    }   
}



#ifdef HAVE_CUDA
template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller( const Functions::MeshFunction< Meshes::Grid< 1, Real, Device, Index > >& input, 
                                Functions::MeshFunction< Meshes::Grid< 1, Real, Device, Index > >& output,
                                Functions::MeshFunction< Meshes::Grid< 1, Real, Device, Index >, 1, bool >& interfaceMap  )
{
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    const Meshes::Grid< 1, Real, Device, Index >& mesh = input.template getMesh< Devices::Cuda >();
    
    if( i < mesh.getDimensions().x()  )
    {
        typedef typename Meshes::Grid< 1, Real, Device, Index >::Cell Cell;
        Cell cell( mesh );
        cell.getCoordinates().x() = i;
        cell.refresh();
        const Index cind = cell.getIndex();


        output[ cind ] =
               input( cell ) >= 0 ? std::numeric_limits< Real >::max() :
                                    - std::numeric_limits< Real >::max();
        interfaceMap[ cind ] = false; 

        const Real& h = mesh.getSpaceSteps().x();
        cell.refresh();
        const Real& c = input( cell );
        if( ! cell.isBoundaryEntity()  )
        {
           auto neighbors = cell.getNeighborEntities();
           Real pom = 0;
           const Index e = neighbors.template getEntityIndex< 1 >();
           const Index w = neighbors.template getEntityIndex< -1 >();
           if( c * input[ e ] <= 0 )
           {
               pom = TNL::sign( c )*( h * c )/( c - input[ e ]);
               if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) )
                   output[ cind ] = pom;                       

               interfaceMap[ cind ] = true;
           }
           if( c * input[ w ] <= 0 )
           {
               pom = TNL::sign( c )*( h * c )/( c - input[ w ]);
               if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
                   output[ cind ] = pom;

               interfaceMap[ cind ] = true;
           }
        }
    }
           
}
template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller( const Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& input, 
                                Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index > >& output,
                                Functions::MeshFunction< Meshes::Grid< 2, Real, Device, Index >, 2, bool >& interfaceMap ) 
{
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    const Meshes::Grid< 2, Real, Device, Index >& mesh = input.template getMesh< Devices::Cuda >();
    
    if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() )
    {
        typedef typename Meshes::Grid< 2, Real, Device, Index >::Cell Cell;
        Cell cell( mesh );
        cell.getCoordinates().x() = i; cell.getCoordinates().y() = j;
        cell.refresh();
        const Index cind = cell.getIndex();


        output[ cind ] =
               input( cell ) >= 0 ? std::numeric_limits< Real >::max() :
                                    - std::numeric_limits< Real >::max();
        interfaceMap[ cind ] = false; 

        const Real& hx = mesh.getSpaceSteps().x();
        const Real& hy = mesh.getSpaceSteps().y();
        cell.refresh();
        const Real& c = input( cell );
        if( ! cell.isBoundaryEntity()  )
        {
           auto neighbors = cell.getNeighborEntities();
           Real pom = 0;
           const Index e = neighbors.template getEntityIndex<  1,  0 >();
           const Index w = neighbors.template getEntityIndex<  -1,  0 >();
           const Index n = neighbors.template getEntityIndex<  0,  1 >();
           const Index s = neighbors.template getEntityIndex<  0,  -1 >();
           
           if( c * input[ n ] <= 0 )
           {
               pom = TNL::sign( c )*( hy * c )/( c - input[ n ]);
               if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
                   output[ cind ] = pom;

               interfaceMap[ cell.getIndex() ] = true;
           }
           if( c * input[ e ] <= 0 )
           {
               pom = TNL::sign( c )*( hx * c )/( c - input[ e ]);
               if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) )
                   output[ cind ] = pom;                       

               interfaceMap[ cind ] = true;
           }
           if( c * input[ w ] <= 0 )
           {
               pom = TNL::sign( c )*( hx * c )/( c - input[ w ]);
               if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
                   output[ cind ] = pom;

               interfaceMap[ cind ] = true;
           }
           if( c * input[ s ] <= 0 )
           {
               pom = TNL::sign( c )*( hy * c )/( c - input[ s ]);
               if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
                   output[ cind ] = pom;

               interfaceMap[ cind ] = true;
           }
        }
    }
}

template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller3d( const Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& input, 
                                  Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index > >& output,
                                  Functions::MeshFunction< Meshes::Grid< 3, Real, Device, Index >, 3, bool >& interfaceMap )
{
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    int k = blockDim.z*blockIdx.z + threadIdx.z;
    const Meshes::Grid< 3, Real, Device, Index >& mesh = input.template getMesh< Devices::Cuda >();
    
    if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() && k < mesh.getDimensions().z() )
    {
        typedef typename Meshes::Grid< 3, Real, Device, Index >::Cell Cell;
        Cell cell( mesh );
        cell.getCoordinates().x() = i; cell.getCoordinates().y() = j; cell.getCoordinates().z() = k;
        cell.refresh();
        const Index cind = cell.getIndex();


        output[ cind ] =
               input( cell ) >= 0 ? std::numeric_limits< Real >::max() :
                                    - std::numeric_limits< Real >::max();
        interfaceMap[ cind ] = false; 
        cell.refresh();

        const Real& hx = mesh.getSpaceSteps().x();
        const Real& hy = mesh.getSpaceSteps().y();
        const Real& hz = mesh.getSpaceSteps().z();
        const Real& c = input( cell );
        if( ! cell.isBoundaryEntity()  )
        {
           auto neighbors = cell.getNeighborEntities();
           Real pom = 0;
           const Index e = neighbors.template getEntityIndex<  1, 0, 0 >();
           const Index w = neighbors.template getEntityIndex<  -1, 0, 0 >();
           const Index n = neighbors.template getEntityIndex<  0, 1, 0 >();
           const Index s = neighbors.template getEntityIndex<  0, -1, 0 >();
           const Index t = neighbors.template getEntityIndex<  0, 0, 1 >();
           const Index b = neighbors.template getEntityIndex<  0, 0, -1 >();
           
           if( c * input[ n ] <= 0 )
           {
               pom = TNL::sign( c )*( hy * c )/( c - input[ n ]);
               if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
                   output[ cind ] = pom;

               interfaceMap[ cind ] = true;
           }
           if( c * input[ e ] <= 0 )
           {
               pom = TNL::sign( c )*( hx * c )/( c - input[ e ]);
               if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) )
                   output[ cind ] = pom;                       

               interfaceMap[ cind ] = true;
           }
           if( c * input[ w ] <= 0 )
           {
               pom = TNL::sign( c )*( hx * c )/( c - input[ w ]);
               if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
                   output[ cind ] = pom;

               interfaceMap[ cind ] = true;
           }
           if( c * input[ s ] <= 0 )
           {
               pom = TNL::sign( c )*( hy * c )/( c - input[ s ]);
               if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
                   output[ cind ] = pom;

               interfaceMap[ cind ] = true;
           }
           if( c * input[ b ] <= 0 )
           {
               pom = TNL::sign( c )*( hz * c )/( c - input[ b ]);
               if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
                   output[ cind ] = pom;

               interfaceMap[ cind ] = true;
           }
           if( c * input[ t ] <= 0 )
           {
               pom = TNL::sign( c )*( hz * c )/( c - input[ t ]);
               if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
                   output[ cind ] = pom;

               interfaceMap[ cind ] = true;
           }
        }
    }
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
updateCell( volatile Real sArray[18][18], int thri, int thrj, const Real hx, const Real hy,
            const Real v )
{
   const RealType value = sArray[ thrj ][ thri ];
   RealType a, b, tmp = std::numeric_limits< RealType >::max();
      
   b = TNL::argAbsMin( sArray[ thrj+1 ][ thri ],
                        sArray[ thrj-1 ][ thri ] );
    
   a = TNL::argAbsMin( sArray[ thrj ][ thri+1 ],
                        sArray[ thrj ][ thri-1 ] );

    if( fabs( a ) == std::numeric_limits< RealType >::max() && 
        fabs( b ) == std::numeric_limits< RealType >::max() )
       return false;
   
    RealType pom[6] = { a, b, std::numeric_limits< RealType >::max(), (RealType)hx, (RealType)hy, 0.0 };
    sortMinims( pom );
    tmp = pom[ 0 ] + TNL::sign( value ) * pom[ 3 ]/v;
    
                                
    if( fabs( tmp ) < fabs( pom[ 1 ] ) ) 
    {
        sArray[ thrj ][ thri ] = argAbsMin( value, tmp );
        tmp = value - sArray[ thrj ][ thri ];
        if ( fabs( tmp ) >  0.001*hx )
            return true;
        else
            return false;
    }
    else
    {
        tmp = ( pom[ 3 ] * pom[ 3 ] * pom[ 1 ] + pom[ 4 ] * pom[ 4 ] * pom[ 0 ] + 
            TNL::sign( value ) * pom[ 3 ] * pom[ 4 ] * TNL::sqrt( ( pom[ 3 ] * pom[ 3 ] +  pom[ 4 ] *  pom[ 4 ] )/( v * v ) - 
            ( pom[ 1 ] - pom[ 0 ] ) * ( pom[ 1 ] - pom[ 0 ] ) ) )/( pom[ 3 ] * pom[ 3 ] + pom[ 4 ] * pom[ 4 ] );
        sArray[ thrj ][ thri ] = argAbsMin( value, tmp );
        tmp = value - sArray[ thrj ][ thri ];
        if ( fabs( tmp ) > 0.001*hx )
            return true;
        else
            return false;
    }
    
    return false;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool
tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > >::
updateCell( volatile Real sArray[18], int thri, const Real h, const Real v )
{
   const RealType value = sArray[ thri ];
   RealType a, tmp = std::numeric_limits< RealType >::max();
      
   a = TNL::argAbsMin( sArray[ thri+1 ],
                       sArray[ thri-1 ] );

    if( fabs( a ) == std::numeric_limits< RealType >::max() )
       return false;
   
    tmp = a + TNL::sign( value ) * h/v;
    
                                
    sArray[ thri ] = argAbsMin( value, tmp );
    
    tmp = value - sArray[ thri ];
    if ( fabs( tmp ) >  0.001*h )
        return true;
    else
        return false;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ 
bool 
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
updateCell( volatile Real sArray[10][10][10], int thri, int thrj, int thrk,
        const Real hx, const Real hy, const Real hz, const Real v )
{
   const RealType value = sArray[thrk][thrj][thri];
   //std::cout << value << std::endl;
   RealType a, b, c, tmp = std::numeric_limits< RealType >::max();
   
   c = TNL::argAbsMin( sArray[ thrk+1 ][ thrj ][ thri ],
                        sArray[ thrk-1 ][ thrj ][ thri ] );
    
   b = TNL::argAbsMin( sArray[ thrk ][ thrj+1 ][ thri ],
                        sArray[ thrk ][ thrj-1 ][ thri ] );
   
   a = TNL::argAbsMin( sArray[ thrk ][ thrj ][ thri+1 ],
                        sArray[ thrk ][ thrj ][ thri-1 ] );
   
   
   if( fabs( a ) == std::numeric_limits< RealType >::max() && 
       fabs( b ) == std::numeric_limits< RealType >::max() &&
       fabs( c ) == std::numeric_limits< RealType >::max() )
      return false;
   
    RealType pom[6] = { a, b, c, (RealType)hx, (RealType)hy, (RealType)hz};
    
    sortMinims( pom );
    
    tmp = pom[ 0 ] + TNL::sign( value ) * pom[ 3 ];
    if( fabs( tmp ) < fabs( pom[ 1 ] ) ) 
    {
        sArray[ thrk ][ thrj ][ thri ] = argAbsMin( value, tmp );
        tmp = value - sArray[ thrk ][ thrj ][ thri ];
        if ( fabs( tmp ) >  0.001*hx )
            return true;
        else
            return false;
    }
    else
    {
        tmp = ( pom[ 3 ] * pom[ 3 ] * pom[ 1 ] + pom[ 4 ] * pom[ 4 ] * pom[ 0 ] + 
            TNL::sign( value ) * pom[ 3 ] * pom[ 4 ] * TNL::sqrt( ( pom[ 3 ] * pom[ 3 ] +  pom[ 4 ] *  pom[ 4 ] )/( v * v ) - 
            ( pom[ 1 ] - pom[ 0 ] ) * ( pom[ 1 ] - pom[ 0 ] ) ) )/( pom[ 3 ] * pom[ 3 ] + pom[ 4 ] * pom[ 4 ] );
        if( fabs( tmp ) < fabs( pom[ 2 ]) ) 
        {
            sArray[ thrk ][ thrj ][ thri ] = argAbsMin( value, tmp );
            tmp = value - sArray[ thrk ][ thrj ][ thri ];
            if ( fabs( tmp ) > 0.001*hx )
                return true;
            else
                return false;
        }
        else
        {
            tmp = ( hy * hy * hz * hz * a + hx * hx * hz * hz * b + hx * hx * hy * hy * c +
                TNL::sign( value ) * hx * hy * hz * TNL::sqrt( ( hx * hx * hz * hz + hy * hy * hz * hz + hx * hx * hy * hy)/( v * v ) - 
                hz * hz * ( a - b ) * ( a - b ) - hy * hy * ( a - c ) * ( a - c ) -
                hx * hx * ( b - c ) * ( b - c ) ) )/( hx * hx * hy * hy + hy * hy * hz * hz + hz * hz * hx *hx );
            sArray[ thrk ][ thrj ][ thri ] = argAbsMin( value, tmp );
            tmp = value - sArray[ thrk ][ thrj ][ thri ];
            if ( fabs( tmp ) > 0.001*hx )
                return true;
            else
                return false;
        }
    }
    
    return false;
}
#endif
