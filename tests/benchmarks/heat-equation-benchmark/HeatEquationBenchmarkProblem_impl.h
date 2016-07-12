#ifndef HeatEquationBenchmarkPROBLEM_IMPL_H_
#define HeatEquationBenchmarkPROBLEM_IMPL_H_

#include <core/mfilename.h>
#include <matrices/tnlMatrixSetter.h>
#include <solvers/pde/tnlExplicitUpdater.h>
#include <solvers/pde/tnlLinearSystemAssembler.h>
#include <solvers/pde/tnlBackwardTimeDiscretisation.h>
#include "TestGridEntity.h"

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
tnlString
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getTypeStatic()
{
   return tnlString( "HeatEquationBenchmarkProblem< " ) + Mesh :: getTypeStatic() + " >";
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
HeatEquationBenchmarkProblem()
: cudaMesh( 0 ),
  cudaBoundaryConditions( 0 ),
  cudaRightHandSide( 0 ),
  cudaDifferentialOperator( 0 )
{
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
tnlString
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getPrologHeader() const
{
   return tnlString( "Heat Equation Benchmark" );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
writeProlog( tnlLogger& logger, const tnlParameterContainer& parameters ) const
{
   /****
    * Add data you want to have in the computation report (log) as follows:
    * logger.writeParameter< double >( "Parameter description", parameter );
    */
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setup( const tnlParameterContainer& parameters )
{
   if( ! this->boundaryConditionPointer->setup( parameters, "boundary-conditions-" ) ||
       ! this->rightHandSidePointer->setup( parameters, "right-hand-side-" ) )
      return false;
   this->cudaKernelType = parameters.getParameter< tnlString >( "cuda-kernel-type" );

   if( std::is_same< DeviceType, tnlCuda >::value )
   {
      this->cudaBoundaryConditions = tnlCuda::passToDevice( *this->boundaryConditionPointer );
      this->cudaRightHandSide = tnlCuda::passToDevice( *this->rightHandSidePointer );
      this->cudaDifferentialOperator = tnlCuda::passToDevice( *this->differentialOperatorPointer );
   }
   
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
typename HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::IndexType
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getDofs( const MeshPointer& meshPointer ) const
{
   /****
    * Return number of  DOFs (degrees of freedom) i.e. number
    * of unknowns to be resolved by the main solver.
    */
   return meshPointer->template getEntitiesCount< typename MeshType::Cell >();
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
bindDofs( const MeshPointer& meshPointer,
          DofVectorPointer& dofsPointer )
{
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setInitialCondition( const tnlParameterContainer& parameters,
                     const MeshPointer& meshPointer,
                     DofVectorPointer& dofsPointer,
                     MeshDependentDataType& meshDependentData )
{
   const tnlString& initialConditionFile = parameters.getParameter< tnlString >( "initial-condition" );
   tnlMeshFunction< Mesh > u( meshPointer, dofsPointer );
   if( ! u.boundLoad( initialConditionFile ) )
   {
      cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << endl;
      return false;
   }
   return true; 
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
   template< typename Matrix >
bool
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setupLinearSystem( const MeshType& mesh,
                   Matrix& matrix )
{
   const IndexType dofs = this->getDofs( mesh );
   typedef typename Matrix::CompressedRowsLengthsVector CompressedRowsLengthsVectorType;
   CompressedRowsLengthsVectorType rowLengths;
   if( ! rowLengths.setSize( dofs ) )
      return false;
   tnlMatrixSetter< MeshType, DifferentialOperator, BoundaryCondition, CompressedRowsLengthsVectorType > matrixSetter;
   matrixSetter.template getCompressedRowsLengths< typename Mesh::Cell >( mesh,
                                                                          differentialOperatorPointer,
                                                                          boundaryConditionPointer,
                                                                          rowLengths );
   matrix.setDimensions( dofs, dofs );
   if( ! matrix.setCompressedRowsLengths( rowLengths ) )
      return false;
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
makeSnapshot( const RealType& time,
              const IndexType& step,
              const MeshPointer& meshPointer,
              DofVectorPointer& dofsPointer,
              MeshDependentDataType& meshDependentData )
{
   cout << endl << "Writing output at time " << time << " step " << step << "." << endl;
   this->bindDofs( meshPointer, dofsPointer );
   MeshFunctionType u;
   u.bind( meshPointer, *dofsPointer );
   tnlString fileName;
   FileNameBaseNumberEnding( "u-", step, 5, ".tnl", fileName );
   if( ! u.save( fileName ) )
      return false;
   return true;
}

#ifdef HAVE_CUDA

template< typename Real, typename Index >
__global__ void boundaryConditionsKernel( Real* u,
                                          Real* fu,
                                          const Index gridXSize, const Index gridYSize )
{
   const Index i = ( blockIdx.x ) * blockDim.x + threadIdx.x;
   const Index j = ( blockIdx.y ) * blockDim.y + threadIdx.y;
   if( i == 0 && j < gridYSize )
   {
      fu[ j * gridXSize ] = 0.0;
      u[ j * gridXSize ] = 0.0; //u[ j * gridXSize + 1 ];
   }
   if( i == gridXSize - 1 && j < gridYSize )
   {
      fu[ j * gridXSize + gridYSize - 1 ] = 0.0;
      u[ j * gridXSize + gridYSize - 1 ] = 0.0; //u[ j * gridXSize + gridXSize - 1 ];      
   }
   if( j == 0 && i > 0 && i < gridXSize - 1 )
   {
      fu[ i ] = 0.0; //u[ j * gridXSize + 1 ];
      u[ i ] = 0.0; //u[ j * gridXSize + 1 ];
   }
   if( j == gridYSize -1  && i > 0 && i < gridXSize - 1 )
   {
      fu[ j * gridXSize + i ] = 0.0; //u[ j * gridXSize + gridXSize - 1 ];      
      u[ j * gridXSize + i ] = 0.0; //u[ j * gridXSize + gridXSize - 1 ];      
   }         
}


template< typename Real, typename Index >
__global__ void heatEquationKernel( const Real* u, 
                                    Real* fu,
                                    const Real tau,
                                    const Real hx_inv,
                                    const Real hy_inv,
                                    const Index gridXSize,
                                    const Index gridYSize )
{
   const Index i = blockIdx.x * blockDim.x + threadIdx.x;
   const Index j = blockIdx.y * blockDim.y + threadIdx.y;
   if( i > 0 && i < gridXSize - 1 &&
       j > 0 && j < gridYSize - 1 )
   {
      const Index c = j * gridXSize + i;
      fu[ c ] = ( u[ c - 1 ] - 2.0 * u[ c ] + u[ c + 1 ] ) * hx_inv +
                ( u[ c - gridXSize ] - 2.0 * u[ c ] + u[ c + gridXSize ] ) * hy_inv;
   }
}

template< typename GridType,
          typename GridEntity,
          typename BoundaryConditions,
          typename MeshFunction >
__global__ void 
boundaryConditionsTemplatedCompact( const GridType* grid,
                                    const BoundaryConditions* boundaryConditions,
                                    MeshFunction* u,
                                    const typename GridType::RealType time,
                                    const typename GridEntity::CoordinatesType begin,
                                    const typename GridEntity::CoordinatesType end,
                                    const typename GridEntity::EntityOrientationType entityOrientation,
                                    const typename GridEntity::EntityBasisType entityBasis,   
                                    const typename GridType::IndexType gridXIdx,
                                    const typename GridType::IndexType gridYIdx )
{
   typename GridType::CoordinatesType coordinates;

   coordinates.x() = begin.x() + ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = begin.y() + ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;  
   
   GridEntity entity( *grid, coordinates, entityOrientation, entityBasis );

   if( entity.getCoordinates().x() < end.x() &&
       entity.getCoordinates().y() < end.y() )
   {
      entity.refresh();
      if( entity.isBoundaryEntity() )
      {
         ( *u )( entity ) = ( *boundaryConditions )( *u, entity, time );
      }
   }
   
   /*typedef typename GridEntity::IndexType IndexType;
   typedef typename GridEntity::RealType RealType;
   RealType* _u = &( *u )[ 0 ];
   const IndexType tidX = begin.x() + ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const IndexType tidY = begin.y() + ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   if( tidX == 0 || tidX == end.x() - 1 || tidY == 0 || tidY == end.y() - 1 )      
   {
      _u[ tidY * grid->getDimensions().x() + tidX ] = 0.0;
   }*/
}

template< typename EntityType, int Dimensions >
struct EntityPointer : public EntityPointer< EntityType, Dimensions - 1 >
{
   __device__ EntityPointer( const EntityType* ptr )
      : EntityPointer< EntityType, Dimensions - 1 >( ptr ), pointer( ptr )
   {      
   }
   
   const EntityType* pointer;
};

template< typename EntityType >
struct EntityPointer< EntityType, 0 >
{
   __device__ inline EntityPointer( const EntityType* ptr )
   :pointer( ptr )
   {      
   }

   
   const EntityType* pointer;
};

template< typename GridType >
struct TestEntity
{
   typedef typename GridType::Cell::CoordinatesType CoordinatesType;
   
   __device__ inline TestEntity( const GridType& grid,
               const typename GridType::Cell::CoordinatesType& coordinates,
               const typename GridType::Cell::EntityOrientationType& orientation,
               const typename GridType::Cell::EntityBasisType& basis )
   : grid( grid ),
      coordinates( coordinates ),
      orientation( orientation ),
      basis( basis ),
      entityIndex( 0 ),
      ptr( &grid )
   {      
   };
  
   const GridType& grid;
   
   EntityPointer< GridType, 2 > ptr; 
   //TestEntity< GridType > *entity1, *entity2, *entity3;
   
   typename GridType::IndexType entityIndex;      
   
   const typename GridType::Cell::CoordinatesType coordinates;
   const typename GridType::Cell::EntityOrientationType orientation;
   const typename GridType::Cell::EntityBasisType basis;
   
};

template< typename GridType,
          typename GridEntity,
          typename DifferentialOperator,
          typename RightHandSide,
          typename MeshFunction >
__global__ void 
heatEquationTemplatedCompact( const GridType* grid,
                              const DifferentialOperator* differentialOperator,
                              const RightHandSide* rightHandSide,
                              MeshFunction* _u,
                              MeshFunction* _fu,
                              const typename GridType::RealType time,
                              const typename GridEntity::CoordinatesType begin,
                              const typename GridEntity::CoordinatesType end,
                              const typename GridEntity::EntityOrientationType entityOrientation,
                              const typename GridEntity::EntityBasisType entityBasis,   
                              const typename GridType::IndexType gridXIdx,
                              const typename GridType::IndexType gridYIdx )
{
   typename GridType::CoordinatesType coordinates;
   typedef typename GridType::IndexType IndexType;
   typedef typename GridType::RealType RealType;

   //TestEntity< GridType > *entities = getSharedMemory< TestEntity< GridType > >();
   //TestEntity< GridType >& entity = entities[ threadIdx.y * 16 + threadIdx.x ];
   //new ( &entity ) TestEntity< GridType >( *grid, coordinates, entityOrientation, entityBasis );
   
   coordinates.x() = begin.x() + ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = begin.y() + ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;  
   
   //TestEntity< GridType > entity( *grid, coordinates, entityOrientation, entityBasis );
   GridEntity entity( *grid, coordinates, entityOrientation, entityBasis );
   //const GridType* g = grid;
   
   MeshFunction& u = *_u;
   MeshFunction& fu = *_fu;

   //if( threadIdx.x == 0 )
   //   printf( "entity size = %d \n", sizeof( GridEntity ) );
   //if( entity.getCoordinates().x() < end.x() &&
   //    entity.getCoordinates().y() < end.y() )
   {
      
      entity.refresh();
      if( ! entity.isBoundaryEntity() )
      {
         fu( entity ) = 
            ( *differentialOperator )( u, entity, time );

         typedef tnlFunctionAdapter< GridType, RightHandSide > FunctionAdapter;
         fu( entity ) +=  FunctionAdapter::getValue( *rightHandSide, entity, time );
      }
   }
      
   //GridEntity entity( grid, coordinates, entityOrientation, entityBasis );
   //printf( "size = %d ", sizeof( GridEntity ) );
   //entity.refresh();
   //typename GridType::TestCell entity( grid, coordinates, entityOrientation, entityBasis );
   
   /*const IndexType tidX = begin.x() + ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const IndexType tidY = begin.y() + ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   MeshFunction& u = *_u;
   MeshFunction& fu = *_fu;
   if( tidX > 0 && tidX < end.x() - 1 && tidY > 0 && tidY < end.y() - 1 )      
   {
      const IndexType& xSize = grid->getDimensions().x();
      const IndexType& c = tidY * xSize + tidX;
      const RealType& hxSquareInverse = grid->template getSpaceStepsProducts< -2, 0 >(); 
      const RealType& hySquareInverse = grid->template getSpaceStepsProducts< 0, -2 >(); 
      fu[ c ] = ( u[ c - 1 ] - 2.0 * u[ c ] + u[ c + 1 ]  ) * hxSquareInverse +
                ( u[ c - xSize ] - 2.0 * u[ c ] + u[ c + xSize ] ) * hySquareInverse;      
   }*/
}
#endif



template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getExplicitRHS( const RealType& time,
                const RealType& tau,
                const MeshPointer& mesh,
                DofVectorPointer& uDofs,
                DofVectorPointer& fuDofs,
                MeshDependentDataType& meshDependentData )
{
   /****
    * If you use an explicit solver like tnlEulerSolver or tnlMersonSolver, you
    * need to implement this method. Compute the right-hand side of
    *
    *   d/dt u(x) = fu( x, u )
    *
    * You may use supporting mesh dependent data if you need.
    */

   if( std::is_same< DeviceType, tnlHost >::value )
   {
      const IndexType gridXSize = mesh->getDimensions().x();
      const IndexType gridYSize = mesh->getDimensions().y();
      const RealType& hx_inv = mesh->template getSpaceStepsProducts< -2,  0 >();
      const RealType& hy_inv = mesh->template getSpaceStepsProducts<  0, -2 >();
      RealType* u = uDofs->getData();
      RealType* fu = fuDofs->getData();
      for( IndexType j = 0; j < gridYSize; j++ )
      {
         fu[ j * gridXSize ] = 0.0; //u[ j * gridXSize + 1 ];
         fu[ j * gridXSize + gridXSize - 2 ] = 0.0; //u[ j * gridXSize + gridXSize - 1 ];
      }
      for( IndexType i = 0; i < gridXSize; i++ )
      {
         fu[ i ] = 0.0; //u[ gridXSize + i ];
         fu[ ( gridYSize - 1 ) * gridXSize + i ] = 0.0; //u[ ( gridYSize - 2 ) * gridXSize + i ];
      }
      
      /*typedef typename MeshType::Cell CellType;
      typedef typename CellType::CoordinatesType CoordinatesType;
      CoordinatesType coordinates( 0, 0 ), entityOrientation( 0,0 ), entityBasis( 0, 0 );*/
      
      //CellType entity( mesh, coordinates, entityOrientation, entityBasis );

      for( IndexType j = 1; j < gridYSize - 1; j++ )
         for( IndexType i = 1; i < gridXSize - 1; i++ )
         {
            const IndexType c = j * gridXSize + i;
            fu[ c ] = tau * ( ( u[ c - 1 ] - 2.0 * u[ c ] + u[ c + 1 ] ) * hx_inv +
                              ( u[ c - gridXSize ] - 2.0 * u[ c ] + u[ c + gridXSize ] ) * hy_inv );
         }
   }
   if( std::is_same< DeviceType, tnlCuda >::value )
   {
      #ifdef HAVE_CUDA         
      if( this->cudaKernelType == "pure-c" )
      {
         const IndexType gridXSize = mesh->getDimensions().x();
         const IndexType gridYSize = mesh->getDimensions().y();
         const RealType& hx_inv = mesh->template getSpaceStepsProducts< -2,  0 >();
         const RealType& hy_inv = mesh->template getSpaceStepsProducts<  0, -2 >();

         dim3 cudaBlockSize( 16, 16 );
         dim3 cudaGridSize( gridXSize / 16 + ( gridXSize % 16 != 0 ),
                            gridYSize / 16 + ( gridYSize % 16 != 0 ) );

         int cudaErr;
         boundaryConditionsKernel<<< cudaGridSize, cudaBlockSize >>>( uDofs->getData(), fuDofs->getData(), gridXSize, gridYSize );
         if( ( cudaErr = cudaGetLastError() ) != cudaSuccess )
         {
            cerr << "Setting of boundary conditions failed. " << cudaErr << endl;
            return;
         }

         /****
          * Laplace operator
          */
         //cout << "Laplace operator ... " << endl;
         heatEquationKernel<<< cudaGridSize, cudaBlockSize >>>
            ( uDofs->getData(), fuDofs->getData(), tau, hx_inv, hy_inv, gridXSize, gridYSize );
         if( cudaGetLastError() != cudaSuccess )
         {
            cerr << "Laplace operator failed." << endl;
            return;
         }
      }
      if( this->cudaKernelType == "templated-compact" )
      {
         typedef typename MeshType::MeshEntity< 2 > CellType;
         //typedef typename MeshType::Cell CellType;
         //std::cerr << "Size of entity is ... " << sizeof( TestEntity< MeshType > ) << " vs. " << sizeof( CellType ) << std::endl;
         typedef typename CellType::CoordinatesType CoordinatesType;
         u->bind( mesh, uDofs );
         fu->bind( mesh, fuDofs );
         fu->getData().setValue( 1.0 );
         const CoordinatesType begin( 0,0 );
         const CoordinatesType& end = mesh->getDimensions();
         CellType cell( mesh.template getData< DeviceType >() );
         dim3 cudaBlockSize( 16, 16 );
         dim3 cudaBlocks;
         cudaBlocks.x = tnlCuda::getNumberOfBlocks( end.x() - begin.x() + 1, cudaBlockSize.x );
         cudaBlocks.y = tnlCuda::getNumberOfBlocks( end.y() - begin.y() + 1, cudaBlockSize.y );
         const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
         const IndexType cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
         
         //std::cerr << "Setting boundary conditions..." << std::endl;

         tnlCuda::synchronizeDevice();
         for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
            for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
               boundaryConditionsTemplatedCompact< MeshType, CellType, BoundaryCondition, MeshFunctionType >
                  <<< cudaBlocks, cudaBlockSize >>>
                  ( &mesh.template getData< tnlCuda >(),
                    &boundaryConditionPointer.template getData< tnlCuda >(),
                    &u.template modifyData< tnlCuda >(),
                    time,
                    begin,
                    end,
                    cell.getOrientation(),
                    cell.getBasis(),
                    gridXIdx,
                    gridYIdx );
         cudaThreadSynchronize();
         
         //std::cerr << "Computing the heat equation ..." << std::endl;
         for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
            for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
               heatEquationTemplatedCompact< MeshType, CellType, DifferentialOperator, RightHandSide, MeshFunctionType >
                  <<< cudaBlocks, cudaBlockSize >>>
                  ( &mesh.template getData< DeviceType >(),
                    &differentialOperatorPointer.template getData< DeviceType >(),
                    &rightHandSidePointer.template getData< DeviceType >(),
                    &u.template modifyData< DeviceType >(),
                    &fu.template modifyData< DeviceType >(),
                    time,
                    begin,
                    end,
                    cell.getOrientation(),
                    cell.getBasis(),
                    gridXIdx,
                    gridYIdx );
         checkCudaDevice;
         cudaThreadSynchronize();         
      }
      #endif
      if( this->cudaKernelType == "templated" )
      {
         //if( !this->cudaMesh )
         //   this->cudaMesh = tnlCuda::passToDevice( &mesh );
         MeshFunctionPointer uPointer( mesh, uDofs );
         MeshFunctionPointer fuPointer( mesh, fuDofs );
         tnlExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
         //explicitUpdater.setGPUTransferTimer( this->gpuTransferTimer ); 
         explicitUpdater.template update< typename Mesh::Cell >( 
            time,
            mesh,
            this->differentialOperatorPointer,
            this->boundaryConditionPointer,
            this->rightHandSidePointer,
            uPointer,
            fuPointer );
            }
   }
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
   template< typename MatrixPointer >
void
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
assemblyLinearSystem( const RealType& time,
                      const RealType& tau,
                      const MeshPointer& mesh,
                      DofVectorPointer& _u,
                      MatrixPointer& matrix,
                      DofVectorPointer& b,
                      MeshDependentDataType& meshDependentData )
{
   tnlLinearSystemAssembler< Mesh,
                             MeshFunctionType,
                             DifferentialOperator,
                             BoundaryCondition,
                             RightHandSide,
                             tnlBackwardTimeDiscretisation,
                             typename MatrixPointer::ObjectType,
                             typename DofVectorPointer::ObjectType > systemAssembler;

   typedef tnlMeshFunction< Mesh > MeshFunctionType;
   typedef tnlSharedPointer< MeshFunctionType, DeviceType > MeshFunctionPointer;
   MeshFunctionPointer u( mesh, *_u );
   systemAssembler.template assembly< typename Mesh::Cell >( time,
                                                             tau,
                                                             mesh,
                                                             this->differentialOperator,
                                                             this->boundaryCondition,
                                                             this->rightHandSide,
                                                             u,
                                                             matrix,
                                                             b );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
~HeatEquationBenchmarkProblem()
{
   if( this->cudaMesh ) tnlCuda::freeFromDevice( this->cudaMesh );
   if( this->cudaBoundaryConditions )  tnlCuda::freeFromDevice( this->cudaBoundaryConditions );
   if( this->cudaRightHandSide ) tnlCuda::freeFromDevice( this->cudaRightHandSide );
   if( this->cudaDifferentialOperator ) tnlCuda::freeFromDevice( this->cudaDifferentialOperator );
}


#endif /* HeatEquationBenchmarkPROBLEM_IMPL_H_ */
