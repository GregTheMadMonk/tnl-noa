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
   if( ! this->boundaryCondition.setup( parameters, "boundary-conditions-" ) ||
       ! this->rightHandSide.setup( parameters, "right-hand-side-" ) )
      return false;
   this->cudaKernelType = parameters.getParameter< tnlString >( "cuda-kernel-type" );

   if( std::is_same< DeviceType, tnlCuda >::value )
   {
      this->cudaBoundaryConditions = tnlCuda::passToDevice( this->boundaryCondition );
      this->cudaRightHandSide = tnlCuda::passToDevice( this->rightHandSide );
      this->cudaDifferentialOperator = tnlCuda::passToDevice( this->differentialOperator );
   }
   
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
typename HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::IndexType
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getDofs( const MeshType& mesh ) const
{
   /****
    * Return number of  DOFs (degrees of freedom) i.e. number
    * of unknowns to be resolved by the main solver.
    */
   return mesh.template getEntitiesCount< typename MeshType::Cell >();
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
bindDofs( const MeshType& mesh,
          DofVectorType& dofVector )
{
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setInitialCondition( const tnlParameterContainer& parameters,
                     const MeshType& mesh,
                     DofVectorType& dofs,
                     MeshDependentDataType& meshDependentData )
{
   const tnlString& initialConditionFile = parameters.getParameter< tnlString >( "initial-condition" );
   tnlMeshFunction< Mesh > u( mesh, dofs );
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
                                                                          differentialOperator,
                                                                          boundaryCondition,
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
              const MeshType& mesh,
              DofVectorType& dofs,
              MeshDependentDataType& meshDependentData )
{
   cout << endl << "Writing output at time " << time << " step " << step << "." << endl;
   this->bindDofs( mesh, dofs );
   MeshFunctionType u;
   u.bind( mesh, dofs );
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
boundaryConditionsTemplatedCompact( const GridType grid,
                                    const BoundaryConditions boundaryConditions,
                                    MeshFunction u,
                                    const typename GridType::RealType time,
                                    const typename GridEntity::CoordinatesType begin,
                                    const typename GridEntity::CoordinatesType end,
                                    const typename GridEntity::EntityOrientationType entityOrientation,
                                    const typename GridEntity::EntityBasisType entityBasis,   
                                    const typename GridType::IndexType gridXIdx,
                                    const typename GridType::IndexType gridYIdx )
{
   /*typename GridType::CoordinatesType coordinates;

   coordinates.x() = begin.x() + ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = begin.y() + ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;  
   
   GridEntity entity( grid, coordinates, entityOrientation, entityBasis );

   if( entity.getCoordinates().x() < end.x() &&
       entity.getCoordinates().y() < end.y() )
   {
      entity.refresh();
      if( entity.isBoundaryEntity() )
      {
         u( entity ) = boundaryConditions( u, entity, time );
      }
   }*/
   typedef typename GridEntity::IndexType IndexType;
   typedef typename GridEntity::RealType RealType;
   RealType* _u = &u[ 0 ];
   const IndexType tidX = begin.x() + ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const IndexType tidY = begin.y() + ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   if( tidX == 0 || tidX == end.x() - 1 || tidY == 0 || tidY == end.y() - 1 )      
   {
      _u[ tidY * grid.getDimensions().x() + tidX ] = 0.0;
   }   
}

/*template< typename Grid,
          int EntityDimensions = 2,
          typename Config = tnlGridEntityNoStencilStorage >
struct TestEntity
{      
      typedef Grid GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef Config ConfigType;
      
      static const int meshDimensions = GridType::meshDimensions;
      
      static const int entityDimensions = EntityDimensions;
            
      constexpr static int getDimensions() { return EntityDimensions; };
      
      constexpr static int getMeshDimensions() { return meshDimensions; };
   
      typedef TestEntity< GridType, EntityDimensions, Config > ThisType;
      typedef tnlNeighbourGridEntitiesStorage< ThisType > NeighbourGridEntitiesStorageType;
   
   
   __cuda_callable__ TestEntity(  const GridType& grid,
               const CoordinatesType& coordinates,
               const CoordinatesType& entityOrientation,
               const CoordinatesType& entityBasis )
      : grid( grid ), coordinates( coordinates ),
      entityOrientation( 0  ),
      entityBasis( 1 ),
      neighbourEntitiesStorage( *this )
   {
      
   }
      
       const GridType& grid;
   
   CoordinatesType coordinates;
   CoordinatesType entityOrientation;
   CoordinatesType entityBasis;
      
   NeighbourGridEntitiesStorageType neighbourEntitiesStorage;   
};*/

template< typename GridType,
          typename GridEntity,
          typename DifferentialOperator,
          typename RightHandSide,
          typename MeshFunction >
__global__ void 
heatEquationTemplatedCompact( const GridType grid,
                              const DifferentialOperator differentialOperator,
                              const RightHandSide rightHandSide,
                              MeshFunction u,
                              MeshFunction fu,
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

   
   /*coordinates.x() = begin.x() + ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = begin.y() + ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;  
   
   GridEntity entity( grid, coordinates, entityOrientation, entityBasis );

   if( entity.getCoordinates().x() < end.x() &&
       entity.getCoordinates().y() < end.y() )
   {
      
      entity.refresh();
      if( ! entity.isBoundaryEntity() )
      {
         fu( entity ) = 
            differentialOperator( u, entity, time );

         typedef tnlFunctionAdapter< GridType, RightHandSide > FunctionAdapter;
         fu( entity ) +=  FunctionAdapter::getValue( rightHandSide, entity, time );
      }
   }*/
      
   //GridEntity entity( grid, coordinates, entityOrientation, entityBasis );
   //printf( "size = %d ", sizeof( GridEntity ) );
   //entity.refresh();
   //typename GridType::TestCell entity( grid, coordinates, entityOrientation, entityBasis );
   
   const IndexType tidX = begin.x() + ( gridXIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const IndexType tidY = begin.y() + ( gridYIdx * tnlCuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   if( tidX > 0 && tidX < end.x() - 1 && tidY > 0 && tidY < end.y() - 1 )      
   {
      const IndexType& xSize = grid.getDimensions().x();
      const IndexType& c = tidY * xSize + tidX;
      const RealType& hxSquareInverse = grid.template getSpaceStepsProducts< -2, 0 >(); 
      const RealType& hySquareInverse = grid.template getSpaceStepsProducts< 0, -2 >(); 
      fu[ c ] = ( u[ c - 1 ] - 2.0 * u[ c ] + u[ c + 1 ]  ) * hxSquareInverse +
                ( u[ c - xSize ] - 2.0 * u[ c ] + u[ c + xSize ] ) * hySquareInverse;      
   }   
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
                const MeshType& mesh,
                DofVectorType& uDofs,
                DofVectorType& fuDofs,
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
      const IndexType gridXSize = mesh.getDimensions().x();
      const IndexType gridYSize = mesh.getDimensions().y();
      const RealType& hx_inv = mesh.template getSpaceStepsProducts< -2,  0 >();
      const RealType& hy_inv = mesh.template getSpaceStepsProducts<  0, -2 >();
      RealType* u = uDofs.getData();
      RealType* fu = fuDofs.getData();
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
      if( this->cudaKernelType == "pure-c" )
      {
         const IndexType gridXSize = mesh.getDimensions().x();
         const IndexType gridYSize = mesh.getDimensions().y();
         const RealType& hx_inv = mesh.template getSpaceStepsProducts< -2,  0 >();
         const RealType& hy_inv = mesh.template getSpaceStepsProducts<  0, -2 >();

         dim3 cudaBlockSize( 16, 16 );
         dim3 cudaGridSize( gridXSize / 16 + ( gridXSize % 16 != 0 ),
                            gridYSize / 16 + ( gridYSize % 16 != 0 ) );

         int cudaErr;
         boundaryConditionsKernel<<< cudaGridSize, cudaBlockSize >>>( uDofs.getData(), fuDofs.getData(), gridXSize, gridYSize );
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
            ( uDofs.getData(), fuDofs.getData(), tau, hx_inv, hy_inv, gridXSize, gridYSize );
         if( cudaGetLastError() != cudaSuccess )
         {
            cerr << "Laplace operator failed." << endl;
            return;
         }
      }
      if( this->cudaKernelType == "templated-compact" )
      {
#ifdef HAVE_CUDA         
         typedef typename MeshType::Cell CellType;
         typedef typename CellType::CoordinatesType CoordinatesType;         
         MeshFunctionType u( mesh, uDofs );
         MeshFunctionType fu( mesh, fuDofs );
         fu.getData().setValue( 1.0 );
         const CoordinatesType begin( 0,0 );
         const CoordinatesType& end = mesh.getDimensions();
         CellType cell( mesh );
         dim3 cudaBlockSize( 16, 16 );
         dim3 cudaBlocks;
         cudaBlocks.x = tnlCuda::getNumberOfBlocks( end.x() - begin.x() + 1, cudaBlockSize.x );
         cudaBlocks.y = tnlCuda::getNumberOfBlocks( end.y() - begin.y() + 1, cudaBlockSize.y );
         const IndexType cudaXGrids = tnlCuda::getNumberOfGrids( cudaBlocks.x );
         const IndexType cudaYGrids = tnlCuda::getNumberOfGrids( cudaBlocks.y );
         
         //std::cerr << "Setting boundary conditions..." << std::endl;

         for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
            for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
               boundaryConditionsTemplatedCompact< MeshType, CellType, BoundaryCondition, MeshFunctionType >
                  <<< cudaBlocks, cudaBlockSize >>>
                  ( mesh,
                    boundaryCondition,
                    u,
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
                  ( mesh,
                    differentialOperator,
                    rightHandSide,
                    u,
                    fu,
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
         MeshFunctionType u( mesh, uDofs );
         MeshFunctionType fu( mesh, fuDofs );
         tnlExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
         //explicitUpdater.setGPUTransferTimer( this->gpuTransferTimer ); 
         explicitUpdater.template update< typename Mesh::Cell >( 
            time,
            mesh,
            this->differentialOperator,
            this->boundaryCondition,
            this->rightHandSide,
            u,
            fu );
            }
   }
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
   template< typename Matrix >
void
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
assemblyLinearSystem( const RealType& time,
                      const RealType& tau,
                      const MeshType& mesh,
                      DofVectorType& _u,
                      Matrix& matrix,
                      DofVectorType& b,
                      MeshDependentDataType& meshDependentData )
{
   tnlLinearSystemAssembler< Mesh,
                             MeshFunctionType,
                             DifferentialOperator,
                             BoundaryCondition,
                             RightHandSide,
                             tnlBackwardTimeDiscretisation,
                             Matrix,
                             DofVectorType > systemAssembler;

   tnlMeshFunction< Mesh > u( mesh, _u );
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
