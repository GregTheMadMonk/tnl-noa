#ifndef TNLNEUMANNBOUNDARYCONDITIONS_H
#define	TNLNEUMANNBOUNDARYCONDITIONS_H

template< typename Mesh,
          typename Function,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlNeumannBoundaryConditions
{

};

/****
 * Base
 */
template< typename Function >
class tnlNeumannBoundaryConditionsBase
{
   public:
      
      typedef Function FunctionType;

      static void configSetup( tnlConfigDescription& config,
                               const tnlString& prefix = "" );

      bool setup( const tnlParameterContainer& parameters,
                  const tnlString& prefix = "" );

      void setFunction( const FunctionType& function );
      
      FunctionType& getFunction();

      const FunctionType& getFunction() const;

   protected:

      FunctionType function;

};

/****
 * 1D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
class tnlNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Function, Real, Index >
   : public tnlNeumannBoundaryConditionsBase< Function >
{
   public:

   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef Function FunctionType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 1, RealType > VertexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;

   template< typename EntityType,
             typename MeshFunction >
   __cuda_callable__
   const RealType getValue( const EntityType& entity,
                            const MeshFunction& u,
                            const RealType& time = 0 ) const;


   template< typename EntityType >
   __cuda_callable__
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const EntityType& entity ) const;

   template< typename MatrixRow,
             typename EntityType,
             typename MeshFunction >
   __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const MeshType& mesh,
                               const IndexType& index,
                               const EntityType& entity,
                               const MeshFunction& u,
                               DofVectorType& b,
                               MatrixRow& matrixRow ) const;
};

/****
 * 2D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
class tnlNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Function, Real, Index >
   : public tnlNeumannBoundaryConditionsBase< Function >
{
   public:

   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef Function FunctionType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 2, RealType > VertexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;

   template< typename EntityType,
             typename MeshFunction >
   __cuda_callable__
   const RealType getValue( const EntityType& entity,                            
                            const MeshFunction& u,
                            const RealType& time = 0 ) const;
      
   template< typename EntityType >
   __cuda_callable__
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const EntityType& entity ) const;

   template< typename MatrixRow,
             typename EntityType,
             typename MeshFunction >
   __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const MeshType& mesh,
                               const IndexType& index,
                               const EntityType& entity,
                               const MeshFunction& u,
                               DofVectorType& b,
                               MatrixRow& matrixRow ) const;
};

/****
 * 3D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
class tnlNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Function, Real, Index >
   : public tnlNeumannBoundaryConditionsBase< Function >
{
   public:

   typedef tnlGrid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef Function FunctionType;
   typedef tnlSharedVector< RealType, DeviceType, IndexType > SharedVector;
   typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
   typedef tnlStaticVector< 3, RealType > VertexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;

   template< typename EntityType,
             typename MeshFunction >
   __cuda_callable__
   const RealType getValue( const EntityType& entity,                            
                            const MeshFunction& u,
                            const RealType& time = 0 ) const;
   

   template< typename EntityType >
   __cuda_callable__
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const EntityType& entity ) const;

   template< typename MatrixRow,
             typename EntityType,
             typename MeshFunction >
   __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const MeshType& mesh,
                               const IndexType& index,
                               const EntityType& entity,
                               const MeshFunction& u,
                               DofVectorType& b,
                               MatrixRow& matrixRow ) const;
};

template< typename Mesh,
          typename Function,
          typename Real,
          typename Index >
ostream& operator << ( ostream& str, const tnlNeumannBoundaryConditions< Mesh, Function, Real, Index >& bc )
{
   str << "Neumann boundary conditions: function = " << bc.getFunction();
   return str;
}


#include <operators/tnlNeumannBoundaryConditions_impl.h>

#endif	/* TNLNEUMANNBOUNDARYCONDITIONS_H */

