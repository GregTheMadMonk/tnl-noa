#ifndef TNLPOISSONPROBLEM_H
#define	TNLPOISSONPROBLEM_H

#include <core/vectors/tnlVector.h>
#include <mesh/tnlGrid.h>

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlPoissonProblem
{
 
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
		  typename Index,
		  typename MatrixType>
class tnlPoissonProblem< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >
{
   public: 
   
   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   static tnlString getType()
   {
	  return tnlString( "tnlPoisssonMatrix< " ) +
			 MeshType::getType() + ", " +
			 ::getType< Real >() + ", " +
			 ::getType< Index >() + " >";
   }
   
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Real getValue( const MeshType& mesh,
                  const IndexType cellIndex,
                  const CoordinatesType& coordinates,
                  const Vector& u,
                  const Real& time ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
								   const CoordinatesType& coordinates ) const
   {
	   return 5;
   }

#ifdef HAVE_CUDA
   __device__ __host__
#endif
	  void initLinearSystem(
                               const MeshType& mesh,
                               const IndexType& index,
							   )
   {
	   _matrix.setElement(index,index,4);
	   _matrix.setElement(index,mesh.getCellYPredecessor( index ),-1);
	   _matrix.setElement(index,mesh.getCellXPredecessor( index ),-1);
	   _matrix.setElement(index,mesh.getCellXSuccessor( index ),-1);
	   _matrix.setElement(index,mesh.getCellYSuccessor( index ),-1);

	  /*matrixRow.setElement( 0, mesh.getCellYPredecessor( index ), -1 );
	  matrixRow.setElement( 1, mesh.getCellXPredecessor( index ), -1 );
	  matrixRow.setElement( 2, index,                            4.0 );
	  matrixRow.setElement( 3, mesh.getCellXSuccessor( index ),   -1 );
	  matrixRow.setElement( 4, mesh.getCellYSuccessor( index ),   -1 );*/
   }

   void init(const MeshType& mesh)
   {
	   typename MatrixType::RowLengthsVector rowLenghts;
	   IndexType N = mesh.getNumberOfCells();
	   rowLenghts.setSize(N);
	   rowLenghts.setValue(5);
	   _matrix.setDimensions( N, N );
	   _matrix.setRowLengths(rowLenghts);

	   for (IndexType i = 0; i < N; i++)
		   initLinearSystem(mesh, i);

   }
   template< typename Vector, typename Solver >
   void solve(Solver & solver, const Vector & rhs, Vector & result)
   {
	   solver.setMatrix(_matrix);
	   solver.solve(rhs,result);
   }

   MatrixType _matrix;
};




#include "tnlPoissonProblem_impl.h"


#endif	/* TNLPOISSONPROBLEM_H */
