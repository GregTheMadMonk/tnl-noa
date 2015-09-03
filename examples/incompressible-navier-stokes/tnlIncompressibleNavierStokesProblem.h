/***************************************************************************
                          tnlIncompressibleNavierStokesProblem.h  -  description
                             -------------------
    begin                : Feb 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLINCOMPRESSIBLENAVIERSTOKESPROBLEM_H_
#define TNLINCOMPRESSIBLENAVIERSTOKESPROBLEM_H_

#include <problems/tnlPDEProblem.h>
#include <operators/diffusion/tnlLinearDiffusion.h>
#include <core/arrays/tnlStaticArray.h>
#include "tnlINSBoundaryConditions.h"
#include "tnlExplicitINSTimeStepper.h"

template<class T> T square(const T & val){return val*val;}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
class tnlIncompressibleNavierStokesProblem : public tnlPDEProblem< Mesh,
                                                                   typename DifferentialOperator::RealType,
                                                                   typename Mesh::DeviceType,
                                                                   typename DifferentialOperator::IndexType  >
{
   public:

      typedef typename DifferentialOperator::RealType RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef typename DifferentialOperator::IndexType IndexType;
      typedef tnlPDEProblem< Mesh, RealType, DeviceType, IndexType > BaseType;
      typedef tnlIncompressibleNavierStokesProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator > ThisType;

      using typename BaseType::MeshType;
      using typename BaseType::DofVectorType;

	  typedef tnlCSRMatrix< RealType, tnlHost, IndexType > MatrixType;
	  typedef tnlSORSolver<MatrixType> LinearSolver;
	  typedef tnlExplicitINSTimeStepper< ThisType, LinearSolver > TimeStepper;
	  typedef typename MeshType::CoordinatesType CoordinatesType;
	  //typedef tnlExplicitINSTimeStepper< ThisType, void > TimeStepper;
	  
	   enum { Dimensions = Mesh::Dimensions };

      static tnlString getTypeStatic() {return tnlString( "tnlNSProblem< " ) + Mesh :: getTypeStatic() + " >";}

      tnlString getPrologHeader() const{return tnlString( "NS equation" );}

      void writeProlog( tnlLogger& logger,
                        const tnlParameterContainer& parameters ) const {}

      bool setup( const tnlParameterContainer& parameters );

      bool setInitialCondition( const tnlParameterContainer& parameters,
                                const MeshType& mesh,
                                DofVectorType& dofs,
                                DofVectorType& auxDofs );

	  //template< typename MatrixType >
      bool setupLinearSystem( const MeshType& mesh,
                              MatrixType& matrix );

      bool makeSnapshot( const RealType& time,
                         const IndexType& step,
                         const MeshType& mesh,
                         DofVectorType& dofs,
                         DofVectorType& auxDofs );

      IndexType getDofs( const MeshType& mesh ) const;

      void bindDofs( const MeshType& mesh,
                     DofVectorType& dofs );

      void getExplicitRHS( const RealType& time,
                           const RealType& tau,
                           const MeshType& mesh,
                           DofVectorType& _u,
                           DofVectorType& _fu );

	  //template< typename MatrixType >
      void assemblyLinearSystem( const RealType& time,
                                 const RealType& tau,
                                 const MeshType& mesh,
                                 DofVectorType& dofs,
                                 DofVectorType& auxDofs,
                                 MatrixType& matrix,
                                 DofVectorType& rightHandSide );

	  void set_zero_neumann(tnlSharedVector< RealType, DeviceType, IndexType > & vec)
	  {
		  /*int ex = a.width()-1, ey=a.height()-1;
		  for (int x=1; x < ex; x++) {a(x,0) = a(x,1); a(x,ey) = a(x,ey-1);}
		  for (int y=1; y < ey; y++) {a(0,y) = a(1,y); a(ex,y) = a(ex-1,y);}
		  a(0,0)=0.5*(a(0,1)+a(1,0));
		  a(ex,0)=0.5*(a(ex-1,0)+a(ex,1));
		  a(0,ey)=0.5*(a(1,ey)+a(0,ey-1));
		  a(ex,ey)=0.5*(a(ex-1,ey)+a(ex,ey-1));*/
	  }

	  static void SetBnd(const MeshType& mesh)
	  {
		  mesh;
	  }

	  void computeVelocityDivergence(IndexType cell, const tnlVector<RealType, DeviceType, IndexType> & v, const MeshType& mesh, tnlVector<RealType, DeviceType, IndexType> & rhs)
	  {
		  double diffU = v[mesh.template getFaceNextToCell<1,0>(cell)] - v[mesh.template getFaceNextToCell<-1,0>(cell)];
		  double diffV = v[mesh.template getFaceNextToCell<0,1>(cell)] - v[mesh.template getFaceNextToCell<0,-1>(cell)];
		  rhs[cell] = -0.5f*(diffU/mesh.getDimensions().x() + diffV/mesh.getDimensions().y()); // -(u_x + v_y)
	  }
	  void updateVelocityByPressureCorrection(IndexType cell, const tnlVector<RealType, DeviceType, IndexType> & v, const MeshType& mesh, tnlVector<RealType, DeviceType, IndexType> & p)
	  {
		  RealType pVal = p[cell];
		  double nx =mesh.getDimensions().x(), ny=mesh.getDimensions().y();
		  vel[mesh.template getFaceNextToCell<-1,0>(cell)] -= 0.5*nx*pVal;
		  vel[mesh.template getFaceNextToCell<+1,0>(cell)] += 0.5*nx*pVal;
		  vel[mesh.template getFaceNextToCell<0,-1>(cell)] -= 0.5*ny*pVal;
		  vel[mesh.template getFaceNextToCell<0,+1>(cell)] += 0.5*ny*pVal;
	  }

	  void project(const MeshType& mesh)
	  {
		  int nx = mesh.getDimensions().x(), ny=mesh.getDimensions().y();
		  for ( int i=0 ; i< nx; i++ ) for (int j=0 ; j< ny; j++ )
			  computeVelocityDivergence(mesh.getCellIndex(CoordinatesType( i, j )), vel, mesh, p_rhs);

		  _matSolver.setMatrix(poissonMat);
		  _matSolver.solve(p_rhs,p);

		  for (int x=1; x< nx-1; x++) for(int y=1; y < ny-1; y++)
			 updateVelocityByPressureCorrection(mesh.getCellIndex(CoordinatesType(x,y)),vel, mesh, p);
	  }
	  void diffuse(RealType dt, const MeshType& mesh)
	  {
		  vel0 = vel;
		   //vytvorit matici s prvky
		  int nx = mesh.getDimensions().x(), ny=mesh.getDimensions().y();
		  double a = dt*visc*nx*nx, b = dt*visc*ny*ny;
		  prepareMatrix<1,0>(diffuseMat, mesh, nx+1, ny, -a, 1+2*a+2*b);
		  prepareMatrix<0,1>(diffuseMat, mesh, nx, ny+1, -b, 1+2*a+2*b);
		  _matSolver.setMatrix(diffuseMat);
		  _matSolver.solve(vel0,vel);
	  }
	  double getCenterU(const MeshType& mesh, IndexType cell) //x,y based on cells
	  {
		  return 0.5*(vel0[mesh.template getFaceNextToCell<-1,0>(cell)] + vel0[mesh.template getFaceNextToCell<+1,0>(cell)] );
	  }
	  double getCenterV(const MeshType& mesh, IndexType cell) //x,y based on cells
	  {
		  return 0.5*(vel0[mesh.template getFaceNextToCell<0,-1>(cell)] + vel0[mesh.template getFaceNextToCell<0,+1>(cell)] );
	  }
	  double getCrossU(const MeshType& mesh, int x, int y) //x,y based (n+1)*(n+1)
	  {
		  const CoordinatesType cellCoords(x,y);
		  const CoordinatesType downCoords(x,y-1);
		  return 0.5*(vel0[mesh.template getFaceNextToCell<-1,0>(mesh.getCellIndex(cellCoords))]
					 +vel0[mesh.template getFaceNextToCell<-1,0>(mesh.getCellIndex(downCoords))]);
	  }
	  double getCrossV(const MeshType& mesh, int x, int y) //x,y based (n+1)*(n+1)
	  {
		  const CoordinatesType cellCoords(x,y);
		  const CoordinatesType leftCoords(x-1,y);
		  return 0.5*(vel0[mesh.template getFaceNextToCell<0,-1>(mesh.getCellIndex(cellCoords))]
					 +vel0[mesh.template getFaceNextToCell<0,-1>(mesh.getCellIndex(leftCoords))]);
	  }
	  void advect(RealType dt, const MeshType& mesh)
	  {
		  vel0 = vel;
			int nx = mesh.getDimensions().x(), ny=mesh.getDimensions().y();
			//U has dimensions (nx+1,ny)
			RealType cx = dt*nx, cy = dt*ny; //dt/h
			for ( int i=1 ; i< nx ; i++ ) for (int j=1 ; j< ny-1 ; j++ )
			{
				const CoordinatesType cellCoordinates( i, j );
				const CoordinatesType leftCoords(i-1,j);
				IndexType cell = mesh.getCellIndex(cellCoordinates);
				IndexType face = mesh.template getFaceNextToCell<-1,0>(cell);
				vel[face] = vel0[face]
							- cx*( square(getCenterU(mesh,cell)) - square(getCenterU(mesh, mesh.getCellIndex(leftCoords)))   //(u^2)_x
							  + getCrossU(mesh,i,j+1)*getCrossV(mesh,i,j+1) - getCrossU(mesh,i,j)*getCrossV(mesh,i,j)
						);
			}
			for ( int i=1 ; i< nx-1 ; i++ ) for (int j=1 ; j< ny ; j++ )
			{
				const CoordinatesType cellCoordinates( i, j );
				const CoordinatesType downCoords(i,j-1);
				IndexType cell = mesh.getCellIndex(cellCoordinates);
				IndexType face = mesh.template getFaceNextToCell<0,-1>(cell);
				vel[face] = vel0[face] - cy*(square(getCenterV(mesh,cell)) - square(getCenterV(mesh,mesh.getCellIndex(downCoords)))  //(v^2)_y
							   + getCrossU(mesh,i+1,j)*getCrossV(mesh,i+1,j) - getCrossU(mesh,i,j)*getCrossV(mesh,i,j)
						);
			}
			/*for (int j=0 ; j< ny ; j++ ) for ( int i=0 ; i < nx ; i++ )
			{
				const CoordinatesType cellCoords(i,j);
				IndexType cell = mesh.getCellIndex(cellCoords);
				int ii = mesh.template getFaceNextToCell<-1,0>(cell);
				cout << ii << " ++ "<< i << "  " << j << "   "<<  vel[ii] << endl;
			}
			cout << "----------" <<endl;
			for (int j=0 ; j< ny ; j++ ) for ( int i=0 ; i < nx ; i++ )
			{
				const CoordinatesType cellCoords(i,j);
				IndexType cell = mesh.getCellIndex(cellCoords);
				int ii = mesh.template getFaceNextToCell<0,-1>(cell);
				cout << ii << " ++ "<< i << "  " << j << "   "<<  vel[ii] << endl;
			}
			int test = 3;
			test++;*/
	  }
	  void save(const char * filename, const MeshType& mesh)
	  {
			//FILE * pFile = fopen (filename, "w");
			//fprintf(pFile, "#X	Y	u	v\n");
			int nx = mesh.getDimensions().x(), ny=mesh.getDimensions().y(), n=nx*ny;
			int dims[] = {nx,ny,1};
			double *vars = new double[n*3];
			double *vvars[] = {vars};

			int varDim[] = {3};
			int centering[] = {0};
			const char * names[] = {"Rychlost"};

			for (IndexType j=0 ; j< ny ; j++ ) for ( IndexType i=0 ; i< nx ; i++ )
			{
				IndexType cell = mesh.getCellIndex(typename MeshType::CoordinatesType(i,j));
				int ii = 3*(j*nx+i);
				vars[ii+0] = getCenterU(mesh, cell);
				vars[ii+1] = getCenterV(mesh, cell);
				vars[ii+2] = 0;
				//fprintf(pFile, "%lg	%lg	%lg	%lg\n", (RealType)i, (RealType)j, getCenterU(mesh, cell), getCenterV(mesh, cell));
			}
			//fclose (pFile);
			void write_regular_mesh(const char *filename, int useBinary, int *dims,
									int nvars, int *vardim, int *centering,
									const char * const *varnames, double **vars);
			write_regular_mesh(filename, 0, dims, 1, varDim, centering, names, vvars );
			delete[] vars;
	  }

	  template< int nx, int ny >
	  void prepareMatrix(MatrixType & matrix, const MeshType& mesh, IndexType Nx, IndexType Ny, RealType off, RealType diag)
	  {
		  for (IndexType y = 0; y < Ny; y++) for (IndexType x = 0; x < Nx; x++)
		  {
			  IndexType i = mesh.template getFaceIndex<nx, ny>(typename MeshType::CoordinatesType(x,y));
			  if (x==0 || x ==Nx-1 || y==0 || y==Ny-1) matrix.setElement(i,i,1);
			  else
			  {
				  matrix.setElement(i,i,diag);
				  matrix.setElement(i,mesh.template getFaceIndex<nx, ny>(typename MeshType::CoordinatesType(x-1,y)),off);
				  matrix.setElement(i,mesh.template getFaceIndex<nx, ny>(typename MeshType::CoordinatesType(x+1,y)),off);
				  matrix.setElement(i,mesh.template getFaceIndex<nx, ny>(typename MeshType::CoordinatesType(x,y-1)),off);
				  matrix.setElement(i,mesh.template getFaceIndex<nx, ny>(typename MeshType::CoordinatesType(x,y+1)),off);
			  }
		  }
	  }

      protected:

	  RealType visc;
	  MatrixType poissonMat, diffuseMat;
	  LinearSolver _matSolver;

	  tnlVector<RealType, DeviceType, IndexType> vel, vel0, p, p_rhs;
	  //tnlStaticArray< Dimensions, tnlSharedVector< RealType, DeviceType, IndexType > > u, v;

      DifferentialOperator differentialOperator;

      BoundaryCondition boundaryCondition;
   
      RightHandSide rightHandSide;
};

#include "tnlIncompressibleNavierStokesProblem_impl.h"

#endif /* TNLINCOMPRESSIBLENAVIERSTOKESPROBLEM_H_ */


//Refaktor, do objektu, setup na parametry, laplace podle tnlLinearDiffusion
