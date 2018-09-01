/***************************************************************************
                          tnlSORSolver.h  -  description
                             -------------------
    begin                : 2007/07/30
    copyright            : (C) 2007 by Tomas Oberhuber
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

#ifndef tnlJacobiSolverH
#define tnlJacobiSolverH

#include <math.h>
#include <core/tnlObject.h>
#include <solvers/preconditioners/tnlDummyPreconditioner.h>
#include <solvers/tnlIterativeSolver.h>
#include <solvers/linear/tnlLinearResidueGetter.h>


template< typename Matrix,
          typename Preconditioner = tnlDummyPreconditioner< typename Matrix :: RealType,
                                                            typename Matrix :: DeviceType,
                                                            typename Matrix :: IndexType> >
class tnlJacobiSolver : public tnlObject,
                     public tnlIterativeSolver< typename Matrix :: RealType,
                                                typename Matrix :: IndexType >
{
   public:

   typedef typename Matrix :: RealType RealType;
   typedef typename Matrix :: IndexType IndexType;
   typedef typename Matrix :: DeviceType DeviceType;
   typedef Matrix MatrixType;
   typedef Preconditioner PreconditionerType;


   tnlJacobiSolver():omega(0){}

   tnlString getType() const
   {
      return tnlString( "tnlJacobiSolver< " ) + this -> matrix -> getType() + ", " +	this -> preconditioner -> getType() + " >";
   }

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" ) {config.addEntry< double >( prefix + "jacobi-omega", "Relaxation parameter of the weighted/damped Jacobi method.", 1.0 );}

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" )
   { 
		tnlIterativeSolver< RealType, IndexType >::setup( parameters, prefix );
		this->setOmega( parameters.getParameter< double >( prefix + "jacobi-omega" ) );
		if( this->omega <= 0.0 || this->omega > 2.0 )
		{
			cerr << "Warning: The Jacobi method parameter omega is out of interval (0,2). The value is " << this->omega << " the method will not converge." << endl;
		}
		return true;   
   }

   void setOmega( const RealType& omega ) {this->omega = omega;}
   const RealType& getOmega() const {return omega;}

   void setMatrix( const MatrixType& matrix ){this -> matrix = &matrix;}
   void setPreconditioner( const Preconditioner& preconditioner ){this -> preconditioner = &preconditioner;}

#ifdef HAVE_NOT_CXX11
   template< typename Vector,
             typename ResidueGetter >
   bool solve( const Vector& b, Vector& x )
#else
   template< typename Vector,
             typename ResidueGetter = tnlLinearResidueGetter< Matrix, Vector > >
   bool solve( const Vector& b, Vector& x, Vector& aux)
#endif
   {
   const IndexType size = matrix -> getRows();

   this -> resetIterations();
   this -> setResidue( this -> getConvergenceResidue() + 1.0 );

   RealType bNorm = b. lpNorm( ( RealType ) 2.0 );
   aux = x;
   while( this->nextIteration() )
   {
      for( IndexType row = 0; row < size; row ++ )
         matrix->performJacobiIteration( b, row, x, aux, this->getOmega() );
	  for( IndexType row = 0; row < size; row ++ )
         matrix->performJacobiIteration( b, row, aux, x, this->getOmega() );
      this -> setResidue( ResidueGetter :: getResidue( *matrix, x, b, bNorm ) );
   }
   this -> setResidue( ResidueGetter :: getResidue( *matrix, x, b, bNorm ) );
   this -> refreshSolverMonitor();
   return this->checkConvergence();
   }


   protected:

   RealType omega;
   const MatrixType* matrix;
   const PreconditionerType* preconditioner;

};

#endif
