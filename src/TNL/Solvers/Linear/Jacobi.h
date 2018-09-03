/***************************************************************************
                          Jacobi.h  -  description
                             -------------------
    begin                : Jul 30, 2007
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#pragma once

#include <TNL/Object.h>
#include <TNL/Solvers/IterativeSolver.h>
#include <TNL/Solvers/Linear/Preconditioners/Dummy.h>
#include <TNL/Solvers/Linear/LinearResidueGetter.h>

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix,
          typename Preconditioner = Preconditioners::Dummy< typename Matrix::RealType,
                                                            typename Matrix::DeviceType,
                                                            typename Matrix::IndexType> >
class Jacobi : public Object,
               public IterativeSolver< typename Matrix::RealType,
                                       typename Matrix::IndexType >
{
   public:

   typedef typename Matrix::RealType RealType;
   typedef typename Matrix::IndexType IndexType;
   typedef typename Matrix::DeviceType DeviceType;
   typedef Matrix MatrixType;
   typedef Preconditioner PreconditionerType;


   Jacobi() : omega(0) {}

   String getType() const
   {
      return String( "Jacobi< " ) + this->matrix->getType() + ", " +   this->preconditioner->getType() + " >";
   }

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" )
   {
      config.addEntry< double >( prefix + "jacobi-omega", "Relaxation parameter of the weighted/damped Jacobi method.", 1.0 );
   }

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" )
   {
      IterativeSolver< RealType, IndexType >::setup( parameters, prefix );
      this->setOmega( parameters.getParameter< double >( prefix + "jacobi-omega" ) );
      if( this->omega <= 0.0 || this->omega > 2.0 )
      {
         std::cerr << "Warning: The Jacobi method parameter omega is out of interval (0,2). The value is " << this->omega << " the method will not converge." << std::endl;
      }
      return true;
   }

   void setOmega( const RealType& omega )
   {
      this->omega = omega;
   }

   const RealType& getOmega() const
   {
      return omega;
   }

   void setMatrix( const MatrixType& matrix )
   {
      this->matrix = &matrix;
   }

   void setPreconditioner( const Preconditioner& preconditioner )
   {
      this->preconditioner = &preconditioner;
   }

   template< typename Vector,
             typename ResidueGetter = LinearResidueGetter< Matrix, Vector > >
   bool solve( const Vector& b, Vector& x, Vector& aux)
   {
      const IndexType size = matrix->getRows();

      this->resetIterations();
      this->setResidue( this->getConvergenceResidue() + 1.0 );

      RealType bNorm = b.lpNorm( ( RealType ) 2.0 );
      aux = x;
      while( this->nextIteration() )
      {
         for( IndexType row = 0; row < size; row ++ )
            matrix->performJacobiIteration( b, row, x, aux, this->getOmega() );
        for( IndexType row = 0; row < size; row ++ )
            matrix->performJacobiIteration( b, row, aux, x, this->getOmega() );
         this->setResidue( ResidueGetter::getResidue( *matrix, x, b, bNorm ) );
      }
      this->setResidue( ResidueGetter::getResidue( *matrix, x, b, bNorm ) );
      this->refreshSolverMonitor();
      return this->checkConvergence();
   }

   protected:

      RealType omega;
      const MatrixType* matrix;
      const PreconditionerType* preconditioner;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL
