/***************************************************************************
                          SOR_impl.h  -  description
                             -------------------
    begin                : Nov 25, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Solvers {
namespace Linear {   

template< typename Matrix, typename Preconditioner >
SOR< Matrix, Preconditioner > :: SOR()
: omega( 1.0 ),
  preconditioner( 0 )
{
}

template< typename Matrix, typename Preconditioner >
String SOR< Matrix, Preconditioner > :: getType() const
{
   return String( "SOR< " ) +
          this->matrix -> getType() + ", " +
          this->preconditioner -> getType() + " >";
}

template< typename Matrix,
          typename Preconditioner >
void
SOR< Matrix, Preconditioner >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   //IterativeSolver< RealType, IndexType >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "sor-omega", "Relaxation parameter of the SOR method.", 1.0 );
}

template< typename Matrix,
          typename Preconditioner >
bool
SOR< Matrix, Preconditioner >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   IterativeSolver< RealType, IndexType >::setup( parameters, prefix );
   this->setOmega( parameters.getParameter< double >( prefix + "sor-omega" ) );
   if( this->omega <= 0.0 || this->omega > 2.0 )
   {
      std::cerr << "Warning: The SOR method parameter omega is out of interval (0,2). The value is " << this->omega << " the method will not converge." << std::endl;
   }
   return true;
}


template< typename Matrix, typename Preconditioner >
void SOR< Matrix, Preconditioner > :: setOmega( const RealType& omega )
{
   this->omega = omega;
}

template< typename Matrix, typename Preconditioner >
const typename SOR< Matrix, Preconditioner > :: RealType& SOR< Matrix, Preconditioner > :: getOmega( ) const
{
   return this->omega;
}

template< typename Matrix,
          typename Preconditioner >
void SOR< Matrix, Preconditioner > :: setMatrix( MatrixPointer& matrix )
{
   this->matrix = matrix;
}

template< typename Matrix,
           typename Preconditioner >
void SOR< Matrix, Preconditioner > :: setPreconditioner( const PreconditionerType& preconditioner )
{
   this->preconditioner = &preconditioner;
}


template< typename Matrix, typename Preconditioner >
   template< typename VectorPointer, typename ResidueGetter >
bool SOR< Matrix, Preconditioner > :: solve( const VectorPointer& b,
                                                      VectorPointer& x )
{
   const IndexType size = matrix -> getRows();   

   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   RealType bNorm = b->lpNorm( ( RealType ) 2.0 );

   while( this->nextIteration() )
   {
      for( IndexType row = 0; row < size; row ++ )
         matrix->performSORIteration( *b,
                                      row,
                                      *x,
                                      this->getOmega() );
      this->setResidue( ResidueGetter::getResidue( matrix, x, b, bNorm ) );
      this->refreshSolverMonitor();
   }
   this->setResidue( ResidueGetter::getResidue( matrix, x, b, bNorm ) );
   this->refreshSolverMonitor( true );
   return this->checkConvergence();
};

template< typename Matrix, typename Preconditioner >
SOR< Matrix, Preconditioner > :: ~SOR()
{
}

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Matrices/CSR.h>
#include <TNL/Matrices/Ellpack.h>
#include <TNL/Matrices/Multidiagonal.h>

namespace TNL {
namespace Solvers {
namespace Linear {   
   
extern template class SOR< Matrices::CSR< float,  Devices::Host, int > >;
extern template class SOR< Matrices::CSR< double, Devices::Host, int > >;
extern template class SOR< Matrices::CSR< float,  Devices::Host, long int > >;
extern template class SOR< Matrices::CSR< double, Devices::Host, long int > >;

// TODO: fix this

/*extern template class SOR< Matrices::Ellpack< float,  Devices::Host, int > >;
extern template class SOR< Matrices::Ellpack< double, Devices::Host, int > >;
extern template class SOR< Matrices::Ellpack< float,  Devices::Host, long int > >;
extern template class SOR< Matrices::Ellpack< double, Devices::Host, long int > >;

extern template class SOR< Matrices::Multidiagonal< float,  Devices::Host, int > >;
extern template class SOR< Matrices::Multidiagonal< double, Devices::Host, int > >;
extern template class SOR< Matrices::Multidiagonal< float,  Devices::Host, long int > >;
extern template class SOR< Matrices::Multidiagonal< double, Devices::Host, long int > >;*/


#ifdef HAVE_CUDA
// TODO: fix this - it does not work with CUDA
/*extern template class SOR< Matrices::CSR< float,  Devices::Cuda, int > >;
extern template class SOR< Matrices::CSR< double, Devices::Cuda, int > >;
extern template class SOR< Matrices::CSR< float,  Devices::Cuda, long int > >;
extern template class SOR< Matrices::CSR< double, Devices::Cuda, long int > >;*/


/*
extern template class SOR< Matrices::Ellpack< float,  Devices::Cuda, int > >;
extern template class SOR< Matrices::Ellpack< double, Devices::Cuda, int > >;
extern template class SOR< Matrices::Ellpack< float,  Devices::Cuda, long int > >;
extern template class SOR< Matrices::Ellpack< double, Devices::Cuda, long int > >;
*/

/*
extern template class SOR< tnlMutliDiagonalMatrix< float,  Devices::Cuda, int > >;
extern template class SOR< tnlMutliDiagonalMatrix< double, Devices::Cuda, int > >;
extern template class SOR< tnlMutliDiagonalMatrix< float,  Devices::Cuda, long int > >;
extern template class SOR< tnlMutliDiagonalMatrix< double, Devices::Cuda, long int > >;
*/
#endif

} // namespace Linear
} // namespace Solvers
} // namespace TNL
