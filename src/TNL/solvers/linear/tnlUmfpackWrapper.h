

#pragma once

#ifdef HAVE_UMFPACK

#include <umfpack.h>

#include <TNL/Object.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/matrices/tnlCSRMatrix.h>
#include <TNL/solvers/preconditioners/tnlDummyPreconditioner.h>
#include <TNL/solvers/tnlIterativeSolver.h>
#include <TNL/solvers/linear/tnlLinearResidueGetter.h>


namespace TNL {

template< typename Matrix >
struct is_csr_matrix
{
    static const bool value = false;
};

template< typename Real, typename Device, typename Index >
struct is_csr_matrix< tnlCSRMatrix< Real, Device, Index > >
{
    static const bool value = true;
};


template< typename Matrix,
          typename Preconditioner = tnlDummyPreconditioner< typename Matrix :: RealType,
                                                            typename Matrix :: DeviceType,
                                                            typename Matrix :: IndexType> >
class tnlUmfpackWrapper
    : public Object,
      // just to ensure the same interface as other linear solvers
      public tnlIterativeSolver< typename Matrix::RealType,
                                 typename Matrix::IndexType >
{
public:
    typedef typename Matrix :: RealType RealType;
    typedef typename Matrix :: IndexType IndexType;
    typedef typename Matrix :: DeviceType DeviceType;
    typedef Matrix MatrixType;
    typedef Preconditioner PreconditionerType;

    tnlUmfpackWrapper()
    {
        if( ! is_csr_matrix< Matrix >::value )
            std::cerr << "The tnlUmfpackWrapper solver is available only for CSR matrices." << std::endl;
        if( std::is_same< typename Matrix::DeviceType, tnlCuda >::value )
            std::cerr << "The tnlUmfpackWrapper solver is not available on CUDA." << std::endl;
        if( ! std::is_same< RealType, double >::value )
            std::cerr << "The tnlUmfpackWrapper solver is available only for double precision." << std::endl;
        if( ! std::is_same< IndexType, int >::value )
            std::cerr << "The tnlUmfpackWrapper solver is available only for 'int' index type." << std::endl;
    }

    static void configSetup( Config::ConfigDescription& config,
                             const String& prefix = "" )
    {};

    bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" )
    {
        return false;
    };

    void setMatrix( const MatrixType& matrix )
    {};

    void setPreconditioner( const Preconditioner& preconditioner )
    {};

    template< typename Vector,
              typename ResidueGetter = tnlLinearResidueGetter< MatrixType, Vector > >
    bool solve( const Vector& b, Vector& x )
    {
        return false;
    };

};


template< typename Preconditioner >
class tnlUmfpackWrapper< tnlCSRMatrix< double, tnlHost, int >, Preconditioner >
    : public Object,
      // just to ensure the same interface as other linear solvers
      public tnlIterativeSolver< double, int >
{
public:
    typedef double RealType;
    typedef int IndexType;
    typedef tnlHost DeviceType;
    typedef tnlCSRMatrix< double, tnlHost, int > MatrixType;
    typedef Preconditioner PreconditionerType;

    tnlUmfpackWrapper();

    String getType() const;

    static void configSetup( Config::ConfigDescription& config,
                             const String& prefix = "" );

    bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" );

    void setMatrix( const MatrixType& matrix );

    void setPreconditioner( const Preconditioner& preconditioner );

    template< typename Vector,
              typename ResidueGetter = tnlLinearResidueGetter< MatrixType, Vector > >
    bool solve( const Vector& b, Vector& x );

protected:
   const MatrixType* matrix;

   const PreconditionerType* preconditioner;
};

} // namespace TNL

#include "tnlUmfpackWrapper_impl.h"


#endif
