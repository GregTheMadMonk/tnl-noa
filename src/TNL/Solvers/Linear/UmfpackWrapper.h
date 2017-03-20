/***************************************************************************
                          UmfpackWrapper.h  -  description
                             -------------------
    begin                : Mar 21, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#ifdef HAVE_UMFPACK

#include <umfpack.h>

#include <TNL/Object.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Matrices/CSR.h>
#include <TNL/Solvers/Linear/Preconditioners/Dummy.h>
#include <TNL/Solvers/IterativeSolver.h>
#include <TNL/Solvers/Linear/LinearResidueGetter.h>


namespace TNL {
namespace Solvers {
namespace Linear {   

template< typename Matrix >
struct is_csr_matrix
{
    static const bool value = false;
};

template< typename Real, typename Device, typename Index >
struct is_csr_matrix< Matrices::CSR< Real, Device, Index > >
{
    static const bool value = true;
};


template< typename Matrix,
          typename Preconditioner = Preconditioners::Dummy< typename Matrix :: RealType,
                                                            typename Matrix :: DeviceType,
                                                            typename Matrix :: IndexType> >
class UmfpackWrapper
    : public Object,
      // just to ensure the same interface as other linear solvers
      public IterativeSolver< typename Matrix::RealType,
                              typename Matrix::IndexType >
{
public:
    typedef typename Matrix :: RealType RealType;
    typedef typename Matrix :: IndexType IndexType;
    typedef typename Matrix :: DeviceType DeviceType;
    typedef Matrix MatrixType;
    typedef Preconditioner PreconditionerType;
    typedef SharedPointer< const MatrixType, DeviceType > MatrixPointer;
    typedef SharedPointer< const PreconditionerType, DeviceType > PreconditionerPointer;

    UmfpackWrapper()
    {
        if( ! is_csr_matrix< Matrix >::value )
            std::cerr << "The UmfpackWrapper solver is available only for CSR matrices." << std::endl;
        if( std::is_same< typename Matrix::DeviceType, Devices::Cuda >::value )
            std::cerr << "The UmfpackWrapper solver is not available on CUDA." << std::endl;
        if( ! std::is_same< RealType, double >::value )
            std::cerr << "The UmfpackWrapper solver is available only for double precision." << std::endl;
        if( ! std::is_same< IndexType, int >::value )
            std::cerr << "The UmfpackWrapper solver is available only for 'int' index type." << std::endl;
    }

    static void configSetup( Config::ConfigDescription& config,
                             const String& prefix = "" )
    {}

    bool setup( const Config::ParameterContainer& parameters,
                const String& prefix = "" )
    {
        return false;
    }

    void setMatrix( const MatrixPointer& matrix )
    {}

    void setPreconditioner( const PreconditionerPointer& preconditioner )
    {}

    template< typename Vector,
              typename ResidueGetter = LinearResidueGetter< MatrixType, Vector > >
    bool solve( const Vector& b, Vector& x )
    {
        return false;
    }
};


template< typename Preconditioner >
class UmfpackWrapper< Matrices::CSR< double, Devices::Host, int >, Preconditioner >
    : public Object,
      // just to ensure the same interface as other linear solvers
      public IterativeSolver< double, int >
{
public:
    typedef double RealType;
    typedef int IndexType;
    typedef Devices::Host DeviceType;
    typedef Matrices::CSR< double, Devices::Host, int > MatrixType;
    typedef Preconditioner PreconditionerType;
    typedef SharedPointer< const MatrixType, DeviceType > MatrixPointer;
    typedef SharedPointer< const PreconditionerType, DeviceType > PreconditionerPointer;

    UmfpackWrapper();

    String getType() const;

    static void configSetup( Config::ConfigDescription& config,
                             const String& prefix = "" );

    bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" );

    void setMatrix( const MatrixPointer& matrix );

    void setPreconditioner( const PreconditionerPointer& preconditioner );

    template< typename Vector,
              typename ResidueGetter = LinearResidueGetter< MatrixType, Vector > >
    bool solve( const Vector& b, Vector& x );

protected:
   MatrixPointer matrix;

   PreconditionerPointer preconditioner;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include "UmfpackWrapper_impl.h"

#endif
