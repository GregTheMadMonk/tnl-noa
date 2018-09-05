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

#include "LinearSolver.h"

#include <TNL/Matrices/CSR.h>
#include <TNL/Solvers/IterativeSolver.h>


namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
struct is_csr_matrix
{
    static constexpr bool value = false;
};

template< typename Real, typename Device, typename Index >
struct is_csr_matrix< Matrices::CSR< Real, Device, Index > >
{
    static constexpr bool value = true;
};


template< typename Matrix >
class UmfpackWrapper
: public LinearSolver< Matrix >,
  // just to ensure the same interface as other linear solvers
  public IterativeSolver< typename Matrix::RealType,
                          typename Matrix::IndexType >
{
   using Base = LinearSolver< Matrix >;
public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   // to avoid ambiguity
   using Base::configSetup;

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

   bool solve( ConstVectorViewType b, VectorViewType x ) override
   {
       return false;
   }
};


template<>
class UmfpackWrapper< Matrices::CSR< double, Devices::Host, int > >
: public LinearSolver< Matrices::CSR< double, Devices::Host, int > >,
  // just to ensure the same interface as other linear solvers
  public IterativeSolver< double, int >
{
   using Base = LinearSolver< Matrices::CSR< double, Devices::Host, int > >;
public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   String getType() const;

   bool solve( ConstVectorViewType b, VectorViewType x ) override;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include "UmfpackWrapper_impl.h"

#endif
