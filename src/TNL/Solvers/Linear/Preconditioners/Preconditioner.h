/***************************************************************************
                          Preconditioner.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <type_traits>  // std::add_const_t
#include <memory>  // std::shared_ptr

#include <TNL/Containers/VectorView.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Solvers/Linear/Traits.h>

namespace TNL {
namespace Solvers {
namespace Linear {
/**
 * \brief Namespace for preconditioners for linear system solvers.
 */
namespace Preconditioners {

template< typename Matrix >
class Preconditioner
{
public:
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorViewType = typename Traits< Matrix >::VectorViewType;
   using ConstVectorViewType = typename Traits< Matrix >::ConstVectorViewType;
   using MatrixType = Matrix;
   using MatrixPointer = std::shared_ptr< std::add_const_t< MatrixType > >;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" )
   {}

   virtual bool setup( const Config::ParameterContainer& parameters,
                       const String& prefix = "" )
   {
      return true;
   }

   virtual void update( const MatrixPointer& matrixPointer )
   {}

   virtual void solve( ConstVectorViewType b, VectorViewType x ) const
   {
      throw std::logic_error("The solve() method of a dummy preconditioner should not be called.");
   }

   virtual ~Preconditioner() {}
};

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL
