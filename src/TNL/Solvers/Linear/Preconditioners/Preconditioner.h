/***************************************************************************
                          Dummy.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>  // std::add_const

#include <TNL/Containers/VectorView.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Config/ParameterContainer.h>

#include "../Traits.h"

namespace TNL {
namespace Solvers {
namespace Linear {
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
   using MatrixPointer = Pointers::SharedPointer< typename std::add_const< MatrixType >::type >;

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
      TNL_ASSERT_TRUE( false, "The solve() method of a dummy preconditioner should not be called." );
   }

   String getType() const
   {
      return String( "Preconditioner" );
   }

   virtual ~Preconditioner() {}
};

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL
