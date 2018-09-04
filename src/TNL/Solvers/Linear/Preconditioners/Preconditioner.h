/***************************************************************************
                          Dummy.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/VectorView.h>
#include <TNL/SharedPointer.h>
#include <TNL/Config/ParameterContainer.h>

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
   using VectorViewType = Containers::VectorView< RealType, DeviceType, IndexType >;
   using ConstVectorViewType = Containers::VectorView< typename std::add_const< RealType >::type, DeviceType, IndexType >;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" )
   {}

   virtual bool setup( const Config::ParameterContainer& parameters,
                       const String& prefix = "" )
   {
      return true;
   }

   virtual void update( const Matrix& matrix )
   {}

   virtual bool solve( ConstVectorViewType b, VectorViewType x ) const
   {
      TNL_ASSERT_TRUE( false, "The solve() method of a dummy preconditioner should not be called." );
      return true;
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
