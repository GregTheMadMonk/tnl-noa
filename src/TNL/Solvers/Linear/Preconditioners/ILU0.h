
#pragma once

#include <TNL/Object.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/CSR.h>

namespace TNL {
namespace Solvers {
namespace Linear {
namespace Preconditioners {

template< typename Real, typename Device, typename Index >
class ILU0
{};

template< typename Real, typename Index >
class ILU0< Real, Devices::Host, Index >
{
public:
   typedef Real RealType;
   typedef Devices::Host DeviceType;
   typedef Index IndexType;

   template< typename Matrix >
   void update( const Matrix& matrix );

   template< typename Vector1, typename Vector2 >
   bool solve( const Vector1& b, Vector2& x ) const;

   String getType() const
   {
      return String( "ILU0" );
   }

protected:
//   Matrices::CSR< RealType, DeviceType, IndexType > A;
   Matrices::CSR< RealType, DeviceType, IndexType > L;
   Matrices::CSR< RealType, DeviceType, IndexType > U;
};

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/Preconditioners/ILU0_impl.h>
//#include <TNL/Solvers/Linear/Preconditioners/ILU0_impl_original.h>
