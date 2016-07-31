
#pragma once

#include <TNL/Object.h>
#include <TNL/Vectors/Vector.h>

namespace TNL {

template< typename Real, typename Device, typename Index >
class tnlDiagonalPreconditioner
{
   public:
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef Vectors::Vector< Real, Device, Index > VectorType;

   template< typename Matrix >
   void update( const Matrix& matrix );

   template< typename Vector1, typename Vector2 >
   bool solve( const Vector1& b, Vector2& x ) const;

   String getType() const
   {
      return String( "tnlDiagonalPreconditioner" );
   }

   protected:
   VectorType diagonal;
};

} // namespace TNL

#include <TNL/solvers/preconditioners/tnlDiagonalPreconditioner_impl.h>
