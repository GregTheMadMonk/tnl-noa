
#pragma once

#include <tnlObject.h>
#include <core/vectors/tnlVector.h>

namespace TNL {

template< typename Real, typename Device, typename Index >
class tnlDiagonalPreconditioner
{
   public:
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlVector< Real, Device, Index > VectorType;

   template< typename Matrix >
   void update( const Matrix& matrix );

   template< typename Vector1, typename Vector2 >
   bool solve( const Vector1& b, Vector2& x ) const;

   tnlString getType() const
   {
      return tnlString( "tnlDiagonalPreconditioner" );
   }

   protected:
   VectorType diagonal;
};

} // namespace TNL

#include <solvers/preconditioners/tnlDiagonalPreconditioner_impl.h>
