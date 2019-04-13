/***************************************************************************
                          CommonVectorOperations.h  -  description
                             -------------------
    begin                : Apr 12, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Device >
struct CommonVectorOperations
{
   using DeviceType = Device;
   
   template< typename Vector, typename ResultType = typename Vector::RealType >
   static ResultType getVectorMax( const Vector& v );

   template< typename Vector, typename ResultType = typename Vector::RealType >
   static ResultType getVectorMin( const Vector& v );

   template< typename Vector, typename ResultType = typename Vector::RealType >
   static ResultType getVectorAbsMax( const Vector& v );

   template< typename Vector, typename ResultType = typename Vector::RealType >
   static ResultType getVectorAbsMin( const Vector& v );

   template< typename Vector, typename ResultType = typename Vector::RealType >
   static ResultType getVectorL1Norm( const Vector& v );

   template< typename Vector, typename ResultType = typename Vector::RealType >
   static ResultType getVectorL2Norm( const Vector& v );

   template< typename Vector, typename ResultType = typename Vector::RealType, typename Scalar >
   static ResultType getVectorLpNorm( const Vector& v, const Scalar p );

   template< typename Vector, typename ResultType = typename Vector::RealType >
   static ResultType getVectorSum( const Vector& v );

   template< typename Vector1, typename Vector2, typename ResultType = typename Vector1::RealType >
   static ResultType getVectorDifferenceMax( const Vector1& v1, const Vector2& v2 );

   template< typename Vector1, typename Vector2, typename ResultType = typename Vector1::RealType >
   static ResultType getVectorDifferenceMin( const Vector1& v1, const Vector2& v2 );

   template< typename Vector1, typename Vector2, typename ResultType = typename Vector1::RealType >
   static ResultType getVectorDifferenceAbsMax( const Vector1& v1, const Vector2& v2 );

   template< typename Vector1, typename Vector2, typename ResultType = typename Vector1::RealType >
   static ResultType getVectorDifferenceAbsMin( const Vector1& v1, const Vector2& v2 );

   template< typename Vector1, typename Vector2, typename ResultType = typename Vector1::RealType >
   static ResultType getVectorDifferenceL1Norm( const Vector1& v1, const Vector2& v2 );

   template< typename Vector1, typename Vector2, typename ResultType = typename Vector1::RealType >
   static ResultType getVectorDifferenceL2Norm( const Vector1& v1, const Vector2& v2 );

   template< typename Vector1, typename Vector2, typename ResultType = typename Vector1::RealType, typename Scalar >
   static ResultType getVectorDifferenceLpNorm( const Vector1& v1, const Vector2& v2, const Scalar p );

   template< typename Vector1, typename Vector2, typename ResultType = typename Vector1::RealType >
   static ResultType getVectorDifferenceSum( const Vector1& v1, const Vector2& v2 );

   template< typename Vector1, typename Vector2, typename ResultType = typename Vector1::RealType >
   static ResultType getScalarProduct( const Vector1& v1, const Vector2& v2 );

};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Algorithms/CommonVectorOperations.hpp>
