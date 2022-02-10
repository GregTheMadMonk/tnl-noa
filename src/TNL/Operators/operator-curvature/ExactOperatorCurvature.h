// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/SharedVector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/tnlFunction.h>

namespace TNL {
namespace Operators {

template< typename ExactOperatorQ, int Dimension >
class ExactOperatorCurvature
{};

template< typename ExactOperatorQ >
class ExactOperatorCurvature< OperatorQ, 1 >
{
public:
   enum
   {
      Dimension = 1
   };

   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0,
             typename Function,
             typename Point,
             typename Real = typename Point::RealType >
   __cuda_callable__
   static Real
   getValue( const Function& function, const Point& v, const Real& time = 0.0, const Real& eps = 1.0 );
};

template< typename ExactOperatorQ >
class ExactOperatorCurvature< ExactOperatorQ, 2 >
{
public:
   enum
   {
      Dimension = 2
   };

   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0,
             typename Function,
             typename Point,
             typename Real = typename Point::RealType >
   __cuda_callable__
   static Real
   getValue( const Function& function, const Point& v, const Real& time = 0.0, const Real& eps = 1.0 );
};

template< typename ExactOperatorQ >
class ExactOperatorCurvature< ExactOperatorQ, 3 >
{
public:
   enum
   {
      Dimension = 3
   };

   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0,
             typename Function,
             typename Point,
             typename Real = typename Point::RealType >
   __cuda_callable__
   static Real
   getValue( const Function& function, const Point& v, const Real& time = 0.0, const Real& eps = 1.0 )
   {
      return 0;
   }
};

template< typename ExactOperatorQ, int Dimension >
class tnlFunctionType< ExactOperatorCurvature< ExactOperatorQ, Dimension > >
{
public:
   enum
   {
      Type = tnlSpaceDomain
   };
};

}  // namespace Operators
}  // namespace TNL

#include <TNL/Operators/operator-curvature/ExactOperatorCurvature_impl.h>
