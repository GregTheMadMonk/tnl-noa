/***************************************************************************
                          ExpBump.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Functions/Domain.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Functions {
namespace Analytic {   

template< typename Real,
          int Dimension >
class CylinderBase : public Domain< Dimension, SpaceDomain >
{
   public:

      typedef Real RealType;

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

      void setDiameter( const RealType& sigma );

      const RealType& getDiameter() const;

   protected:

      RealType diameter;
};

template< int Dimension,
          typename Real >
class Cylinder
{
};

template< typename Real >
class Cylinder< 1, Real > : public CylinderBase< Real, 1 >
{
   public:

      enum { Dimension = 1 };
      typedef Real RealType;
      typedef Containers::StaticVector< Dimension, Real > PointType;

      Cylinder();

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0,
                typename Point = PointType >
      __cuda_callable__
      RealType getPartialDerivative( const Point& v,
                                     const Real& time = 0.0 ) const;

      __cuda_callable__
      RealType operator()( const PointType& v,
                           const Real& time = 0.0 ) const;
 
};

template< typename Real >
class Cylinder< 2, Real > : public CylinderBase< Real, 2 >
{
   public:

      enum { Dimension = 2 };
      typedef Real RealType;
      typedef Containers::StaticVector< Dimension, Real > PointType;

      Cylinder();

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0,
                typename Point = PointType >
      __cuda_callable__
      RealType getPartialDerivative( const Point& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const PointType& v,
                           const Real& time = 0.0 ) const;
 
};

template< typename Real >
class Cylinder< 3, Real > : public CylinderBase< Real, 3 >
{
   public:

      enum { Dimension = 3 };
      typedef Real RealType;
      typedef Containers::StaticVector< Dimension, Real > PointType;

      Cylinder();

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0,
                typename Point = PointType >
      __cuda_callable__
      RealType getPartialDerivative( const Point& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const PointType& v,
                           const Real& time = 0.0 ) const;
 
};

template< int Dimension,
          typename Real >
std::ostream& operator << ( std::ostream& str, const Cylinder< Dimension, Real >& f )
{
   str << "Cylinder function.";
   return str;
}

} // namespace Analytic
} // namespace Functions
} // namespace TNL

#include <TNL/Functions/Analytic/Cylinder_impl.h>

