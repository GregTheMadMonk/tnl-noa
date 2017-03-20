/***************************************************************************
                          Paraboloid.h  -  description
                             -------------------
    begin                : Oct 13, 2014
    copyright            : (C) 2014 by Tomas Sobotik

 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Functions/Domain.h>

namespace TNL {
   namespace Functions {
      namespace Analytic {

template< int dimensions,
          typename Real = double >
class ParaboloidBase : public Functions::Domain< dimensions, SpaceDomain >
{
   public:

   ParaboloidBase();

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   void setXCenter( const Real& waveLength );

   Real getXCenter() const;

   void setYCenter( const Real& waveLength );

   Real getYCenter() const;

   void setZCenter( const Real& waveLength );

   Real getZCenter() const;

   void setCoefficient( const Real& coefficient );

   Real getCoefficient() const;

   void setOffset( const Real& offset );

   Real getOffset() const;

   protected:

   Real xCenter, yCenter, zCenter, coefficient, radius;
};

template< int Dimensions, typename Real >
class Paraboloid
{
};

template< typename Real >
class Paraboloid< 1, Real > : public ParaboloidBase< 1, Real >
{
   public:

      typedef Real RealType;
      typedef Containers::StaticVector< 1, RealType > VertexType;

#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
#endif
      __cuda_callable__
      RealType getPartialDerivative( const VertexType& v,
                                     const Real& time = 0.0 ) const;

      __cuda_callable__
      RealType operator()( const VertexType& v,
                           const Real& time = 0.0 ) const
      {
         return this->getPartialDerivative< 0, 0, 0 >( v, time );
      }

};

template< typename Real >
class Paraboloid< 2, Real > : public ParaboloidBase< 2, Real >
{
   public:

      typedef Real RealType;
      typedef Containers::StaticVector< 2, RealType > VertexType;

#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
#endif
      __cuda_callable__
      RealType getPartialDerivative( const VertexType& v,
                                     const Real& time = 0.0 ) const;

      __cuda_callable__
      RealType operator()( const VertexType& v,
                           const Real& time = 0.0 ) const
      {
         return this->getPartialDerivative< 0, 0, 0 >( v, time );
      }

};

template< typename Real >
class Paraboloid< 3, Real > : public ParaboloidBase< 3, Real >
{
   public:

      typedef Real RealType;
      typedef Containers::StaticVector< 3, RealType > VertexType;



#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
#endif
      __cuda_callable__
      RealType getPartialDerivative( const VertexType& v,
                         const Real& time = 0.0 ) const;

      __cuda_callable__
      RealType operator()( const VertexType& v,
                           const Real& time = 0.0 ) const
      {
         return this->getPartialDerivative< 0, 0, 0 >( v, time );
      }

};

template< int Dimensions,
          typename Real >
std::ostream& operator << ( std::ostream& str, const Paraboloid< Dimensions, Real >& f )
{
   str << "SDF Paraboloid function: amplitude = " << f.getCoefficient()
       << " offset = " << f.getOffset();
   return str;
}
        
      } // namespace Analytic
   } // namespace Functions
} // namespace TNL

#include <TNL/Functions/Analytic/Paraboloid_impl.h>

