/***************************************************************************
                          tnlSinBumpsFunction.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Vectors/StaticVector.h>
#include <TNL/functions/tnlDomain.h>

namespace TNL {

template< typename Vertex >
class tnlSinBumpsFunctionBase : public tnlDomain< Vertex::size, SpaceDomain >
{
   public:
 
      typedef Vertex VertexType;
      typedef typename Vertex::RealType RealType;
      enum { Dimensions = VertexType::size };

      void setWaveLength( const VertexType& waveLength );

      const VertexType& getWaveLength() const;

      void setAmplitude( const RealType& amplitude );

      const RealType& getAmplitude() const;

      void setPhase( const VertexType& phase );

      const VertexType& getPhase() const;

   protected:

      RealType amplitude;

      VertexType waveLength, phase;
};

template< int Dimensions, typename Real >
class tnlSinBumpsFunction
{
};

template< typename Real >
class tnlSinBumpsFunction< 1, Real  > : public tnlSinBumpsFunctionBase< Vectors::StaticVector< 1, Real > >
{
   public:
 
      typedef Real RealType;
      typedef Vectors::StaticVector< 1, RealType > VertexType;


      tnlSinBumpsFunction();

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

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
                        const Real& time = 0.0 ) const;
 
};

template< typename Real >
class tnlSinBumpsFunction< 2, Real > : public tnlSinBumpsFunctionBase< Vectors::StaticVector< 2, Real > >
{
   public:

      typedef Real RealType;
      typedef Vectors::StaticVector< 2, RealType > VertexType;
 

      tnlSinBumpsFunction();

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

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
                        const Real& time = 0.0 ) const;
 
};

template< typename Real >
class tnlSinBumpsFunction< 3, Real > : public tnlSinBumpsFunctionBase< Vectors::StaticVector< 3, Real > >
{
   public:

      typedef Real RealType;
      typedef Vectors::StaticVector< 3, RealType > VertexType;

      tnlSinBumpsFunction();

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

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
                        const Real& time = 0.0 ) const;
 
};

template< int Dimensions,
          typename Real >
std::ostream& operator << ( std::ostream& str, const tnlSinBumpsFunction< Dimensions, Real >& f )
{
   str << "Sin Bumps. function: amplitude = " << f.getAmplitude()
       << " wavelength = " << f.getWaveLength()
       << " phase = " << f.getPhase();
   return str;
}

} // namespace TNL

#include <TNL/functions/tnlSinBumpsFunction_impl.h>


