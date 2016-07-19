/***************************************************************************
                          tnlSinWaveFunction.h  -  description
                             -------------------
    begin                : Nov 19, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <config/tnlParameterContainer.h>
#include <core/vectors/tnlStaticVector.h>
#include <functions/tnlDomain.h>

namespace TNL {

template< int dimensions,
          typename Real = double >
class tnlSinWaveFunctionBase : public tnlDomain< dimensions, SpaceDomain >
{
   public:
 
   tnlSinWaveFunctionBase();

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   void setWaveLength( const Real& waveLength );

   Real getWaveLength() const;

   void setAmplitude( const Real& amplitude );

   Real getAmplitude() const;

   void setPhase( const Real& phase );

   Real getPhase() const;

   protected:

   Real waveLength, amplitude, phase, wavesNumber;
};

template< int Dimensions, typename Real >
class tnlSinWaveFunction
{
};

template< typename Real >
class tnlSinWaveFunction< 1, Real > : public tnlSinWaveFunctionBase< 1, Real >
{
   public:
 
      typedef Real RealType;
      typedef tnlStaticVector< 1, RealType > VertexType;

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
class tnlSinWaveFunction< 2, Real > : public tnlSinWaveFunctionBase< 2, Real >
{
   public:
 
      typedef Real RealType;
      typedef tnlStaticVector< 2, RealType > VertexType;
 
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
class tnlSinWaveFunction< 3, Real > : public tnlSinWaveFunctionBase< 3, Real >
{
   public:
 
      typedef Real RealType;
      typedef tnlStaticVector< 3, RealType > VertexType;


 
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
std::ostream& operator << ( std::ostream& str, const tnlSinWaveFunction< Dimensions, Real >& f )
{
   str << "Sin Wave. function: amplitude = " << f.getAmplitude()
       << " wavelength = " << f.getWaveLength()
       << " phase = " << f.getPhase();
   return str;
}

} // namespace TNL

#include <functions/tnlSinWaveFunction_impl.h>

