/***************************************************************************
                          tnlTestFunction_impl.h  -  description
                             -------------------
    begin                : Aug 3, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLTESTFUNCTION_IMPL_H_
#define TNLTESTFUNCTION_IMPL_H_

#include <core/tnlCuda.h>
#include <functions/tnlConstantFunction.h>
#include <functions/tnlExpBumpFunction.h>
#include <functions/tnlSinBumpsFunction.h>
#include <functions/tnlSinWaveFunction.h>

#include <functions/tnlSDFParaboloid.h>
#include <functions/tnlSDFSinBumpsFunction.h>
#include <functions/tnlSDFSinWaveFunction.h>
#include <functions/tnlSDFParaboloidSDF.h>
#include <functions/tnlSDFSinBumpsFunctionSDF.h>
#include <functions/tnlSDFSinWaveFunctionSDF.h>



template< int FunctionDimensions,
          typename Real,
          typename Device >
tnlTestFunction< FunctionDimensions, Real, Device >::
tnlTestFunction()
: function( 0 ),
  timeDependence( none ),
  timeScale( 1.0 )
{
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
void
tnlTestFunction< FunctionDimensions, Real, Device >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
   config.addEntry     < tnlString >( prefix + "test-function", "Testing function.", "exp-bump" );
      config.addEntryEnum( "sin-wave" );
      config.addEntryEnum( "sin-bumps" );
      config.addEntryEnum( "exp-bump" );
      config.addEntryEnum( "sdf-sin-wave" );
      config.addEntryEnum( "sdf-sin-bumps" );
      config.addEntryEnum( "sdf-para" );
      config.addEntryEnum( "sdf-sin-wave-sdf" );
      config.addEntryEnum( "sdf-sin-bumps-sdf" );
      config.addEntryEnum( "sdf-para-sdf" );
   config.addEntry     < double >( prefix + "value", "Value of the constant function.", 0.0 );
   config.addEntry     < double >( prefix + "wave-length", "Wave length of the sine based test functions.", 1.0 );
   config.addEntry     < double >( prefix + "wave-length-x", "Wave length of the sine based test functions.", 1.0 );
   config.addEntry     < double >( prefix + "wave-length-y", "Wave length of the sine based test functions.", 1.0 );
   config.addEntry     < double >( prefix + "wave-length-z", "Wave length of the sine based test functions.", 1.0 );
   config.addEntry     < double >( prefix + "phase", "Phase of the sine based test functions.", 0.0 );
   config.addEntry     < double >( prefix + "phase-x", "Phase of the sine based test functions.", 0.0 );
   config.addEntry     < double >( prefix + "phase-y", "Phase of the sine based test functions.", 0.0 );
   config.addEntry     < double >( prefix + "phase-z", "Phase of the sine based test functions.", 0.0 );
   config.addEntry     < double >( prefix + "amplitude", "Amplitude length of the sine based test functions.", 1.0 );
   config.addEntry     < double >( prefix + "waves-number", "Cut-off for the sine based test functions.", 0.0 );
   config.addEntry     < double >( prefix + "waves-number-x", "Cut-off for the sine based test functions.", 0.0 );
   config.addEntry     < double >( prefix + "waves-number-y", "Cut-off for the sine based test functions.", 0.0 );
   config.addEntry     < double >( prefix + "waves-number-z", "Cut-off for the sine based test functions.", 0.0 );
   config.addEntry     < double >( prefix + "sigma", "Sigma for the exp based test functions.", 1.0 );
   config.addEntry     < double >( prefix + "offset", "Offset for paraboloids.", 1.0 );
   config.addEntry     < double >( prefix + "coefficient", "Coefficient for paraboloids.", 1.0 );
   config.addEntry     < double >( prefix + "x-centre", "x-centre for paraboloids.", 0.0 );
   config.addEntry     < double >( prefix + "y-centre", "y-centre for paraboloids.", 0.0 );
   config.addEntry     < double >( prefix + "z-centre", "z-centre for paraboloids.", 0.0 );
   config.addEntry     < tnlString >( prefix + "test-function-time-dependence", "Time dependence of the test function.", "none" );
      config.addEntryEnum( "none" );
      config.addEntryEnum( "linear" );
      config.addEntryEnum( "quadratic" );
      config.addEntryEnum( "cosine" );
   config.addEntry     < double >( prefix + "time-scale", "Time scaling for the time dependency of the test function.", 1.0 );

}

template< int FunctionDimensions,
          typename Real,
          typename Device >
   template< typename FunctionType >
bool
tnlTestFunction< FunctionDimensions, Real, Device >::
initFunction( const tnlParameterContainer& parameters,
              const tnlString& prefix )
{
   FunctionType* auxFunction = new FunctionType;
   if( ! auxFunction->setup( parameters, prefix ) )
   {
      delete auxFunction;
      return false;
   }

   if( Device::DeviceType == ( int ) tnlHostDevice )
   {
      this->function = auxFunction;
   }
   if( Device::DeviceType == ( int ) tnlCudaDevice )
   {
      this->function = tnlCuda::passToDevice( *auxFunction );
      delete auxFunction;
      if( ! checkCudaDevice )
         return false;
   }
   return true;
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
bool
tnlTestFunction< FunctionDimensions, Real, Device >::
setup( const tnlParameterContainer& parameters,
      const tnlString& prefix )
{
   const tnlString& timeDependence =
            parameters.GetParameter< tnlString >(
                     prefix +
                     "test-function-time-dependence" );
   if( timeDependence == "none" )
      this->timeDependence = none;
   if( timeDependence == "linear" )
      this->timeDependence = linear;
   if( timeDependence == "quadratic" )
      this->timeDependence = quadratic;
   if( timeDependence == "sine" )
      this->timeDependence = sine;

   this->timeScale = parameters.GetParameter< double >( prefix + "time-scale" );

   const tnlString& testFunction = parameters.GetParameter< tnlString >( prefix + "test-function" );
   if( testFunction == "constant" )
   {
      typedef tnlConstantFunction< Dimensions, Real > FunctionType;
      functionType = constant;
      return initFunction< FunctionType >( parameters );
   }
   if( testFunction == "exp-bump" )
   {
      typedef tnlExpBumpFunction< Dimensions, Real > FunctionType;
      functionType = expBump;
      return initFunction< FunctionType >( parameters );
   }
   if( testFunction == "sin-bumps" )
   {
      typedef tnlSinBumpsFunction< Dimensions, Real > FunctionType;
      functionType = sinBumps;
      return initFunction< FunctionType >( parameters );
   }
   if( testFunction == "sin-wave" )
   {
      typedef tnlSinWaveFunction< Dimensions, Real > FunctionType;
      functionType = sinWave;
      return initFunction< FunctionType >( parameters );
   }

   if( testFunction == "sdf-para" )
   {
      typedef tnlSDFParaboloid< Dimensions, Real > FunctionType;
      functionType = sdfParaboloid;
      return initFunction< FunctionType >( parameters );
   }
   if( testFunction == "sdf-sin-bumps" )
   {
      typedef tnlSDFSinBumpsFunction< Dimensions, Real > FunctionType;
      functionType = sdfSinBumps;
      return initFunction< FunctionType >( parameters );
   }
   if( testFunction == "sdf-sin-wave" )
   {
      typedef tnlSDFSinWaveFunction< Dimensions, Real > FunctionType;
      functionType = sdfSinWave;
      return initFunction< FunctionType >( parameters );
   }
   if( testFunction == "sdf-para-sdf" )
   {
      typedef tnlSDFParaboloidSDF< Dimensions, Real > FunctionType;
      functionType = sdfParaboloidSDF;
      return initFunction< FunctionType >( parameters );
   }
   if( testFunction == "sdf-sin-bumps-sdf" )
   {
      typedef tnlSDFSinBumpsFunctionSDF< Dimensions, Real > FunctionType;
      functionType = sdfSinBumpsSDF;
      return initFunction< FunctionType >( parameters );
   }
   if( testFunction == "sdf-sin-wave-sdf" )
   {
      typedef tnlSDFSinWaveFunctionSDF< Dimensions, Real > FunctionType;
      functionType = sdfSinWaveSDF;
      return initFunction< FunctionType >( parameters );
   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Real
tnlTestFunction< FunctionDimensions, Real, Device >::
getValue( const Vertex& vertex,
          const Real& time ) const
{
   Real scale( 1.0 );
   switch( this->timeDependence )
   {
      case none:
         break;
      case linear:
         scale = 1.0 - this->timeScale * time;
         break;
      case quadratic:
         scale = this->timeScale * time;
         scale *= scale;
         scale = 1.0 - scale;
         break;
      case sine:
         scale = 1.0 - sin( this->timeScale * time );
         break;
   }
   //cout << "scale = " << scale << " time= " << time << " timeScale = " << timeScale << " timeDependence = " << ( int ) timeDependence << endl;
   switch( functionType )
   {
      case constant:
         return scale * ( ( tnlConstantFunction< Dimensions, Real >* ) function )->
                   getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case expBump:
         return scale * ( ( tnlExpBumpFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case sinBumps:
         return scale * ( ( tnlSinBumpsFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case sinWave:
         return scale * ( ( tnlSinWaveFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case sdfParaboloid:
         return scale * ( ( tnlSDFParaboloid< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case sdfSinBumps:
         return scale * ( ( tnlSDFSinBumpsFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case sdfSinWave:
         return scale * ( ( tnlSDFSinWaveFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case sdfParaboloidSDF:
         return scale * ( ( tnlSDFParaboloidSDF< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case sdfSinBumpsSDF:
         return scale * ( ( tnlSDFSinBumpsFunctionSDF< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case sdfSinWaveSDF:
         return scale * ( ( tnlSDFSinWaveFunctionSDF< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      default:
         return 0.0;
         break;
   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Real
tnlTestFunction< FunctionDimensions, Real, Device >::
getTimeDerivative( const Vertex& vertex,
                   const Real& time ) const
{
   Real scale( 0.0 );
   switch( timeDependence )
   {
      case none:
         break;
      case linear:
         scale = -this->timeScale;
         break;
      case quadratic:
         scale = -2.0 * this->timeScale * this->timeScale * time;
         break;
      case sine:
         scale = -this->timeScale * cos( this->timeScale * time );
         break;
   }
   switch( functionType )
   {
      case constant:
         return scale * ( ( tnlConstantFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case expBump:
         return scale * ( ( tnlExpBumpFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case sinBumps:
         return scale * ( ( tnlSinBumpsFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case sinWave:
         return scale * ( ( tnlSinWaveFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case sdfParaboloid:
         return scale * ( ( tnlSDFParaboloid< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case sdfSinBumps:
         return scale * ( ( tnlSDFSinBumpsFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case sdfSinWave:
         return scale * ( ( tnlSDFSinWaveFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case sdfParaboloidSDF:
         return scale * ( ( tnlSDFParaboloidSDF< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case sdfSinBumpsSDF:
         return scale * ( ( tnlSDFSinBumpsFunctionSDF< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      case sdfSinWaveSDF:
         return scale * ( ( tnlSDFSinWaveFunctionSDF< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder, Vertex >( vertex, time );
         break;
      default:
         return 0.0;
         break;
   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
   template< typename FunctionType >
void
tnlTestFunction< FunctionDimensions, Real, Device >::
deleteFunction()
{
   if( Device::DeviceType == ( int ) tnlHostDevice )
      delete ( FunctionType * ) function;
   if( Device::DeviceType == ( int ) tnlCudaDevice )
      tnlCuda::freeFromDevice( ( FunctionType * ) function );
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
tnlTestFunction< FunctionDimensions, Real, Device >::
~tnlTestFunction()
{
   switch( functionType )
   {
      case constant:
         deleteFunction< tnlConstantFunction< Dimensions, Real> >();
         break;
      case expBump:
         deleteFunction< tnlExpBumpFunction< Dimensions, Real> >();
         break;
      case sinBumps:
         deleteFunction< tnlSinBumpsFunction< Dimensions, Real> >();
         break;
      case sinWave:
         deleteFunction< tnlSinWaveFunction< Dimensions, Real> >();
         break;
   }
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template class tnlTestFunction< 1, float, tnlHost >;
extern template class tnlTestFunction< 2, float, tnlHost >;
extern template class tnlTestFunction< 3, float, tnlHost >;

extern template class tnlTestFunction< 1, double, tnlHost >;
extern template class tnlTestFunction< 2, double, tnlHost >;
extern template class tnlTestFunction< 3, double, tnlHost >;

extern template class tnlTestFunction< 1, long double, tnlHost >;
extern template class tnlTestFunction< 2, long double, tnlHost >;
extern template class tnlTestFunction< 3, long double, tnlHost >;

#ifdef HAVE_CUDA
extern template class tnlTestFunction< 1, float, tnlCuda>;
extern template class tnlTestFunction< 2, float, tnlCuda >;
extern template class tnlTestFunction< 3, float, tnlCuda >;

extern template class tnlTestFunction< 1, double, tnlCuda >;
extern template class tnlTestFunction< 2, double, tnlCuda >;
extern template class tnlTestFunction< 3, double, tnlCuda >;

extern template class tnlTestFunction< 1, long double, tnlCuda >;
extern template class tnlTestFunction< 2, long double, tnlCuda >;
extern template class tnlTestFunction< 3, long double, tnlCuda >;
#endif

#endif


#endif /* TNLTESTFUNCTION_IMPL_H_ */
