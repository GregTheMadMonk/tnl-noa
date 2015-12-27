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

// This is from origin/mean-curvature
#include <functions/tnlConstantFunction.h>
#include <functions/tnlExpBumpFunction.h>
#include <functions/tnlSinBumpsFunction.h>
#include <functions/tnlSinWaveFunction.h>
#include <functions/initial_conditions/tnlCylinderFunction.h>
#include <functions/initial_conditions/tnlFlowerpotFunction.h>
#include <functions/initial_conditions/tnlTwinsFunction.h>
#include <functions/initial_conditions/level_set_functions/tnlBlobFunction.h>
#include <functions/initial_conditions/level_set_functions/tnlPseudoSquareFunction.h>

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
   config.addRequiredEntry< tnlString >( prefix + "test-function", "Testing function." );
      config.addEntryEnum( "constant" );
      config.addEntryEnum( "exp-bump" );
      config.addEntryEnum( "sin-wave" );
      config.addEntryEnum( "sin-bumps" );
      config.addEntryEnum( "cylinder" );
      config.addEntryEnum( "flowerpot" );
      config.addEntryEnum( "twins" );
      config.addEntryEnum( "pseudoSquare" );
      config.addEntryEnum( "blob" );
   config.addEntry     < double >( prefix + "constant", "Value of the constant function.", 0.0 );
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
   config.addEntry     < double >( prefix + "diameter", "Diameter for the cylinder, flowerpot test functions.", 1.0 );
  config.addEntry     < double >( prefix + "height", "Height of zero-level-set function for the blob, pseudosquare test functions.", 1.0 );
   config.addEntry     < tnlString >( prefix + "time-dependence", "Time dependence of the test function.", "none" );
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
setupFunction( const tnlParameterContainer& parameters,
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
   cout << "Test function setup ... " << endl;
   const tnlString& timeDependence =
            parameters.getParameter< tnlString >(
                     prefix +
                     "time-dependence" );
   cout << "Time dependence ... " << timeDependence << endl;
   if( timeDependence == "none" )
      this->timeDependence = none;
   if( timeDependence == "linear" )
      this->timeDependence = linear;
   if( timeDependence == "quadratic" )
      this->timeDependence = quadratic;
   if( timeDependence == "cosine" )
      this->timeDependence = cosine;

   this->timeScale = parameters.getParameter< double >( prefix + "time-scale" );

   const tnlString& testFunction = parameters.getParameter< tnlString >( prefix + "test-function" );
   cout << "Test function ... " << testFunction << endl;
   if( testFunction == "constant" )
   {
      typedef tnlConstantFunction< Dimensions, Real > FunctionType;
      functionType = constant;
      return setupFunction< FunctionType >( parameters );
   }
   if( testFunction == "exp-bump" )
   {
      typedef tnlExpBumpFunction< Dimensions, Real > FunctionType;
      functionType = expBump;
      return setupFunction< FunctionType >( parameters );
   }
   if( testFunction == "sin-bumps" )
   {
      typedef tnlSinBumpsFunction< Dimensions, Real > FunctionType;
      functionType = sinBumps;
      return setupFunction< FunctionType >( parameters );
   }
   if( testFunction == "sin-wave" )
   {
      typedef tnlSinWaveFunction< Dimensions, Real > FunctionType;
      functionType = sinWave;
      return setupFunction< FunctionType >( parameters );
   }
   if( testFunction == "cylinder" )
   {
      typedef tnlCylinderFunction< Dimensions, Real > FunctionType;
      functionType = cylinder;
      return setupFunction< FunctionType >( parameters );
   }
   if( testFunction == "flowerpot" )
   {
      typedef tnlFlowerpotFunction< Dimensions, Real > FunctionType;
      functionType = flowerpot;
      return setupFunction< FunctionType >( parameters );
   }
   if( testFunction == "twins" )
   {
      typedef tnlTwinsFunction< Dimensions, Real > FunctionType;
      functionType = twins;
      return setupFunction< FunctionType >( parameters );
   }
   if( testFunction == "pseudoSquare" )
   {
      typedef tnlPseudoSquareFunction< Dimensions, Real > FunctionType;
      functionType = pseudoSquare;
      return setupFunction< FunctionType >( parameters );
   }
   if( testFunction == "blob" )
   {
      typedef tnlBlobFunction< Dimensions, Real > FunctionType;
      functionType = blob;
      return setupFunction< FunctionType >( parameters );
   }
   cerr << "Unknown function " << testFunction << endl;
   return false;
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
const tnlTestFunction< FunctionDimensions, Real, Device >&
tnlTestFunction< FunctionDimensions, Real, Device >::
operator = ( const tnlTestFunction& function )
{
   /*****
    * TODO: if the function is on the device we cannot do the following
    */
   abort();
   this->functionType   = function.functionType;
   this->timeDependence = function.timeDependence;
   this->timeScale      = function.timeScale;

   this->deleteFunctions();

   switch( this->functionType )
   {
      case constant:
         this->copyFunction< tnlConstantFunction< FunctionDimensions, Real > >( function.function );
         break;
      case expBump:
         this->copyFunction< tnlExpBumpFunction< FunctionDimensions, Real > >( function.function );
         break;
      case sinBumps:
         this->copyFunction< tnlSinBumpsFunction< FunctionDimensions, Real > >( function.function );
         break;
      case sinWave:
         this->copyFunction< tnlSinWaveFunction< FunctionDimensions, Real > >( function.function );
         break;
      case cylinder:
         this->copyFunction< tnlCylinderFunction< FunctionDimensions, Real > >( function.function );
         break;
      case flowerpot:
         this->copyFunction< tnlFlowerpotFunction< FunctionDimensions, Real > >( function.function );
         break;
      case twins:
         this->copyFunction< tnlTwinsFunction< FunctionDimensions, Real > >( function.function );
         break;
      case pseudoSquare:
         this->copyFunction< tnlPseudoSquareFunction< FunctionDimensions, Real > >( function.function );
         break;
      case blob:
         this->copyFunction< tnlBlobFunction< FunctionDimensions, Real > >( function.function );
         break;
      default:
         tnlAssert( false, );
         break;
   }

}

template< int FunctionDimensions,
          typename Real,
          typename Device >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
tnlTestFunction< FunctionDimensions, Real, Device >::
getValue( const VertexType& vertex,
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
      case cosine:
         scale = cos( this->timeScale * time );
         break;
   }
   switch( functionType )
   {
      case constant:
         return scale * ( ( tnlConstantFunction< Dimensions, Real >* ) function )->
                   getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      case expBump:
         return scale * ( ( tnlExpBumpFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      case sinBumps:
         return scale * ( ( tnlSinBumpsFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      case sinWave:
         return scale * ( ( tnlSinWaveFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      case cylinder:
         return scale * ( ( tnlCylinderFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      case flowerpot:
         return scale * ( ( tnlFlowerpotFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      case twins:
         return scale * ( ( tnlTwinsFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      case pseudoSquare:
         return scale * ( ( tnlPseudoSquareFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      case blob:
         return scale * ( ( tnlBlobFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      default:
         return 0.0;
   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
tnlTestFunction< FunctionDimensions, Real, Device >::
getTimeDerivative( const VertexType& vertex,
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
      case cosine:
         scale = -this->timeScale * sin( this->timeScale * time );
         break;
   }
   switch( functionType )
   {
      case constant:
         return scale * ( ( tnlConstantFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      case expBump:
         return scale * ( ( tnlExpBumpFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      case sinBumps:
         return scale * ( ( tnlSinBumpsFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      case sinWave:
         return scale * ( ( tnlSinWaveFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;
      case cylinder:
         return scale * ( ( tnlCylinderFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;
      case flowerpot:
         return scale * ( ( tnlFlowerpotFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;
      case twins:
         return scale * ( ( tnlTwinsFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;
      case pseudoSquare:
         return scale * ( ( tnlPseudoSquareFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;
      case blob:
         return scale * ( ( tnlBlobFunction< Dimensions, Real >* ) function )->
                  getValue< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;
      default:
         return 0.0;
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
   {
      if( function )
         delete ( FunctionType * ) function;
   }
   if( Device::DeviceType == ( int ) tnlCudaDevice )
   {
      if( function )
         tnlCuda::freeFromDevice( ( FunctionType * ) function );
   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
void
tnlTestFunction< FunctionDimensions, Real, Device >::
deleteFunctions()
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
      case cylinder:
         deleteFunction< tnlCylinderFunction< Dimensions, Real> >();
         break;
      case flowerpot:
         deleteFunction< tnlFlowerpotFunction< Dimensions, Real> >();
         break;
      case twins:
         deleteFunction< tnlTwinsFunction< Dimensions, Real> >();
         break;
      case pseudoSquare:
         deleteFunction< tnlPseudoSquareFunction< Dimensions, Real> >();
         break;
      case blob:
         deleteFunction< tnlBlobFunction< Dimensions, Real> >();
         break;
   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
   template< typename FunctionType >
void
tnlTestFunction< FunctionDimensions, Real, Device >::
copyFunction( const void* function )
{
   if( Device::DeviceType == ( int ) tnlHostDevice ) 
   {
      FunctionType* f = new FunctionType;
      *f = * ( FunctionType* )function;
   }
   if( Device::DeviceType == ( int ) tnlCudaDevice )
   {
      tnlAssert( false, );
      abort();
   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
   template< typename FunctionType >
ostream&
tnlTestFunction< FunctionDimensions, Real, Device >::
printFunction( ostream& str ) const
{
   FunctionType* f = ( FunctionType* ) this->function;
   switch( Device::DeviceType )
   {
      case tnlHostDevice:
         str << *f;
         return str;
      case tnlCudaDevice:
         tnlCuda::print( f, str );
         return str;
      default:
         return str;
   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
ostream&
tnlTestFunction< FunctionDimensions, Real, Device >::
print( ostream& str ) const
{
   str << " timeDependence = " << this->timeDependence;
   str << " functionType = " << this->functionType;
   str << " function = " << this->function << "; ";
   switch( functionType )
   {
      case constant:
         return printFunction< tnlConstantFunction< Dimensions, Real> >( str );
      case expBump:
         return printFunction< tnlExpBumpFunction< Dimensions, Real> >( str );
      case sinBumps:
         return printFunction< tnlSinBumpsFunction< Dimensions, Real> >( str );
      case sinWave:
         return printFunction< tnlSinWaveFunction< Dimensions, Real> >( str );
      case cylinder:
         return printFunction< tnlCylinderFunction< Dimensions, Real> >( str );
      case flowerpot:
         return printFunction< tnlFlowerpotFunction< Dimensions, Real> >( str );
      case twins:
         return printFunction< tnlTwinsFunction< Dimensions, Real> >( str );
      case pseudoSquare:
         return printFunction< tnlPseudoSquareFunction< Dimensions, Real> >( str );
      case blob:
         return printFunction< tnlBlobFunction< Dimensions, Real> >( str );
   }
   return str;
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
tnlTestFunction< FunctionDimensions, Real, Device >::
~tnlTestFunction()
{
   deleteFunctions();
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

/*extern template class tnlTestFunction< 1, long double, tnlCuda >;
extern template class tnlTestFunction< 2, long double, tnlCuda >;
extern template class tnlTestFunction< 3, long double, tnlCuda >;*/
#endif

#endif


#endif /* TNLTESTFUNCTION_IMPL_H_ */
