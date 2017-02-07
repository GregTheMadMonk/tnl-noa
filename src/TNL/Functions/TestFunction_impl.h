/***************************************************************************
                          TestFunction_impl.h  -  description
                             -------------------
    begin                : Aug 3, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Cuda.h>
#include <TNL/Functions/Analytic/Constant.h>
#include <TNL/Functions/Analytic/ExpBump.h>
#include <TNL/Functions/Analytic/SinBumps.h>
#include <TNL/Functions/Analytic/SinWave.h>
#include <TNL/Functions/Analytic/Constant.h>
#include <TNL/Functions/Analytic/ExpBump.h>
#include <TNL/Functions/Analytic/SinBumps.h>
#include <TNL/Functions/Analytic/SinWave.h>
#include <TNL/Functions/Analytic/Cylinder.h>
#include <TNL/Functions/Analytic/Flowerpot.h>
#include <TNL/Functions/Analytic/Twins.h>
#include <TNL/Functions/Analytic/Blob.h>
#include <TNL/Functions/Analytic/PseudoSquare.h>
#include <TNL/Functions/Analytic/Paraboloid.h>
/****
 * The signed distance test functions
 */
#include <TNL/Functions/Analytic/SinBumpsSDF.h>
#include <TNL/Functions/Analytic/SinWaveSDF.h>
#include <TNL/Functions/Analytic/ParaboloidSDF.h>

#include <TNL/Operators/Analytic/Identity.h>
#include <TNL/Operators/Analytic/Heaviside.h>

namespace TNL {
namespace Functions {   


template< int FunctionDimensions,
          typename Real,
          typename Device >
TestFunction< FunctionDimensions, Real, Device >::
TestFunction()
: function( 0 ),
  operator_( 0 ),
  timeDependence( none ),
  timeScale( 1.0 )
{
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
void
TestFunction< FunctionDimensions, Real, Device >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   config.addRequiredEntry< String >( prefix + "test-function", "Testing function." );
      config.addEntryEnum( "constant" );
      config.addEntryEnum( "paraboloid" );
      config.addEntryEnum( "exp-bump" );
      config.addEntryEnum( "sin-wave" );
      config.addEntryEnum( "sin-bumps" );
      config.addEntryEnum( "cylinder" );
      config.addEntryEnum( "flowerpot" );
      config.addEntryEnum( "twins" );
      config.addEntryEnum( "pseudoSquare" );
      config.addEntryEnum( "blob" );
      config.addEntryEnum( "paraboloid-sdf" );      
      config.addEntryEnum( "sin-wave-sdf" );
      config.addEntryEnum( "sin-bumps-sdf" );
      config.addEntryEnum( "heaviside-of-paraboloid" );

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
	config.addEntry     < double >( prefix + "radius", "Radius for paraboloids.", 1.0 );
   config.addEntry     < double >( prefix + "coefficient", "Coefficient for paraboloids.", 1.0 );
   config.addEntry     < double >( prefix + "x-center", "x-center for paraboloids.", 0.0 );
   config.addEntry     < double >( prefix + "y-center", "y-center for paraboloids.", 0.0 );
   config.addEntry     < double >( prefix + "z-center", "z-center for paraboloids.", 0.0 );
   config.addEntry     < double >( prefix + "diameter", "Diameter for the cylinder, flowerpot test functions.", 1.0 );
  config.addEntry     < double >( prefix + "height", "Height of zero-level-set function for the blob, pseudosquare test functions.", 1.0 );
   config.addEntry     < String >( prefix + "time-dependence", "Time dependence of the test function.", "none" );
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
TestFunction< FunctionDimensions, Real, Device >::
setupFunction( const Config::ParameterContainer& parameters,
               const String& prefix )
{
   FunctionType* auxFunction = new FunctionType;
   if( ! auxFunction->setup( parameters, prefix ) )
   {
      delete auxFunction;
      return false;
   }

   if( std::is_same< Device, Devices::Host >::value )
   {
      this->function = auxFunction;
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
      this->function = Devices::Cuda::passToDevice( *auxFunction );
      delete auxFunction;
      if( ! checkCudaDevice )
         return false;
   }
   return true;
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
   template< typename OperatorType >
bool
TestFunction< FunctionDimensions, Real, Device >::
setupOperator( const Config::ParameterContainer& parameters,
               const String& prefix )
{
   OperatorType* auxOperator = new OperatorType;
   if( ! auxOperator->setup( parameters, prefix ) )
   {
      delete auxOperator;
      return false;
   }

   if( std::is_same< Device, Devices::Host >::value )
   {
      this->operator_ = auxOperator;
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
      this->operator_ = Devices::Cuda::passToDevice( *auxOperator );
      delete auxOperator;
      if( ! checkCudaDevice )
         return false;
   }
   return true;
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
bool
TestFunction< FunctionDimensions, Real, Device >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   using namespace TNL::Functions::Analytic;
   using namespace TNL::Operators::Analytic;
   std::cout << "Test function setup ... " << std::endl;
   const String& timeDependence =
            parameters.getParameter< String >(
                     prefix +
                     "time-dependence" );
  std::cout << "Time dependence ... " << timeDependence << std::endl;
   if( timeDependence == "none" )
      this->timeDependence = none;
   if( timeDependence == "linear" )
      this->timeDependence = linear;
   if( timeDependence == "quadratic" )
      this->timeDependence = quadratic;
   if( timeDependence == "cosine" )
      this->timeDependence = cosine;

   this->timeScale = parameters.getParameter< double >( prefix + "time-scale" );

   const String& testFunction = parameters.getParameter< String >( prefix + "test-function" );
   std::cout << "Test function ... " << testFunction << std::endl;
   if( testFunction == "constant" )
   {
      typedef Constant< Dimensions, Real > FunctionType;
      typedef Identity< FunctionType > OperatorType;
      functionType = constant;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "paraboloid" )
   {
      typedef Paraboloid< Dimensions, Real > FunctionType;
      typedef Identity< FunctionType > OperatorType;
      functionType = paraboloid;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }   
   if( testFunction == "exp-bump" )
   {
      typedef ExpBump< Dimensions, Real > FunctionType;
      typedef Identity< FunctionType > OperatorType;
      functionType = expBump;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "sin-bumps" )
   {
      typedef SinBumps< Dimensions, Real > FunctionType;
      typedef Identity< FunctionType > OperatorType;
      functionType = sinBumps;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "sin-wave" )
   {
      typedef SinWave< Dimensions, Real > FunctionType;
      typedef Identity< FunctionType > OperatorType;
      functionType = sinWave;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "cylinder" )
   {
      typedef Cylinder< Dimensions, Real > FunctionType;
      typedef Identity< FunctionType > OperatorType;
      functionType = cylinder;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "flowerpot" )
   {
      typedef Flowerpot< Dimensions, Real > FunctionType;
      typedef Identity< FunctionType > OperatorType;
      functionType = flowerpot;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "twins" )
   {
      typedef Twins< Dimensions, Real > FunctionType;
      typedef Identity< FunctionType > OperatorType;
      functionType = twins;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "pseudoSquare" )
   {
      typedef PseudoSquare< Dimensions, Real > FunctionType;
      typedef Identity< FunctionType > OperatorType;
      functionType = pseudoSquare;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "blob" )
   {
      typedef Blob< Dimensions, Real > FunctionType;
      typedef Identity< FunctionType > OperatorType;
      functionType = blob;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "paraboloid-sdf" )
   {
      typedef ParaboloidSDF< Dimensions, Real > FunctionType;
      typedef Identity< FunctionType > OperatorType;
      functionType = paraboloidSDF;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }   
   if( testFunction == "sin-bumps-sdf" )
   {
      typedef SinBumpsSDF< Dimensions, Real > FunctionType;
      typedef Identity< FunctionType > OperatorType;
      functionType = sinBumpsSDF;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }   
   if( testFunction == "sin-wave-sdf" )
   {
      typedef SinWaveSDF< Dimensions, Real > FunctionType;
      typedef Identity< FunctionType > OperatorType;
      functionType = sinWaveSDF;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "heaviside-of-paraboloid" )
   {
      typedef Paraboloid< Dimensions, Real > FunctionType;
      typedef Heaviside< FunctionType > OperatorType;
      functionType = paraboloid;
      operatorType = heaviside;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   std::cerr << "Unknown function " << testFunction << std::endl;
   return false;
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
const TestFunction< FunctionDimensions, Real, Device >&
TestFunction< FunctionDimensions, Real, Device >::
operator = ( const TestFunction& function )
{
   /*****
    * TODO: if the function is on the device we cannot do the following
    */
   abort();
   using namespace TNL::Functions::Analytic;
   this->functionType   = function.functionType;
   this->timeDependence = function.timeDependence;
   this->timeScale      = function.timeScale;

   this->deleteFunctions();

   switch( this->functionType )
   {
      case constant:
         this->copyFunction< Constant< FunctionDimensions, Real > >( function.function );
         break;
      case expBump:
         this->copyFunction< ExpBump< FunctionDimensions, Real > >( function.function );
         break;
      case sinBumps:
         this->copyFunction< SinBumps< FunctionDimensions, Real > >( function.function );
         break;
      case sinWave:
         this->copyFunction< SinWave< FunctionDimensions, Real > >( function.function );
         break;
      case cylinder:
         this->copyFunction< Cylinder< FunctionDimensions, Real > >( function.function );
         break;
      case flowerpot:
         this->copyFunction< Flowerpot< FunctionDimensions, Real > >( function.function );
         break;
      case twins:
         this->copyFunction< Twins< FunctionDimensions, Real > >( function.function );
         break;
      case pseudoSquare:
         this->copyFunction< PseudoSquare< FunctionDimensions, Real > >( function.function );
         break;
      case blob:
         this->copyFunction< Blob< FunctionDimensions, Real > >( function.function );
         break;

      case paraboloidSDF:
         this->copyFunction< Paraboloid< FunctionDimensions, Real > >( function.function );
         break;
      case sinBumpsSDF:
         this->copyFunction< SinBumpsSDF< FunctionDimensions, Real > >( function.function );
         break;
      case sinWaveSDF:
         this->copyFunction< SinWaveSDF< FunctionDimensions, Real > >( function.function );
         break;
      default:
         TNL_ASSERT( false, );
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
TestFunction< FunctionDimensions, Real, Device >::
getPartialDerivative( const VertexType& vertex,
                      const Real& time ) const
{
   using namespace TNL::Functions::Analytic;
   using namespace TNL::Operators::Analytic;
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
         scale = ::cos( this->timeScale * time );
         break;
   }
   switch( functionType )
   {
      case constant:
      {
         typedef Constant< Dimensions, Real > FunctionType;
         typedef Identity< FunctionType > OperatorType;

         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case paraboloid:
      {
         typedef Paraboloid< Dimensions, Real > FunctionType;
         if( operatorType == identity )
         {
            typedef Identity< FunctionType > OperatorType;

            return scale * ( ( OperatorType* ) this->operator_ )->
                      template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
         }
         if( operatorType == heaviside )
         {
            typedef Heaviside< FunctionType > OperatorType;

            return scale * ( ( OperatorType* ) this->operator_ )->
                      template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
         }
      }
      case expBump:
      {
         typedef ExpBump< Dimensions, Real > FunctionType;
         typedef Identity< FunctionType > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case sinBumps:
      {
         typedef SinBumps< Dimensions, Real > FunctionType;
         typedef Identity< FunctionType > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case sinWave:
      {
         typedef SinWave< Dimensions, Real > FunctionType;
         typedef Identity< FunctionType > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case cylinder:
      {
         typedef Cylinder< Dimensions, Real > FunctionType;
         typedef Identity< FunctionType > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case flowerpot:
      {
         typedef Flowerpot< Dimensions, Real > FunctionType;
         typedef Identity< FunctionType > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case twins:
      {
         typedef Twins< Dimensions, Real > FunctionType;
         typedef Identity< FunctionType > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case pseudoSquare:
      {
         typedef PseudoSquare< Dimensions, Real > FunctionType;
         typedef Identity< FunctionType > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case blob:
      {
         typedef Blob< Dimensions, Real > FunctionType;
         typedef Identity< FunctionType > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case sinBumpsSDF:
      {
         typedef SinBumpsSDF< Dimensions, Real > FunctionType;
         typedef Identity< FunctionType > OperatorType;
                  
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case sinWaveSDF:
      {
         typedef SinWaveSDF< Dimensions, Real > FunctionType;
         typedef Identity< FunctionType > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case paraboloidSDF:
      {
         typedef ParaboloidSDF< Dimensions, Real > FunctionType;
         typedef Identity< FunctionType > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      
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
TestFunction< FunctionDimensions, Real, Device >::
getTimeDerivative( const VertexType& vertex,
                   const Real& time ) const
{
   using namespace TNL::Functions::Analytic;
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
         scale = -this->timeScale * ::sin( this->timeScale * time );
         break;
   }
   switch( functionType )
   {
      case constant:
         return scale * ( ( Constant< Dimensions, Real >* ) function )->
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      case paraboloid:
         return scale * ( ( Paraboloid< Dimensions, Real >* ) function )->
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      case expBump:
         return scale * ( ( ExpBump< Dimensions, Real >* ) function )->
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      case sinBumps:
         return scale * ( ( SinBumps< Dimensions, Real >* ) function )->
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      case sinWave:
         return scale * ( ( SinWave< Dimensions, Real >* ) function )->
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;
      case cylinder:
         return scale * ( ( Cylinder< Dimensions, Real >* ) function )->
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;
      case flowerpot:
         return scale * ( ( Flowerpot< Dimensions, Real >* ) function )->
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;
      case twins:
         return scale * ( ( Twins< Dimensions, Real >* ) function )->
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;
      case pseudoSquare:
         return scale * ( ( PseudoSquare< Dimensions, Real >* ) function )->
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;
      case blob:
         return scale * ( ( Blob< Dimensions, Real >* ) function )->
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;


      case paraboloidSDF:
         return scale * ( ( ParaboloidSDF< Dimensions, Real >* ) function )->
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      case sinBumpsSDF:
         return scale * ( ( SinBumpsSDF< Dimensions, Real >* ) function )->
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      case sinWaveSDF:
         return scale * ( ( SinWaveSDF< Dimensions, Real >* ) function )->
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      default:
         return 0.0;
   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
   template< typename FunctionType >
void
TestFunction< FunctionDimensions, Real, Device >::
deleteFunction()
{
   if( std::is_same< Device, Devices::Host >::value )
   {
      if( function )
         delete ( FunctionType * ) function;
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
      if( function )
         Devices::Cuda::freeFromDevice( ( FunctionType * ) function );
   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
   template< typename OperatorType >
void
TestFunction< FunctionDimensions, Real, Device >::
deleteOperator()
{
   if( std::is_same< Device, Devices::Host >::value )
   {
      if( operator_ )
         delete ( OperatorType * ) operator_;
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
      if( operator_ )
         Devices::Cuda::freeFromDevice( ( OperatorType * ) operator_ );
   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
void
TestFunction< FunctionDimensions, Real, Device >::
deleteFunctions()
{
   using namespace TNL::Functions::Analytic;
   using namespace TNL::Operators::Analytic;
   switch( functionType )
   {
      case constant:
      {
         typedef Constant< Dimensions, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< FunctionType> >();
         break;
      }
      case paraboloid:
      {
         typedef Paraboloid< Dimensions, Real> FunctionType;
         deleteFunction< FunctionType >();
         if( operatorType == identity )
            deleteOperator< Identity< FunctionType> >();
         if( operatorType == heaviside )
            deleteOperator< Heaviside< FunctionType> >();
         break;
      }
      case expBump:
      {
         typedef ExpBump< Dimensions, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< FunctionType> >();
         break;
      }
      case sinBumps:
      {
         typedef SinBumps< Dimensions, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< FunctionType> >();
         break;
      }
      case sinWave:
      {
         typedef SinWave< Dimensions, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< FunctionType> >();
         break;
      }
      case cylinder:
      {
         typedef Cylinder< Dimensions, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< FunctionType> >();
         break;
      }
      case flowerpot:
      {
         typedef Flowerpot< Dimensions, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< FunctionType> >();
         break;
      }
      case twins:
      {
         typedef Twins< Dimensions, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< FunctionType> >();
         break;
      }
      case pseudoSquare:
      {
         typedef PseudoSquare< Dimensions, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< FunctionType> >();
         break;
      }
      case blob:
      {
         typedef Blob< Dimensions, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< FunctionType> >();
         break;
      }
      
      case paraboloidSDF:
      {
         typedef ParaboloidSDF< Dimensions, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< FunctionType> >();
         break;
      }
      case sinBumpsSDF:
      {
         typedef SinBumpsSDF< Dimensions, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< FunctionType> >();
         break;
      }
      case sinWaveSDF:
      {
         typedef SinWaveSDF< Dimensions, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< FunctionType> >();
         break;
      }
   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
   template< typename FunctionType >
void
TestFunction< FunctionDimensions, Real, Device >::
copyFunction( const void* function )
{
   if( std::is_same< Device, Devices::Host >::value )
   {
      FunctionType* f = new FunctionType;
      *f = * ( FunctionType* )function;
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
      TNL_ASSERT( false, );
      abort();
   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
   template< typename FunctionType >
std::ostream&
TestFunction< FunctionDimensions, Real, Device >::
printFunction( std::ostream& str ) const
{
   FunctionType* f = ( FunctionType* ) this->function;
   if( std::is_same< Device, Devices::Host >::value )
   {
      str << *f;
      return str;
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
      Devices::Cuda::print( f, str );
      return str;
   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
std::ostream&
TestFunction< FunctionDimensions, Real, Device >::
print( std::ostream& str ) const
{
   using namespace TNL::Functions::Analytic;
   str << " timeDependence = " << this->timeDependence;
   str << " functionType = " << this->functionType;
   str << " function = " << this->function << "; ";
   switch( functionType )
   {
      case constant:
         return printFunction< Constant< Dimensions, Real> >( str );
      case expBump:
         return printFunction< ExpBump< Dimensions, Real> >( str );
      case sinBumps:
         return printFunction< SinBumps< Dimensions, Real> >( str );
      case sinWave:
         return printFunction< SinWave< Dimensions, Real> >( str );
      case cylinder:
         return printFunction< Cylinder< Dimensions, Real> >( str );
      case flowerpot:
         return printFunction< Flowerpot< Dimensions, Real> >( str );
      case twins:
         return printFunction< Twins< Dimensions, Real> >( str );
      case pseudoSquare:
         return printFunction< PseudoSquare< Dimensions, Real> >( str );
      case blob:
         return printFunction< Blob< Dimensions, Real> >( str );
   }
   return str;
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
TestFunction< FunctionDimensions, Real, Device >::
~TestFunction()
{
   deleteFunctions();
}


#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
extern template class TestFunction< 1, float, Devices::Host >;
extern template class TestFunction< 2, float, Devices::Host >;
extern template class TestFunction< 3, float, Devices::Host >;
#endif

extern template class TestFunction< 1, double, Devices::Host >;
extern template class TestFunction< 2, double, Devices::Host >;
extern template class TestFunction< 3, double, Devices::Host >;

#ifdef INSTANTIATE_LONG_DOUBLE
extern template class TestFunction< 1, long double, Devices::Host >;
extern template class TestFunction< 2, long double, Devices::Host >;
extern template class TestFunction< 3, long double, Devices::Host >;
#endif

#ifdef HAVE_CUDA
#ifdef INSTANTIATE_FLOAT
extern template class TestFunction< 1, float, Devices::Cuda>;
extern template class TestFunction< 2, float, Devices::Cuda >;
extern template class TestFunction< 3, float, Devices::Cuda >;
#endif

extern template class TestFunction< 1, double, Devices::Cuda >;
extern template class TestFunction< 2, double, Devices::Cuda >;
extern template class TestFunction< 3, double, Devices::Cuda >;

#ifdef INSTANTIATE_LONG_DOUBLE
extern template class TestFunction< 1, long double, Devices::Cuda >;
extern template class TestFunction< 2, long double, Devices::Cuda >;
extern template class TestFunction< 3, long double, Devices::Cuda >;
#endif
#endif

#endif

} // namespace Functions
} // namespace TNL

