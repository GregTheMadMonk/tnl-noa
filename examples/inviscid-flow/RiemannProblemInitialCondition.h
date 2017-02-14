/***************************************************************************
                          RiemannProblemInitialCondition.h  -  description
                             -------------------
    begin                : Feb 13, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/StaticVector.h>
#include <TNL/Operators/Analytic/Sign.h>
#include <TNL/Functions/MeshFunctionEvaluator.h>
#include <TNL/Operators/Analytic/Sign.h>
#include "CompressibleConservativeVariables.h"

namespace TNL {

template< typename Mesh >
class RiemannProblemInitialCondition
{
   public:
      
      typedef Mesh MeshType;
      typedef typename MeshType::RealType RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      static const int Dimensions = MeshType::getMeshDimensions();
      typedef Containers::StaticVector< Dimensions, RealType > VertexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      typedef SharedPointer< MeshFunctionType > MeshFunctionPointer;
      typedef Functions::VectorField< Dimensions, MeshType > VectorFieldType;
      
      RiemannProblemInitialCondition()
         : discontinuityPlacement( 0.5 ),
           leftDensity( 1.0 ), rightDensity( 0.0 ),
           leftVelocity( 1.0 ), rightVelocity( 0.0 ),
           leftPressure( 1.0e5 ), rightPressure( 0.0 ),
           gamma( 1.67 ){}

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "discontinuity-placement-0", "x-coordinate of the discontinuity placement.", 0.5 );
         config.addEntry< double >( prefix + "discontinuity-placement-1", "y-coordinate of the discontinuity placement.", 0.5 );
         config.addEntry< double >( prefix + "discontinuity-placement-2", "z-coordinate of the discontinuity placement.", 0.5 );
         config.addEntry< double >( prefix + "left-density", "Density on the left side of the discontinuity.", 1.0 );
         config.addEntry< double >( prefix + "right-density", "Density on the right side of the discontinuity.", 0.0 );
         config.addEntry< double >( prefix + "left-velocity-0", "x-coordinate of the velocity on the left side of the discontinuity.", 1.0 );
         config.addEntry< double >( prefix + "left-velocity-1", "y-coordinate of the velocity on the left side of the discontinuity.", 1.0 );
         config.addEntry< double >( prefix + "left-velocity-2", "z-coordinate of the velocity on the left side of the discontinuity.", 1.0 );
         config.addEntry< double >( prefix + "right-velocity-0", "x-coordinate of the velocity on the right side of the discontinuity.", 0.0 );
         config.addEntry< double >( prefix + "right-velocity-1", "y-coordinate of the velocity on the right side of the discontinuity.", 0.0 );
         config.addEntry< double >( prefix + "right-velocity-2", "z-coordinate of the velocity on the right side of the discontinuity.", 0.0 );
         config.addEntry< double >( prefix + "left-pressure", "Pressure on the left side of the discontinuity.", 1.0 );
         config.addEntry< double >( prefix + "right-pressure", "Pressure on the right side of the discontinuity.", 0.0 );
         config.addEntry< double >( prefix + "gamma", "Gamma in the ideal gas state equation.", 1.67 );
      }      
      
      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         this->discontinuityPlacement.setup( parameters, prefix + "discontinuity-placement-" );
         this->leftVelocity.setup( parameters, prefix + "left-velocity-" );
         this->rightVelocity.setup( parameters, prefix + "right-velocity-" );
         this->leftDensity = parameters.getParameter< double >( prefix + "left-density" );
         this->rightDensity = parameters.getParameter< double >( prefix + "right-density" );
         this->leftPressure = parameters.getParameter< double >( prefix + "left-pressure" );
         this->rightPressure = parameters.getParameter< double >( prefix + "right-pressure" );
         this->gamma = parameters.getParameter< double >( prefix + "gamma" );
         return true;
      };
      
      void setDiscontinuityPlacement( const VertexType& v )
      {
         this->discontinuityPlacement = v;
      }
      
      const VertexType& getDiscontinuityPlasement() const
      {
         return this->discontinuityPlacement;
      }
      
      void setLeftDensity( const RealType& leftDensity )
      {
         this->leftDensity = leftDensity;
      }
      
      const RealType& getLeftDensity() const
      {
         return this->leftDensity;
      }
      
      void setRightDensity( const RealType& rightDensity )
      {
         this->rightDensity = rightDensity;
      }
      
      const RealType& getRightDensity() const
      {
         return this->rightDensity;
      }

      void setLeftVelocity( const VertexType& leftVelocity )
      {
         this->leftVelocity = leftVelocity;
      }
      
      const VertexType& getLeftVelocity() const
      {
         return this->leftVelocity;
      }
      
      void setRightVelocity( const RealType& rightVelocity )
      {
         this->rightVelocity = rightVelocity;
      }
      
      const VertexType& getRightVelocity() const
      {
         return this->rightVelocity;
      }

      void setLeftPressure( const RealType& leftPressure )
      {
         this->leftPressure = leftPressure;
      }
      
      const RealType& getLeftPressure() const
      {
         return this->leftPressure;
      }
      
      void setRightPressure( const RealType& rightPressure )
      {
         this->rightPressure = rightPressure;
      }
      
      const RealType& getRightPressure() const
      {
         return this->rightPressure;
      }
      
      void setInitialCondition( CompressibleConservativeVariables< MeshType >& conservativeVariables,
                                const VertexType& center = VertexType( 0.0 ) )
      {
         typedef Functions::Analytic::VectorNorm< Dimensions, RealType > VectorNormType;
         typedef Operators::Analytic::Sign< Dimensions, RealType > SignType;
         typedef Functions::OperatorFunction< SignType, VectorNormType > InitialConditionType;
         typedef SharedPointer< InitialConditionType, DeviceType > InitialConditionPointer;
         
         InitialConditionPointer initialCondition;
         initialCondition->getFunction().setCenter( center );
         initialCondition->getFunction().setMaxNorm( true );
         initialCondition->getFunction().setRadius( discontinuityPlacement[ 0 ] );
         discontinuityPlacement *= 1.0 / discontinuityPlacement[ 0 ];
         for( int i = 1; i < Dimensions; i++ )
            discontinuityPlacement[ i ] = 1.0 / discontinuityPlacement[ i ];
         initialCondition->getFunction().setAnisotropy( discontinuityPlacement );
         initialCondition->getFunction().setMultiplicator( -1.0 );
         
         Functions::MeshFunctionEvaluator< MeshFunctionType, InitialConditionType > evaluator;

         /****
          * Density
          */
         initialCondition->getOperator().setPositiveValue( leftDensity );
         initialCondition->getOperator().setNegativeValue( rightDensity );
         evaluator.evaluate( conservativeVariables.getDensity(), initialCondition );
         
         /****
          * Momentum
          */
         for( int i = 0; i < Dimensions; i++ )
         {
            initialCondition->getOperator().setPositiveValue( leftDensity * leftVelocity[ i ] );
            initialCondition->getOperator().setNegativeValue( rightDensity * rightVelocity[ i ] );
            evaluator.evaluate( ( *conservativeVariables.getMomentum() )[ i ], initialCondition );
         }
         
         /****
          * Energy
          */
         const RealType leftKineticEnergy = leftVelocity.lpNorm( 2.0 );
         const RealType rightKineticEnergy = rightVelocity.lpNorm( 2.0 );
         const RealType leftEnergy = leftPressure / ( gamma + 1.0 ) + 0.2 * leftDensity * leftKineticEnergy * leftKineticEnergy;
         const RealType rightEnergy = rightPressure / ( gamma + 1.0 ) + 0.2 * rightDensity * rightKineticEnergy * rightKineticEnergy;
         initialCondition->getOperator().setPositiveValue( leftEnergy );
         initialCondition->getOperator().setNegativeValue( rightEnergy );
         evaluator.evaluate( conservativeVariables.getEnergy(), initialCondition );
      }
      
      
   protected:
      
      VertexType discontinuityPlacement;
      
      RealType leftDensity, rightDensity;
      VertexType leftVelocity, rightVelocity;
      RealType leftPressure, rightPressure;
      
      RealType gamma; // gamma in the ideal gas state equation
};

} //namespace TNL