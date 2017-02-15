/***************************************************************************
                          CompressibleConservativeVariables.h  -  description
                             -------------------
    begin                : Feb 12, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/SharedPointer.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Functions/VectorField.h>
#include <TNL/Functions/MeshFunctionEvaluator.h>
#include "CompressibleConservativeVariables.h"

namespace TNL {
   
template< typename Mesh >
class PhysicalVariablesGetter
{
   public:
      
      typedef Mesh MeshType;
      typedef typename MeshType::RealType RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      static const int Dimensions = MeshType::getMeshDimensions();
      
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      typedef SharedPointer< MeshFunctionType > MeshFunctionPointer;
      typedef CompressibleConservativeVariables< MeshType > ConservativeVariablesType;
      typedef SharedPointer< ConservativeVariablesType > ConservativeVariablesPointer;
      typedef Functions::VectorField< Dimensions, MeshFunctionType > VelocityFieldType;
      typedef SharedPointer< VelocityFieldType > VelocityFieldPointer;
      
      
      
      class VelocityGetter : public Functions::Domain< Dimensions, Functions::MeshDomain >
      {
         public:
            typedef typename MeshType::RealType RealType;
            
            VelocityGetter( MeshFunctionPointer density, 
                            MeshFunctionPointer momentum )
            : density( density ), momentum( momentum ) {}
            
            template< typename EntityType >
            __cuda_callable__
            RealType operator()( const EntityType& meshEntity,
                                        const RealType& time = 0.0 ) const
            {
               return momentum.template getData< DeviceType >()( meshEntity ) / 
                      density.template getData< DeviceType >()( meshEntity );
            }
            
         protected:
            const MeshFunctionPointer density, momentum;
      };
      
      void getVelocity( const ConservativeVariablesPointer& conservativeVariables,
                        VelocityFieldPointer& velocity )
      {
         Functions::MeshFunctionEvaluator< MeshFunctionType, VelocityGetter > evaluator;
         for( int i = 0; i < Dimensions; i++ )
         {
            SharedPointer< VelocityGetter, DeviceType > velocityGetter( conservativeVariables->getDensity(),
                                                                        ( *conservativeVariables->getMomentum() )[ i ] );
            evaluator.evaluate( ( *velocity )[ i ], velocityGetter );
         }
      }
      
      
};
   
} //namespace TNL
