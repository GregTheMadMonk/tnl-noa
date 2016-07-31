/***************************************************************************
                          tnlFunctionEvaluator.h  -  description
                             -------------------
    begin                : Mar 5, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/tnlFunctionAdapter.h>

namespace TNL {
namespace Functions {   

template< typename MeshFunction,
          typename Function >
class tnlFunctionEvaluatorTraverserUserData
{
   public:

      typedef typename MeshFunction::RealType RealType;

      const RealType *time;

      const Function* function;

      MeshFunction *u;

      const RealType* functionCoefficient;

      const RealType* dofVectorCoefficient;

      tnlFunctionEvaluatorTraverserUserData( const RealType& time,
                                              const Function& function,
                                              MeshFunction& u,
                                              const RealType& functionCoefficient,
                                              const RealType& dofVectorCoefficient )
      : time( &time ),
        function( &function ),
        u( &u ),
        functionCoefficient( &functionCoefficient ),
        dofVectorCoefficient( &dofVectorCoefficient )
      {};
 
};


template< typename MeshFunction,
          typename Function >
class tnlFunctionEvaluator
{
   public:
      typedef typename MeshFunction::MeshType MeshType;
      typedef typename MeshFunction::RealType RealType;
      typedef typename MeshFunction::DeviceType DeviceType;
      typedef typename MeshFunction::IndexType IndexType;
      typedef tnlFunctionEvaluatorTraverserUserData< MeshFunction,
                                                     Function > TraverserUserData;

      void assignment( const Function& function,
                       MeshFunction& u,
                       const RealType& functionCoefficient = 1.0,
                       const RealType& meshFunctionCoefficient = 0.0,
                       const RealType& time = 0.0 ) const;

      //void addition( ... );
 
      //void subtruction( .... );
 
      //void multiplication( .... );

      class TraverserEntitiesProcessor
      {
         public:

            template< typename EntityType >
            __cuda_callable__
            static void processEntity( const MeshType& mesh,
                                       TraverserUserData& userData,
                                       const EntityType& entity )
            {
               typedef tnlFunctionAdapter< MeshType, Function > FunctionAdapter;
               if( ! *userData.dofVectorCoefficient  )
                  ( *userData.u )( entity ) =
                     ( *userData.functionCoefficient ) * FunctionAdapter::getValue( *userData.function,
                                                                                    entity,
                                                                                    *userData.time );
               else
                 ( *userData.u )( entity ) =
                             ( *userData.dofVectorCoefficient ) * ( *userData.u )( entity ) +
                             ( *userData.functionCoefficient ) * FunctionAdapter::getValue( *userData.function,
                                                                                            entity,
                                                                                            *userData.time );
            }

      };

};

} // namespace Functions
} // namespace TNL

#include <TNL/Functions/tnlFunctionEvaluator_impl.h>

