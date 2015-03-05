/***************************************************************************
                          tnlFunctionEnumerator.h  -  description
                             -------------------
    begin                : Mar 5, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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
#ifndef SRC_FUNCTIONS_TNLFUNCTIONENUMERATOR_H_
#define SRC_FUNCTIONS_TNLFUNCTIONENUMERATOR_H_

#include <functions/tnlFunctionAdapter.h>

template< typename Function,
          typename DofVector >
class tnlFunctionEnumeratorTraversalUserData
{
   public:

      typedef DofVector::RealType RealType;

      const RealType *time;

      const Function* function;

      DofVector *u;

      const RealType* functionCoefficient;

      const RealType* dofVectorCoefficient;

      tnlFunctionEnumeratorTraversalUserData( const RealType& time,
                                              const Function& function,
                                              DofVector& u,
                                              const RealType& functionCoefficient,
                                              const RealType& dofVectorCoefficient )
      : time( &time ),
        function( &function ),
        u( &u ),
        functionCoefficient( &functionCoefficient ),
        dofVectorCoefficient( &dofVectorCoefficient )
      {};
};


template< typename Mesh,
          typename Function,
          typename DofVector >
class tnlFunctionEnumerator
{
   public:
      typedef Mesh MeshType;
      typedef typename DofVector::RealType RealType;
      typedef typename DofVector::DeviceType DeviceType;
      typedef typename DofVector::IndexType IndexType;
      typedef tnlFunctionEnumeratorTraversalUserData< Function,
                                                      DofVector > TraversalUserData;

      template< int EntityDimensions >
      void enumerate( const MeshType& mesh,
                      const Function& function
                      DofVector& u,
                      const RealType& functionCoefficient = 1.0,
                      const RealType& dofVectorCoefficient = 0.0,
                      const RealType& time = 0.0 ) const;


      class TraversalEntitiesProcessor
      {
         public:

            template< int EntityDimensions >
#ifdef HAVE_CUDA
            __host__ __device__
#endif
            static void processEntity( const MeshType& mesh,
                                       TraversalUserData& userData,
                                       const IndexType index )
            {
               typedef tnlFunctionAdapter< MeshType, Function > FunctionAdapter;
               ( *userData.u )[ index ] =
                        ( *userData.dofVectorCoefficient ) * ( *userData.u )[ index ] +
                        ( *userData.functionCoefficient ) * FunctionAdapter::getValue( mesh,
                                                                                       *userData.function,
                                                                                       index,
                                                                                       *userData.time );
            }

      };

};

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Function,
          typename DofVector >
class tnlFunctionEnumerator< tnlGrid< Dimensions, Real, Device, Index >,
                             Function,
                             DofVector >
{
   public:

      typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
      typedef typename MeshType::RealType RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef tnlFunctionEnumeratorTraversalUserData< Function,
                                                      DofVector > TraversalUserData;

      template< int EntityDimensions >
      void enumerate( const MeshType& mesh,
                      const Function& function,
                      DofVector& u,
                      const RealType& time = 0.0 ) const;

      class TraversalEntitiesProcessor
      {
         public:

         typedef typename MeshType::VertexType VertexType;

#ifdef HAVE_CUDA
            __host__ __device__
#endif
            static void processCell( const MeshType& mesh,
                                     TraversalUserData& userData,
                                     const IndexType index,
                                     const CoordinatesType& coordinates )
            {
               typedef tnlFunctionAdapter< MeshType, Function > FunctionAdapter;
               ( *userData.u )[ index ] =
                        ( *userData.dofVectorCoefficient ) * ( *userData.u )[ index ] +
                        ( *userData.functionCoefficient ) * FunctionAdapter::getValue( mesh,
                                                                                       *userData.function,
                                                                                       index,
                                                                                       coordinates,
                                                                                       *userData.time );

            }

#ifdef HAVE_CUDA
            __host__ __device__
#endif
            static void processFace( const MeshType& mesh,
                                     TraversalUserData& userData,
                                     const IndexType index,
                                     const CoordinatesType& coordinates )
            {
               typedef tnlFunctionAdapter< MeshType, Function > FunctionAdapter;
               ( *userData.u )[ index ] =
                        ( *userData.dofVectorCoefficient ) * ( *userData.u )[ index ] +
                        ( *userData.functionCoefficient ) * FunctionAdapter::getValue( mesh,
                                                                                       *userData.function,
                                                                                       index,
                                                                                       coordinates,
                                                                                       *userData.time );
            }
      };

};

#include <functions/tnlFunctionEnumerator_impl.h>



#endif /* SRC_FUNCTIONS_TNLFUNCTIONENUMERATOR_H_ */
