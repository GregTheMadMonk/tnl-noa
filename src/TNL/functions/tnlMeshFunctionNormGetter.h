/***************************************************************************
                          tnlMeshFunctionNormGetter.h  -  description
                             -------------------
    begin                : Jan 5, 2016
    copyright            : (C) 2016 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< typename MeshFunction,
          typename Mesh = typename MeshFunction::MeshType >
class tnlMeshFunctionNormGetter
{
};

/***
 * Specialization for grids
 * TODO: implement this even for other devices
 */
template< int Dimensions,
          typename MeshReal,
          typename MeshIndex,
          int EntityDimensions,
          typename Real >
class tnlMeshFunctionNormGetter< tnlMeshFunction< tnlGrid< Dimensions, MeshReal, Devices::Host, MeshIndex >, EntityDimensions, Real >,
                                 tnlGrid< Dimensions, MeshReal, Devices::Host, MeshIndex > >
{
   public:
 
      typedef tnlMeshFunction< tnlGrid< Dimensions, MeshReal, Devices::Host, MeshIndex >, EntityDimensions, Real > MeshFunctionType;
      typedef tnlGrid< Dimensions, MeshReal, Devices::Host, MeshIndex > GridType;
      typedef MeshReal MeshRealType;
      typedef Devices::Host DeviceType;
      typedef MeshIndex MeshIndexType;
      typedef typename MeshFunctionType::RealType RealType;
      typedef typename MeshFunctionType::MeshType MeshType;
      typedef typename MeshType::Face EntityType;
 
      static RealType getNorm( const MeshFunctionType& function,
                               const RealType& p )
      {
         if( EntityDimensions == Dimensions )
         {
            if( p == 1.0 )
               return function.getMesh().getCellMeasure() * function.getData().lpNorm( 1.0 );
            if( p == 2.0 )
               return std::sqrt( function.getMesh().getCellMeasure() ) * function.getData().lpNorm( 2.0 );
            return std::pow( function.getMesh().getCellMeasure(), 1.0 / p ) * function.getData().lpNorm( p );
         }
         if( EntityDimensions > 0 )
         {
            if( p == 1.0 )
            {
               RealType result( 0.0 );
               for( MeshIndexType i = 0;
                    i < function.getMesh().template getEntitiesCount< EntityType >();
                    i++ )
               {
                  EntityType entity = function.getMesh().template getEntity< EntityType >( i );
                  result += std::fabs( function[ i ] ) * entity.getMeasure();
               }
               return result;
            }
            if( p == 2.0 )
            {
               RealType result( 0.0 );
               for( MeshIndexType i = 0;
                    i < function.getMesh().template getEntitiesCount< EntityType >();
                    i++ )
               {
                  EntityType entity = function.getMesh().template getEntity< EntityType >( i );
                  result += function[ i ] * function[ i ] * entity.getMeasure();
               }
               return std::sqrt( result );
            }

            RealType result( 0.0 );
            for( MeshIndexType i = 0;
                 i < function.getMesh().template getEntitiesCount< EntityType >();
                 i++ )
            {
               EntityType entity = function.getMesh().template getEntity< EntityType >( i );
               result += std::pow( std::fabs( function[ i ] ), p ) * entity.getMeasure();
            }
            return std::pow( result, 1.0 / p );
         }
 
         if( p == 1.0 )
            return function.getData().lpNorm( 1.0 );
         if( p == 2.0 )
            return function.getData().lpNorm( 2.0 );
         return function.getData().lpNorm( p );
      }
};

/****
 * Specialization for CUDA devices
 */
template< int Dimensions,
          typename MeshReal,
          typename MeshIndex,
          int EntityDimensions,
          typename Real >
class tnlMeshFunctionNormGetter< tnlMeshFunction< tnlGrid< Dimensions, MeshReal, Devices::Cuda, MeshIndex >, EntityDimensions, Real >,
                                 tnlGrid< Dimensions, MeshReal, Devices::Cuda, MeshIndex > >
{
   public:
 
      typedef tnlMeshFunction< tnlGrid< Dimensions, MeshReal, Devices::Cuda, MeshIndex >, EntityDimensions, Real > MeshFunctionType;
      typedef tnlGrid< Dimensions, MeshReal, Devices::Cuda, MeshIndex > GridType;
      typedef MeshReal MeshRealType;
      typedef Devices::Cuda DeviceType;
      typedef MeshIndex MeshIndexType;
      typedef typename MeshFunctionType::RealType RealType;
      typedef typename MeshFunctionType::MeshType MeshType;
      typedef typename MeshType::Face EntityType;
 
      static RealType getNorm( const MeshFunctionType& function,
                               const RealType& p )
      {
         if( EntityDimensions == Dimensions )
         {
            if( p == 1.0 )
               return function.getMesh().getCellMeasure() * function.getData().lpNorm( 1.0 );
            if( p == 2.0 )
               return ::sqrt( function.getMesh().getCellMeasure() ) * function.getData().lpNorm( 2.0 );
            return ::pow( function.getMesh().getCellMeasure(), 1.0 / p ) * function.getData().lpNorm( p );
         }
         if( EntityDimensions > 0 )
         {
            Assert( false, std::cerr << "Not implemented yet." << std::endl );
         }
 
         if( p == 1.0 )
            return function.getData().lpNorm( 1.0 );
         if( p == 2.0 )
            return function.getData().lpNorm( 2.0 );
         return function.getData().lpNorm( p );
      }
};


} // namespace TNL

