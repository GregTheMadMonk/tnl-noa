/* 
 * File:   tnlDirectEikonalMethodsBase_impl.h
 * Author: oberhuber
 *
 * Created on July 14, 2016, 3:22 PM
 */

#pragma once

#include <limits>

template< typename Real,
          typename Device,
          typename Index >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > >::
initInterface( const MeshFunctionType& input,
               MeshFunctionType& output,
               InterfaceMapType& interfaceMap  )
{
   const MeshType& mesh = input.getMesh();
   typedef typename MeshType::Cell Cell;
   Cell cell( mesh );
   for( cell.getCoordinates().x() = 1;
        cell.getCoordinates().x() < mesh.getDimensions().x() - 1;
        cell.getCoordinates().x() ++ )
   {
      cell.refresh();
      const RealType& c = input( cell );      
      if( ! cell.isBoundaryEntity()  )
      {
         const auto& neighbors = cell.getNeighborEntities();
         //const IndexType& c = cell.getIndex();
         const IndexType e = neighbors.template getEntityIndex<  1 >();
         const IndexType w = neighbors.template getEntityIndex< -1 >();

         if( c * input[ e ] <= 0 || c * input[ w ] <= 0 )
         {
            output[ cell.getIndex() ] = c;
            interfaceMap[ cell.getIndex() ] = true;
            continue;
         }
      }
      output[ cell.getIndex() ] =
      c > 0 ? std::numeric_limits< RealType >::max() :
             -std::numeric_limits< RealType >::max();
      interfaceMap[ cell.getIndex() ] = false;
   }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshEntity >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
            const MeshEntity& cell )
{
}


template< typename Real,
          typename Device,
          typename Index >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
initInterface( const MeshFunctionType& input,
               MeshFunctionType& output,
               InterfaceMapType& interfaceMap  )
{
   const MeshType& mesh = input.getMesh();
   typedef typename MeshType::Cell Cell;
   Cell cell( mesh );
   for( cell.getCoordinates().y() = 0;
        cell.getCoordinates().y() < mesh.getDimensions().y();
        cell.getCoordinates().y() ++ )
      for( cell.getCoordinates().x() = 0;
           cell.getCoordinates().x() < mesh.getDimensions().x();
           cell.getCoordinates().x() ++ )
      {
         cell.refresh();
         const RealType& c = input( cell );
         if( ! cell.isBoundaryEntity()  )
         {
            auto neighbors = cell.getNeighborEntities();
            const IndexType e = neighbors.template getEntityIndex<  1,  0 >();
            const IndexType w = neighbors.template getEntityIndex< -1,  0 >();
            const IndexType n = neighbors.template getEntityIndex<  0,  1 >();
            const IndexType s = neighbors.template getEntityIndex<  0, -1 >();            
            if( c * input[ e ] <= 0 || c * input[ w ] <= 0 ||
                c * input[ n ] <= 0 || c * input[ s ] <= 0 )
            {
               output[ cell.getIndex() ] = c;
               interfaceMap[ cell.getIndex() ] = true;
               continue;
            }
         }
         output[ cell.getIndex() ] =
            c > 0 ? std::numeric_limits< RealType >::max() :
                   -std::numeric_limits< RealType >::max();  
         interfaceMap[ cell.getIndex() ] = false;
      }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshEntity >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
            const MeshEntity& cell )
{
   const auto& neighborEntities = cell.template getNeighborEntities< 2 >();
   const MeshType& mesh = cell.getMesh();
  
   const RealType& h = mesh.getSpaceSteps().x(); 
   const RealType value = u( cell );
   Real a, b, tmp;

   if( cell.getCoordinates().x() == 0 )
      a = u[ neighborEntities.template getEntityIndex< 1,  0 >() ];
   else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
      a = u[ neighborEntities.template getEntityIndex< -1,  0 >() ];
   else
   {
      a = argAbsMin( u[ neighborEntities.template getEntityIndex< -1,  0 >() ],
                     u[ neighborEntities.template getEntityIndex<  1,  0 >() ] );
   }

   if( cell.getCoordinates().y() == 0 )
      b = u[ neighborEntities.template getEntityIndex< 0,  1 >()];
   else if( cell.getCoordinates().y() == mesh.getDimensions().y() - 1 )
      b = u[ neighborEntities.template getEntityIndex< 0,  -1 >() ];
   else
   {
      b = argAbsMin( u[ neighborEntities.template getEntityIndex< 0,  -1 >() ],
                     u[ neighborEntities.template getEntityIndex< 0,   1 >() ] );
   }

   if( fabs( a ) == std::numeric_limits< Real >::max() && 
       fabs( b ) == std::numeric_limits< Real >::max() )
      return;
   if( fabs( a ) == std::numeric_limits< Real >::max() ||
       fabs( b ) == std::numeric_limits< Real >::max() ||
       fabs( a - b ) >= h )
   {
      tmp = argAbsMin( a, b ) + sign( value ) * h;
      /*   std::cerr << "a = " << a << " b = " << b << " h = " << h 
             << " ArgAbsMin( a, b ) = " << ArgAbsMin( a, b ) << " sign( value ) = " << sign( value )
             << " sign( value ) * h = " << sign( value ) * h
             << " ArgAbsMin( a, b ) + sign( value ) * h = " << ArgAbsMin( a, b ) + sign( value ) * h           
             << " tmp = " << tmp << std::endl;
      tmp = ArgAbsMin( a, b ) + sign( value ) * h;
      tmp = ArgAbsMin( a, b ) + sign( value ) * h;
      tmp = ArgAbsMin( a, b ) + sign( value ) * h;
      res = ArgAbsMin( a, b ) + sign( value ) * h;
      std::cerr << " tmp = " << tmp << std::endl;
      std::cerr << " res = " << res << std::endl;*/

   }
   else
      tmp = 0.5 * ( a + b + sign( value ) * sqrt( 2.0 * h * h - ( a - b ) * ( a - b ) ) );

   u[ cell.getIndex() ] = argAbsMin( value, tmp );
   //std::cerr << ArgAbsMin( value, tmp ) << " ";   
}


template< typename Real,
          typename Device,
          typename Index >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
initInterface( const MeshFunctionType& input,
               MeshFunctionType& output,
               InterfaceMapType& interfaceMap  )
{
   const MeshType& mesh = input.getMesh();
   typedef typename MeshType::Cell Cell;
   Cell cell( mesh );
   for( cell.getCoordinates().z() = 1;
        cell.getCoordinates().z() < mesh.getDimensions().z() - 1;
        cell.getCoordinates().z() ++ )   
      for( cell.getCoordinates().y() = 1;
           cell.getCoordinates().y() < mesh.getDimensions().y() - 1;
           cell.getCoordinates().y() ++ )
         for( cell.getCoordinates().x() = 1;
              cell.getCoordinates().x() < mesh.getDimensions().x() - 1;
              cell.getCoordinates().x() ++ )
         {
            cell.refresh();
            const RealType& c = input( cell );
            if( ! cell.isBoundaryEntity() )
            {
               auto neighbors = cell.getNeighborEntities();
               //const IndexType& c = cell.getIndex();
               const IndexType e = neighbors.template getEntityIndex<  1,  0,  0 >();
               const IndexType w = neighbors.template getEntityIndex< -1,  0,  0 >();
               const IndexType n = neighbors.template getEntityIndex<  0,  1,  0 >();
               const IndexType s = neighbors.template getEntityIndex<  0, -1,  0 >();
               const IndexType t = neighbors.template getEntityIndex<  0,  0,  1 >();
               const IndexType b = neighbors.template getEntityIndex<  0,  0, -1 >();

               if( c * input[ e ] <= 0 || c * input[ w ] <= 0 ||
                   c * input[ n ] <= 0 || c * input[ s ] <= 0 ||
                   c * input[ t ] <= 0 || c * input[ b ] <= 0 )
               {
                  output[ cell.getIndex() ] = c;
                  interfaceMap[ cell.getIndex() ] = true;
                  continue;
               }
            }
            output[ cell.getIndex() ] =
               c > 0 ? std::numeric_limits< RealType >::max() :
                      -std::numeric_limits< RealType >::max();
            interfaceMap[ cell.getIndex() ] = false;
         }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshEntity >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
            const MeshEntity& cell )
{
   
}
