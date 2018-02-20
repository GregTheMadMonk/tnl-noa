/* 
 * File:   tnlDirectEikonalMethodsBase_impl.h
 * Author: oberhuber
 *
 * Created on July 14, 2016, 3:22 PM
 */

#pragma once

#include <TNL/TypeInfo.h>

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
         const RealType& h = mesh.getSpaceSteps().x();
         //const IndexType& c = cell.getIndex();
         const IndexType e = neighbors.template getEntityIndex<  1 >();
         const IndexType w = neighbors.template getEntityIndex< -1 >();
         if( c * input[ e ] <=0 )
         {
             output[ cell.getIndex() ] =
             c >= 0 ? ( h * c )/( c - input[ e ] ) :
                      - ( h * c )/( c - input[ e ] );
             interfaceMap[ cell.getIndex() ] = true;
         }
         if( c * input[ w ] <=0 )
         {
             output[ cell.getIndex() ] =
             c >= 0 ? ( h * c )/( c - input[ w ] ) :
                      - ( h * c )/( c - input[ w ] );
             interfaceMap[ cell.getIndex() ] = true;
         }
      }
      output[ cell.getIndex() ] =
      c > 0 ? TNL::TypeInfo< RealType >::getMaxValue() :
             -TypeInfo< RealType >::getMaxValue();
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
            output[ cell.getIndex() ] =
                input( cell ) >= 0 ? TypeInfo< RealType >::getMaxValue() :
                                   - TypeInfo< RealType >::getMaxValue();
            interfaceMap[ cell.getIndex() ] = false;
        }
   
   const RealType& hx = mesh.getSpaceSteps().x();
   const RealType& hy = mesh.getSpaceSteps().y();     
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
            Real pom = 0;
            const IndexType e = neighbors.template getEntityIndex<  1,  0 >();
            const IndexType n = neighbors.template getEntityIndex<  0,  1 >();
            if( c * input[ n ] <= 0 )
            {
                if( c >= 0 )
                {
                    pom = ( hy * c )/( c - input[ n ]);
                    if( output[ cell.getIndex() ] > pom ) 
                        output[ cell.getIndex() ] = pom;
                    
                    output[ n ] = ( hy * c )/( c - input[ n ]) - hy;
                }else
                {
                    pom = - ( hy * c )/( c - input[ n ]);
                    if( output[ cell.getIndex() ] < pom )
                        output[ cell.getIndex() ] = pom;
                    
                    output[ n ] = hy - ( hy * c )/( c - input[ n ]);
                }
                interfaceMap[ cell.getIndex() ] = true;
                interfaceMap[ n ] = true;
            }
            if( c * input[ e ] <= 0 )
            {
                if( c >= 0 )
                {
                    pom = ( hx * c )/( c - input[ e ]);
                    if( output[ cell.getIndex() ] > pom )
                        output[ cell.getIndex() ] = pom;
                   
                    pom = pom - hx; //output[ e ] = (hx * c)/( c - input[ e ]) - hx;
                    if( output[ e ] >= pom )
                        output[ e ] = pom;                         
                }else
                {
                    pom = - (hx * c)/( c - input[ e ]);
                    if( output[ cell.getIndex() ] < pom )
                        output[ cell.getIndex() ] = pom;
                    
                    pom = pom + hx; //output[ e ] = hx - (hx * c)/( c - input[ e ]);
                    if( output[ e ] < pom )
                        output[ e ] = pom;
                }
                interfaceMap[ cell.getIndex() ] = true;
                interfaceMap[ e ] = true;
            }
         }
         /*output[ cell.getIndex() ] =
            c > 0 ? TypeInfo< RealType >::getMaxValue() :
                   -TypeInfo< RealType >::getMaxValue();*/  
         //interfaceMap[ cell.getIndex() ] = false; //is on line 90
      }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshEntity >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
            const MeshEntity& cell,
            double v)
{
   const auto& neighborEntities = cell.template getNeighborEntities< 2 >();
   const MeshType& mesh = cell.getMesh();
  
   const RealType& hx = mesh.getSpaceSteps().x();
   const RealType& hy = mesh.getSpaceSteps().y();
   const RealType value = u( cell );
   Real a, b, tmp;
   
   if( cell.getCoordinates().x() == 0 )
      a = u[ neighborEntities.template getEntityIndex< 1,  0 >() ];
   else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
      a = u[ neighborEntities.template getEntityIndex< -1,  0 >() ];
   else
   {
      a = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< -1,  0 >() ],
                     u[ neighborEntities.template getEntityIndex<  1,  0 >() ] );
   }

   if( cell.getCoordinates().y() == 0 )
      b = u[ neighborEntities.template getEntityIndex< 0,  1 >()];
   else if( cell.getCoordinates().y() == mesh.getDimensions().y() - 1 )
      b = u[ neighborEntities.template getEntityIndex< 0,  -1 >() ];
   else
   {
      b = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< 0,  -1 >() ],
                     u[ neighborEntities.template getEntityIndex< 0,   1 >() ] );
   }
   if( fabs( a ) == TypeInfo< Real >::getMaxValue() && 
       fabs( b ) == TypeInfo< Real >::getMaxValue() )
      return;
   /*if( fabs( a ) == TypeInfo< Real >::getMaxValue() ||
       fabs( b ) == TypeInfo< Real >::getMaxValue() ||
       fabs( a - b ) >= TNL::sqrt( (hx * hx + hy * hy)/v ) )
   {
      tmp = 
        fabs( a ) >= fabs( b ) ? b + TNL::sign( value ) * hy :
                                 a + TNL::sign( value ) * hx;
   }*/
   tmp = meet2DCondition( a, b, hx, hy, value, v);
   if( tmp == 0 )
      tmp = 
          fabs( a ) >= fabs( b ) ? fabs( b ) + TNL::sign( value ) * hy :
                    fabs( a ) + TNL::sign( value ) * hx;
            
   
   u[ cell.getIndex() ] = argAbsMin( value, tmp );
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
   for( cell.getCoordinates().z() = 0;
        cell.getCoordinates().z() < mesh.getDimensions().z();
        cell.getCoordinates().z() ++ )
        for( cell.getCoordinates().y() = 0;
             cell.getCoordinates().y() < mesh.getDimensions().y();
             cell.getCoordinates().y() ++ )
            for( cell.getCoordinates().x() = 0;
                 cell.getCoordinates().x() < mesh.getDimensions().x();
                 cell.getCoordinates().x() ++ )
            {
                output[ cell.getIndex() ] =
                input( cell ) >= 0 ? TypeInfo< RealType >::getMaxValue() :
                                   - TypeInfo< RealType >::getMaxValue();
                interfaceMap[ cell.getIndex() ] = false;
            }
    
   const RealType& hx = mesh.getSpaceSteps().x();
   const RealType& hy = mesh.getSpaceSteps().y();
   const RealType& hz = mesh.getSpaceSteps().z();
   for( cell.getCoordinates().z() = 0;
        cell.getCoordinates().z() < mesh.getDimensions().z();
        cell.getCoordinates().z() ++ )   
      for( cell.getCoordinates().y() = 0;
           cell.getCoordinates().y() < mesh.getDimensions().y();
           cell.getCoordinates().y() ++ )
         for( cell.getCoordinates().x() = 0;
              cell.getCoordinates().x() < mesh.getDimensions().x();
              cell.getCoordinates().x() ++ )
         {
            cell.refresh();
            const RealType& c = input( cell );
            if( ! cell.isBoundaryEntity() )
            {
               auto neighbors = cell.getNeighborEntities();
               Real pom = 0;
               //const IndexType& c = cell.getIndex();
               const IndexType e = neighbors.template getEntityIndex<  1,  0,  0 >();
               //const IndexType w = neighbors.template getEntityIndex< -1,  0,  0 >();
               const IndexType n = neighbors.template getEntityIndex<  0,  1,  0 >();
               //const IndexType s = neighbors.template getEntityIndex<  0, -1,  0 >();
               const IndexType t = neighbors.template getEntityIndex<  0,  0,  1 >();
               //const IndexType b = neighbors.template getEntityIndex<  0,  0, -1 >();
               /*if( c * input[ e ] <= 0 || c * input[ w ] <= 0 ||
                   c * input[ n ] <= 0 || c * input[ s ] <= 0 ||
                   c * input[ t ] <= 0 || c * input[ b ] <= 0 )
               {
                  output[ cell.getIndex() ] = c;
                  interfaceMap[ cell.getIndex() ] = true;
                  continue;
               }*/
               if( c * input[ n ] <= 0 )
               {
                   if( c >= 0 )
                   {
                   pom = ( hy * c )/( c - input[ n ]);
                   if( output[ cell.getIndex() ] > pom ) 
                   output[ cell.getIndex() ] = pom;
                    
                   output[ n ] = ( hy * c )/( c - input[ n ]) - hy;
                   }else
                   {
                     pom = - ( hy * c )/( c - input[ n ]);
                     if( output[ cell.getIndex() ] < pom )
                         output[ cell.getIndex() ] = pom;
                    
                     output[ n ] = hy - ( hy * c )/( c - input[ n ]);
                   }
               interfaceMap[ cell.getIndex() ] = true;
               interfaceMap[ n ] = true;
               }
               if( c * input[ e ] <= 0 )
               {
                   if( c >= 0 )
                   {
                       pom = ( hx * c )/( c - input[ e ]);
                       if( output[ cell.getIndex() ] > pom )
                       output[ cell.getIndex() ] = pom;
                   
                       pom = pom - hx; //output[ e ] = (hx * c)/( c - input[ e ]) - hx;
                       if( output[ e ] >= pom )
                           output[ e ] = pom;                         
                   }else
                   {
                       pom = - (hx * c)/( c - input[ e ]);
                       if( output[ cell.getIndex() ] < pom )
                           output[ cell.getIndex() ] = pom;
                    
                       pom = pom + hx; //output[ e ] = hx - (hx * c)/( c - input[ e ]);
                       if( output[ e ] < pom )
                           output[ e ] = pom;
                   }
               interfaceMap[ cell.getIndex() ] = true;
               interfaceMap[ e ] = true;
               }
               if( c * input[ t ] <= 0 )
               {
                   if( c >= 0 )
                   {
                       pom = ( hz * c )/( c - input[ t ]);
                       if( output[ cell.getIndex() ] > pom )
                           output[ cell.getIndex() ] = pom;
                   
                       pom = pom - hz; //output[ e ] = (hx * c)/( c - input[ e ]) - hx;
                       if( output[ t ] >= pom )
                           output[ t ] = pom;                         
                   }else
                   {
                       pom = - (hz * c)/( c - input[ t ]);
                       if( output[ cell.getIndex() ] < pom )
                           output[ cell.getIndex() ] = pom;
                    
                       pom = pom + hz; //output[ e ] = hx - (hx * c)/( c - input[ e ]);
                       if( output[ t ] < pom )
                           output[ t ] = pom;
                   }
               interfaceMap[ cell.getIndex() ] = true;
               interfaceMap[ t ] = true;
               }           
            }
            /*output[ cell.getIndex() ] =
               c > 0 ? TypeInfo< RealType >::getMaxValue() :
                      -TypeInfo< RealType >::getMaxValue();
            interfaceMap[ cell.getIndex() ] = false;*/ //is on line 245
         }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshEntity >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
            const MeshEntity& cell, 
            double v )
{
   const auto& neighborEntities = cell.template getNeighborEntities< 2 >();
   const MeshType& mesh = cell.getMesh();
  
   const RealType& hx = mesh.getSpaceSteps().x();
   const RealType& hy = mesh.getSpaceSteps().y();
   const RealType& hz = mesh.getSpaceSteps().z();
   const RealType value = u( cell );
   Real a, b, c, tmp = 0;
   
   if( cell.getCoordinates().x() == 0 )
      a = u[ neighborEntities.template getEntityIndex< 1,  0 >() ];
   else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
      a = u[ neighborEntities.template getEntityIndex< -1,  0 >() ];
   else
   {
      a = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< -1,  0 >() ],
                     u[ neighborEntities.template getEntityIndex<  1,  0 >() ] );
   }
   
   if( cell.getCoordinates().x() == 0 )
      a = u[ neighborEntities.template getEntityIndex< 1, 0, 0 >() ];
   else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
      a = u[ neighborEntities.template getEntityIndex< -1, 0, 0 >() ];
   else
   {
      a = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< -1, 0, 0 >() ],
                        u[ neighborEntities.template getEntityIndex< 1, 0, 0 >() ] );
   }
   if( cell.getCoordinates().x() == 0 )
      b = u[ neighborEntities.template getEntityIndex< 0, 1, 0 >() ];
   else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
      b = u[ neighborEntities.template getEntityIndex< 0, -1, 0 >() ];
   else
   {
      b = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< 0, -1, 0 >() ],
                        u[ neighborEntities.template getEntityIndex< 0, 1, 0 >() ] );
   }if( cell.getCoordinates().x() == 0 )
      c = u[ neighborEntities.template getEntityIndex< 0, 0, 1 >() ];
   else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
      c = u[ neighborEntities.template getEntityIndex< 0, 0, -1 >() ];
   else
   {
      c = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< 0, 0, -1 >() ],
                         u[ neighborEntities.template getEntityIndex< 0, 0, 1 >() ] );
   }
   if( fabs( a ) == TypeInfo< Real >::getMaxValue() && 
       fabs( b ) == TypeInfo< Real >::getMaxValue() &&
       fabs( c ) == TypeInfo< Real >::getMaxValue() )
      return;
   if( fabs( a ) == TypeInfo< Real >::getMaxValue() ||
       fabs( b ) == TypeInfo< Real >::getMaxValue() ||
       fabs( c ) == TypeInfo< Real >::getMaxValue() ||
       hz * hz * fabs( a - b ) * fabs( a - b ) + hy * hy * fabs( a - c ) * fabs( a - c ) +
           hx * hx * fabs( b - c ) * fabs( b - c ) >=  ( hx * hx * hy * hy + hx * hx * hz * hz + hy * hy * hz * hz )/v )
   {
       /*if( fabs( a ) != TypeInfo< Real >::getMaxValue() &&
           fabs( b ) != TypeInfo< Real >::getMaxValue() &&
           fabs( a - b ) >= TNL::sqrt( (hx * hx + hy * hy)/v ) )
       {
           tmp = ( hx * hx * a + hy * hy * b + 
                sign( value ) * hx * hy * sqrt( ( hx * hx + hy * hy )/v - 
                ( a - b ) * ( a - b ) ) )/( hx * hx + hy * hy );
       }
       if( fabs( a ) != TypeInfo< Real >::getMaxValue() &&
           fabs( c ) != TypeInfo< Real >::getMaxValue() &&
           fabs( a - c ) >= TNL::sqrt( (hx * hx + hz * hz)/v ) )
       {
           tmp = ( hx * hx * a + hz * hz * c + 
                sign( value ) * hx * hz * sqrt( ( hx * hx + hz * hz )/v - 
                ( a - c ) * ( a - c ) ) )/( hx * hx + hz * hz );
       }
       if( fabs( b ) != TypeInfo< Real >::getMaxValue() &&
           fabs( c ) != TypeInfo< Real >::getMaxValue() &&
           fabs( b - c ) >= TNL::sqrt( (hy * hy + hz * hz)/v ) )
       {
           tmp = ( hy * hy * b + hz * hz * c + 
                sign( value ) * hy * hz * sqrt( ( hy * hy + hz * hz )/v - 
                ( b - c ) * ( b - c ) ) )/( hy * hy + hz * hz );
       }*/
       Real pom = sortMinims( a, b, c, (Real)hx, (Real)hy, (Real)hz);
       tmp = meet2DCondition( pom[0], pom[1], pom[3], pom[4], value, v);
       
       if( tmp == 0 )
           tmp = pom[ 0 ] + TNL::sign( value ) * pom[ 3 ]; 
        
   }
   else
      tmp = ( hx * hx * a + hy * hy * b + hz * hz * c +
            sign( value ) * hx * hy * hz * sqrt( ( hx * hx + hy * hy + hz * hz)/v - 
            hz * hz * ( a - b ) * ( a - b ) + hy * hy * ( a - c ) * ( a - c ) +
            hx * hx * ( b - c ) * ( b - c ) ) )/( hx * hx + hy * hy + hz * hz );
            
   
   u[ cell.getIndex() ] = argAbsMin( value, tmp );
}

template < typename T1, typename T2 >
T1 meet2DCondition( T1 a, T1 b, const T2 ha, const T2 hb, const T1 value, double v)
{
   T1 tmp;
   if( fabs( a ) != TypeInfo< T1 >::getMaxValue() &&
       fabs( b ) != TypeInfo< T1 >::getMaxValue() &&
       fabs( a - b ) <= TNL::sqrt( (ha * ha + hb * hb)/v ) )
   {
      tmp = ( ha * ha * a + hb * hb * b + 
            sign( value ) * hb * hb * sqrt( ( ha * ha + hb * hb )/v - 
            ( a - b ) * ( a - b ) ) )/( ha * ha + hb * hb );
   }
   else
       tmp = 0;
   
   return tmp;
}

template < typename T1 >
T1 sortMinims( T1 a, T1 b, T1 c, T1 ha, T1 hb, T1 hc)
{
    T1 tmp[6];
    if( a <= b && b <= c){
        tmp = {a,b,c,ha,hb,hc};
    }
    else if( a <= c && c <= b ){
        tmp = {a,c,b,ha,hc,hb};
    }
    else if( b<=a && a<=c ){
        tmp = {b,a,c,hb,ha,hc};
    }
    else if( b <= c && c <= a ){
        tmp = {b,c,a,hb,hc,ha};
    }
    else if( c <= a && a <= b ){
        tmp = {c,a,b,hc,ha,hb};
    }
    else if( c <=b && b <= a ){
        tmp = {c,b,a,hc,hb,ha};
    }
        
    return tmp;
}