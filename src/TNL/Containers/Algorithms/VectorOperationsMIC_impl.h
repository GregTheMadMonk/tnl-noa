/***************************************************************************
                          VectorOperationsMIC_impl.h  -  description
                                by hanouvit
                          -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

#include <TNL/Devices/MIC.h>


namespace TNL {
namespace Containers {   
namespace Algorithms {

//static const int OpenMPVectorOperationsThreshold = 65536; // TODO: check this threshold

template< typename Vector >
void VectorOperations< Devices::MIC >::addElement( Vector& v,
                                                 const typename Vector::IndexType i,
                                                 const typename Vector::RealType& value )
{
  // v[ i ] += value;
    //cout << "Errorous function, not clear wher should be called (device or Host)" << endl;
    v.setElement(i,v.getElemet(i)+value);
}

template< typename Vector >
void VectorOperations< Devices::MIC >::addElement( Vector& v,
                                                 const typename Vector::IndexType i,
                                                 const typename Vector::RealType& value,
                                                 const typename Vector::RealType& thisElementMultiplicator )
{
   //v[ i ] = thisElementMultiplicator * v[ i ] + value;
    //    cout << "Errorous function, not clear wher should be called (device or Host)" << endl;
    v.setElement(i,thisElementMultiplicator*v.getElemet(i)+value);
}

template< typename Vector >
typename Vector::RealType VectorOperations< Devices::MIC >::getVectorMax( const Vector& v )
{
 //tady je možnost paralelizace  
  typename Vector :: RealType result;
  typename Vector ::IndexType size=v.getSize();
  Devices::MICHider<const typename Vector :: RealType > vct;
  vct.pointer=v.getData();
  
  #pragma offload target(mic) in(vct,size) out(result)
   {
       result=vct.pointer[0];
       for(typename Vector ::IndexType i=1;i<size;i++)
       {
           if(result<vct.pointer[i])
               result=vct.pointer[i];
       }
   }    
   return result;
}

template< typename Vector >
typename Vector :: RealType VectorOperations< Devices::MIC > :: getVectorMin( const Vector& v )
{
 //tady je možnost paralelizace  
  typename Vector :: RealType result;
  typename Vector ::IndexType size=v.getSize();
  Devices::MICHider<const typename Vector :: RealType > vct;
  vct.pointer=v.getData();
  
  #pragma offload target(mic) in(vct,size) out(result)
   {
       result=vct.pointer[0];
       for(typename Vector ::IndexType i=1;i<size;i++)
       {
           if(result>vct.pointer[i])
               result=vct.pointer[i];
       }
   }    
   return result;
}

template< typename Vector >
typename Vector :: RealType VectorOperations< Devices::MIC > :: getVectorAbsMax( const Vector& v )
{
 //tady je možnost paralelizace  
  typename Vector :: RealType result;
  typename Vector ::IndexType size=v.getSize();
  Devices::MICHider<const typename Vector :: RealType > vct;
  vct.pointer=v.getData();
  
  #pragma offload target(mic) in(vct,size) out(result)
   {
       result=std::fabs(vct.pointer[0]);
       for(typename Vector ::IndexType i=1;i<size;i++)
       {
           if(result<std::fabs(vct.pointer[i]))
               result=std::fabs(vct.pointer[i]);
       }
   }    
   return result;
}


template< typename Vector >
typename Vector :: RealType VectorOperations< Devices::MIC > :: getVectorAbsMin( const Vector& v )
{
 //tady je možnost paralelizace  
  typename Vector :: RealType result;
  typename Vector ::IndexType size=v.getSize();
  Devices::MICHider<const typename Vector :: RealType > vct;
  vct.pointer=v.getData();
  
  #pragma offload target(mic) in(vct,size) out(result)
   {
       result=std::fabs(vct.pointer[0]);
       for(typename Vector ::IndexType i=1;i<size;i++)
       {
           if(result>std::fabs(vct.pointer[i]))
               result=std::fabs(vct.pointer[i]);
       }
   }    
   return result;
}

template< typename Vector >
typename Vector::RealType
VectorOperations< Devices::MIC >::getVectorL1Norm( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   TNL_ASSERT( v. getSize() > 0, );
   
   Real result( 0.0 );
   const Index n = v. getSize();
   Devices::MICHider<const Real > vct;
   vct.pointer=v.getData();

#pragma offload target(mic) in(vct,n) inout(result)
{
    #pragma omp parallel for reduction(+:result)// if( n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
        for( Index i = 0; i < n; i ++ )
            result += std::fabs( vct.pointer[ i ] );
}
   return result;
}

template< typename Vector >
typename Vector::RealType
VectorOperations< Devices::MIC >::getVectorL2Norm( const Vector& v )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   TNL_ASSERT( v. getSize() > 0, );
   Real result( 0.0 );
   const Index n = v. getSize();
   Devices::MICHider<const Real > vct;
   vct.pointer=v.getData();

#pragma offload target(mic) in(vct,n) inout(result)
{
    #pragma omp parallel for reduction(+:result) //if( n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold       
   for( Index i = 0; i < n; i ++ )
   {
      const Real& aux = vct.pointer[ i ];
      result += aux * aux;
   }
}
   return std::sqrt( result );
}

template< typename Vector >
typename Vector::RealType
VectorOperations< Devices::MIC >:: getVectorLpNorm( const Vector& v,
                 const typename Vector :: RealType& p )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   TNL_ASSERT( v. getSize() > 0, );
   TNL_ASSERT( p > 0.0,
              std::cerr << " p = " << p );
   if( p == 1.0 )
      return getVectorL1Norm( v );
   if( p == 2.0 )
      return getVectorL2Norm( v );
   
   Real result( 0.0 );
   const Index n = v. getSize();
   Devices::MICHider<const Real > vct;
   vct.pointer=v.getData();

#pragma offload target(mic) in(vct,n) inout(result)
{
    #pragma omp parallel for reduction(+:result) //if( n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold       
   for( Index i = 0; i < n; i ++ )
   {
      result += pow( std::fabs( vct.pointer[ i ] ), p );
   }
}
   return pow( result, 1.0 / p );
}

template< typename Vector >
typename Vector :: RealType VectorOperations< Devices::MIC > :: getVectorSum( const Vector& v )
{
 
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   TNL_ASSERT( v. getSize() > 0, );

   Real result( 0.0 );
   const Index n = v. getSize();
   Devices::MICHider<const Real > vct;
   vct.pointer=v.getData();

#pragma offload target(mic) in(vct,n) inout(result)
{
    #pragma omp parallel for reduction(+:result)// if( n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
        for( Index i = 0; i < n; i ++ )
            result += vct.pointer[ i ] ;
}
   return result;
}


template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType VectorOperations< Devices::MIC> :: getVectorDifferenceMax( const Vector1& v1,
                                                                                       const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;
   TNL_ASSERT( v1. getSize() > 0, );
   TNL_ASSERT( v1. getSize() == v2. getSize(), );
 
   Real result( 0.0 );
   const Index n = v1. getSize();
   Devices::MICHider<const Real > vct1;
   Devices::MICHider<const Real > vct2;
   vct1.pointer=v1.getData();
   vct2.pointer=v2.getData();
  
#pragma offload target(mic) in(n,vct1,vct2) out(result) 
{
   result = vct1.pointer[0] - vct2.pointer[0];
   for( Index i = 1; i < n; i ++ )
      result = max( result, vct1.pointer[ i ] - vct2.pointer[ i ] );
}  
   return result;
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType VectorOperations< Devices::MIC > :: getVectorDifferenceMin( const Vector1& v1,
                                                                                       const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;
   TNL_ASSERT( v1. getSize() > 0, );
   TNL_ASSERT( v1. getSize() == v2. getSize(), );
 
   Real result( 0.0 );
   const Index n = v1. getSize();
   Devices::MICHider<const Real > vct1;
   Devices::MICHider<const Real > vct2;
   vct1.pointer=v1.getData();
   vct2.pointer=v2.getData();
  
#pragma offload target(mic) in(n,vct1,vct2) out(result) 
{
   result = vct1.pointer[0] - vct2.pointer[0];
   for( Index i = 1; i < n; i ++ )
      result = min( result, vct1.pointer[ i ] - vct2.pointer[ i ] );
}  
   return result;
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType VectorOperations< Devices::MIC > :: getVectorDifferenceAbsMax( const Vector1& v1,
                                                                                          const Vector2& v2 )
{
    typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;
   TNL_ASSERT( v1. getSize() > 0, );
   TNL_ASSERT( v1. getSize() == v2. getSize(), );
 
   Real result( 0.0 );
   const Index n = v1. getSize();
   Devices::MICHider<const Real > vct1;
   Devices::MICHider<const Real > vct2;
   vct1.pointer=v1.getData();
   vct2.pointer=v2.getData();
  
#pragma offload target(mic) in(n,vct1,vct2) out(result) 
{
   result = std::fabs(vct1.pointer[0] - vct2.pointer[0]);
   for( Index i = 1; i < n; i ++ )
      result = max( result, std::fabs(vct1.pointer[ i ] - vct2.pointer[ i ]) );
}  
   return result;   
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType VectorOperations< Devices::MIC > :: getVectorDifferenceAbsMin( const Vector1& v1,
                                                                                          const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;
   TNL_ASSERT( v1. getSize() > 0, );
   TNL_ASSERT( v1. getSize() == v2. getSize(), );
 
   Real result( 0.0 );
   const Index n = v1. getSize();
   Devices::MICHider<const Real > vct1;
   Devices::MICHider<const Real > vct2;
   vct1.pointer=v1.getData();
   vct2.pointer=v2.getData();
  
#pragma offload target(mic) in(n,vct1,vct2) out(result) 
{
   result = std::fabs(vct1.pointer[0] - vct2.pointer[0]);
   for( Index i = 1; i < n; i ++ )
      result = min( result, std::fabs(vct1.pointer[ i ] - vct2.pointer[ i ]) );
}  
   return result;   
}

template< typename Vector1, typename Vector2 >
typename Vector1::RealType
VectorOperations< Devices::MIC >::
getVectorDifferenceL1Norm( const Vector1& v1,
                           const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   TNL_ASSERT( v1. getSize() > 0, );
   TNL_ASSERT( v1. getSize() == v2. getSize(), );

   Real result( 0.0 );
   const Index n = v1. getSize();
   Devices::MICHider<const Real> vct1;
   Devices::MICHider<const Real > vct2;
   vct1.pointer=v1.getData();
   vct2.pointer=v2.getData();
  
#pragma offload target(mic) in(n,vct1,vct2) inout(result) 
{
   for( Index i = 0; i < n; i ++ )
      result += std::fabs( vct1.pointer[ i ] - vct2.pointer[ i ] );
}
   return result;
   
}

template< typename Vector1, typename Vector2 >
typename Vector1::RealType
VectorOperations< Devices::MIC >::
getVectorDifferenceL2Norm( const Vector1& v1,
                           const Vector2& v2 )
{
    
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   TNL_ASSERT( v1. getSize() > 0, );
   TNL_ASSERT( v1. getSize() == v2. getSize(), );

   Real result( 0.0 );
   const Index n = v1. getSize();
   
   Devices::MICHider<const Real > vct1;
   Devices::MICHider<const Real > vct2;
   vct1.pointer=v1.getData();
   vct2.pointer=v2.getData();
  
#pragma offload target(mic) in(n,vct1,vct2) inout(result) 
{
   for( Index i = 0; i < n; i ++ )
   {
      Real aux = std::fabs( vct1.pointer[ i ] - vct2.pointer[ i ] );
      result += aux * aux;
   }
}
   
 return std::sqrt( result );
   
}


template< typename Vector1, typename Vector2 >
typename Vector1::RealType
VectorOperations< Devices::MIC >::
getVectorDifferenceLpNorm( const Vector1& v1,
                           const Vector2& v2,
                           const typename Vector1::RealType& p )
{
    
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   TNL_ASSERT( p > 0.0,
              std::cerr << " p = " << p );
   TNL_ASSERT( v1. getSize() > 0, );
   TNL_ASSERT( v1. getSize() == v2. getSize(), );

   if( p == 1.0 )
      return getVectorDifferenceL1Norm( v1, v2 );
   if( p == 2.0 )
      return getVectorDifferenceL2Norm( v1, v2 );

   Real result( 0.0 );
   const Index n = v1. getSize();
   
   Devices::MICHider<const Real > vct1;
   Devices::MICHider<const Real > vct2;
   vct1.pointer=v1.getData();
   vct2.pointer=v2.getData();
  
#pragma offload target(mic) in(n,vct1,vct2) inout(result) 
{
   for( Index i = 0; i < n; i ++ )
   {
      result += pow( std::fabs( vct1.pointer[ i ] - vct2.pointer[ i ] ), p );
   }
}
   return pow( result, 1.0 / p );
 
}

template< typename Vector1, typename Vector2 >
typename Vector1::RealType VectorOperations< Devices::MIC > :: getVectorDifferenceSum( const Vector1& v1,
                                                                                     const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   TNL_ASSERT( v1. getSize() > 0, );
   TNL_ASSERT( v1. getSize() == v2. getSize(), );

   Real result( 0.0 );
   const Index n = v1. getSize();
   Devices::MICHider<const Real > vct1;
   Devices::MICHider<const Real > vct2;
   vct1.pointer=v1.getData();
   vct2.pointer=v2.getData();
  
#pragma offload target(mic) in(n,vct1,vct2) inout(result) 
{
   for( Index i = 0; i < n; i ++ )
      result +=  vct1.pointer[ i ] - vct2.pointer[ i ];
}
   return result;
}


template< typename Vector >
void VectorOperations< Devices::MIC > :: vectorScalarMultiplication( Vector& v,
                                                                   const typename Vector :: RealType& alpha )
{
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   TNL_ASSERT( v. getSize() > 0, );

   const Index n = v. getSize();
   Devices::MICHider<Real > vct;
   vct.pointer=v.getData();
   Real a=alpha;
   
#pragma offload target(mic) in(vct,a,n)
{
   for( Index i = 0; i < n; i ++ )
      vct.pointer[ i ] *= a;
}    

}


template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType VectorOperations< Devices::MIC > :: getScalarProduct( const Vector1& v1,
                                                                                 const Vector2& v2 )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;
   TNL_ASSERT( v1. getSize() > 0, );
   TNL_ASSERT( v1. getSize() == v2. getSize(), );

   Real result( 0.0 );
   const Index n = v1. getSize();
   Devices::MICHider<const Real > vct1;
   Devices::MICHider<const Real > vct2;
   vct1.pointer=v1.getData();
   vct2.pointer=v2.getData();
   
#pragma offload target(mic) in(vct1,vct2,n) inout(result)
{

#pragma omp parallel for reduction(+:result)// if( n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
   for( Index i = 0; i < n; i++ )
      result += vct1.pointer[ i ] * vct2.pointer[ i ];
}
   /*Real result1( 0.0 ), result2( 0.0 ), result3( 0.0 ), result4( 0.0 ),
        result5( 0.0 ), result6( 0.0 ), result7( 0.0 ), result8( 0.0 );
   Index i( 0 );
   while( i + 8 < n )
   {
      result1 += v1[ i ] * v2[ i ];
      result2 += v1[ i + 1 ] * v2[ i + 1 ];
      result3 += v1[ i + 2 ] * v2[ i + 2 ];
      result4 += v1[ i + 3 ] * v2[ i + 3 ];
      result5 += v1[ i + 4 ] * v2[ i + 4 ];
      result6 += v1[ i + 5 ] * v2[ i + 5 ];
      result7 += v1[ i + 6 ] * v2[ i + 6 ];
      result8 += v1[ i + 7 ] * v2[ i + 7 ];
      i += 8;
   }
   Real result = result1 + result2 + result3 + result4 + result5 +result6 +result7 +result8;
   while( i < n )
      result += v1[ i ] * v2[ i++ ];*/
   return result;      
}

template< typename Vector1, typename Vector2 >
void VectorOperations< Devices::MIC > :: addVector( Vector1& y,
                                                  const Vector2& x,
                                                  const typename Vector2::RealType& alpha,
                                                  const typename Vector1::RealType& thisMultiplicator )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;
   TNL_ASSERT( x. getSize() > 0, );
   TNL_ASSERT( x. getSize() == y. getSize(), );
   
   const Index n = y. getSize();
   Devices::MICHider<Real> vct;
   Devices::MICHider<const Real> vct2;
   vct.pointer=y.getData();
   vct2.pointer=x.getData();
   Real a=alpha;
   Real t=thisMultiplicator;
   
#pragma offload target(mic) in(vct,vct2,n,a,t)
{
      for( Index i = 0; i < n; i ++ )
         vct.pointer[ i ] = t * vct.pointer[ i ] + a * vct2.pointer[ i ];
}
    
}

template< typename Vector1,
          typename Vector2,
          typename Vector3 >
void
VectorOperations< Devices::MIC >::
addVectors( Vector1& v,
            const Vector2& v1,
            const typename Vector2::RealType& multiplicator1,
            const Vector3& v2,
            const typename Vector3::RealType& multiplicator2,
            const typename Vector1::RealType& thisMultiplicator )
{
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;
   TNL_ASSERT( v.getSize() > 0, );
   TNL_ASSERT( v.getSize() == v1.getSize(), );
   TNL_ASSERT( v.getSize() == v2.getSize(), );
   
    const Index n = v. getSize();
    Devices::MICHider<Real> vct;
    Devices::MICHider<const Real> vct1;
    Devices::MICHider<const Real> vct2;
    vct.pointer=v.getData();
    vct1.pointer=v1.getData();
    vct2.pointer=v2.getData();
    Real t=thisMultiplicator;
    Real m1=multiplicator1;
    Real m2=multiplicator2;

#pragma offload target(mic) in(vct,vct1,vct2,n,t,m1,m2)
{
      for( Index i = 0; i < n; i ++ )
         vct.pointer[ i ] = t * vct.pointer[ i ] + m1 * vct1.pointer[ i ] + m2 * vct2.pointer[ i ];
}
    
}

template< typename Vector >
void VectorOperations< Devices::MIC >::computePrefixSum( Vector& v,
                                                       typename Vector::IndexType begin,
                                                       typename Vector::IndexType end )
{
    
   typedef typename Vector::IndexType Index;
   
   //std::cout << v.getSize()<< "    " << end <<endl;
   
   TNL_ASSERT( v.getSize() > 0, );
   TNL_ASSERT( v.getSize() >= end, );
   TNL_ASSERT( v.getSize() > begin, );
   TNL_ASSERT( end > begin, );
   
   Devices::MICHider<typename Vector :: RealType> vct;
   vct.pointer=v.getData();
#pragma offload target(mic) in(vct,begin,end)
{
   for( Index i = begin + 1; i < end; i++ )
      vct.pointer[ i ] += vct.pointer[ i - 1 ];
}
    
}

template< typename Vector >
void VectorOperations< Devices::MIC >::computeExclusivePrefixSum( Vector& v,
                                                                typename Vector::IndexType begin,
                                                                typename Vector::IndexType end )
{
   
   typedef typename Vector::IndexType Index;
   typedef typename Vector::RealType Real;
   TNL_ASSERT( v.getSize() > 0, );
   TNL_ASSERT( v.getSize() >= end, );
   TNL_ASSERT( v.getSize() > begin, );
   TNL_ASSERT( begin >= 0, );
   TNL_ASSERT( end > begin, );
   
   Devices::MICHider<Real> vct;
   vct.pointer=v.getData();
   
#pragma offload target(mic) in(vct,begin,end)
{ 
   
   Real aux( vct.pointer[ begin ] );
   vct.pointer[ begin ] = 0.0;
   for( Index i = begin + 1; i < end; i++ )
   {
      Real x = vct.pointer[ i ];
      vct.pointer[ i ] = aux;
      aux += x;
   }
 }
}


} // namespace Algorithms
} // namespace Containers
} // namespace TNL
