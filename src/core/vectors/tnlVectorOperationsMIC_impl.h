/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tnlVectorOperationsMIC_impl.h
 * Author: hanouvit
 *
 * Created on 2. května 2016, 12:57
 */

#ifndef TNLVECTOROPERATIONSMIC_IMPL_H
#define TNLVECTOROPERATIONSMIC_IMPL_H

//static const int OpenMPVectorOperationsThreshold = 65536; // TODO: check this threshold

template< typename Vector >
void tnlVectorOperations< tnlMIC >::addElement( Vector& v,
                                                 const typename Vector::IndexType i,
                                                 const typename Vector::RealType& value )
{
  // v[ i ] += value;
    //cout << "Errorous function, not clear wher should be called (device or Host)" << endl;
    v.setElement(i,v.getElemet(i)+value);
}

template< typename Vector >
void tnlVectorOperations< tnlMIC >::addElement( Vector& v,
                                                 const typename Vector::IndexType i,
                                                 const typename Vector::RealType& value,
                                                 const typename Vector::RealType& thisElementMultiplicator )
{
   //v[ i ] = thisElementMultiplicator * v[ i ] + value;
    //    cout << "Errorous function, not clear wher should be called (device or Host)" << endl;
    v.setElement(i,thisElementMultiplicator*v.getElemet(i)+value);
}

template< typename Vector >
typename Vector::RealType tnlVectorOperations< tnlMIC >::getVectorMax( const Vector& v )
{
 //tady je možnost paralelizace  
  typename Vector :: RealType result;
  #pragma offload target(mic) out(result)
   {
       result=v[0];
       for(typename Vector :: Index i=1;i<v.getSize();i++)
       {
           if(result<v[i])
               result=v[i];
       }
   }    
   return result;
}

template< typename Vector >
typename Vector :: RealType tnlVectorOperations< tnlMIC > :: getVectorMin( const Vector& v )
{
 //tady je možnost paralelizace  
  typename Vector :: RealType result;
  #pragma offload target(mic) out(result)
   {
       result=v[0];
       for(typename Vector :: Index i=1;i<v.getSize();i++)
       {
           if(result>v[i])
               result=v[i];
       }
   }    
   return result;
}

template< typename Vector >
typename Vector :: RealType tnlVectorOperations< tnlMIC > :: getVectorAbsMax( const Vector& v )
{
 //tady je možnost paralelizace  
  typename Vector :: RealType result;
 /* #pragma offload target(mic) out(result)
   {
       result=fabs(v[0]);
       for(typename Vector :: Index i=1;i<v.getSize();i++)
       {
           if(result<fabs(v[i]))
               result=fabs(v[i]);
       }
   }*/    
   return result;
}


template< typename Vector >
typename Vector :: RealType tnlVectorOperations< tnlMIC > :: getVectorAbsMin( const Vector& v )
{
   /*typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   tnlAssert( v. getSize() > 0, );
   Real result = fabs( v. getElement( 0 ) );
   const Index n = v. getSize();
   for( Index i = 1; i < n; i ++ )
      result = Min( result, ( Real ) fabs( v. getElement( i ) ) );
   return result;*/
    return 0;
}

template< typename Vector >
typename Vector::RealType
tnlVectorOperations< tnlMIC >::getVectorL1Norm( const Vector& v )
{
/*   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   tnlAssert( v. getSize() > 0, );

   Real result( 0.0 );
   const Index n = v. getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( tnlHost::isOMPEnabled() && n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif           
   for( Index i = 0; i < n; i ++ )
      result += fabs( v[ i ] );
   return result;*/
    return 0;
}

template< typename Vector >
typename Vector::RealType
tnlVectorOperations< tnlMIC >::getVectorL2Norm( const Vector& v )
{
  /* typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   tnlAssert( v. getSize() > 0, );
   Real result( 0.0 );
   const Index n = v. getSize();
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:result) if( tnlHost::isOMPEnabled() &&n > OpenMPVectorOperationsThreshold ) // TODO: check this threshold
#endif           
   for( Index i = 0; i < n; i ++ )
   {
      const Real& aux = v[ i ];
      result += aux * aux;
   }
   return sqrt( result );*/
    return 0;
}

template< typename Vector >
typename Vector::RealType
tnlVectorOperations< tnlMIC >:: getVectorLpNorm( const Vector& v,
                 const typename Vector :: RealType& p )
{
   /*typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   tnlAssert( v. getSize() > 0, );
   tnlAssert( p > 0.0,
              cerr << " p = " << p );
   if( p == 1.0 )
      return getVectorL1Norm( v );
   if( p == 2.0 )
      return getVectorL2Norm( v );

   Real result( 0.0 );
   const Index n = v. getSize();
          
   for( Index i = 0; i < n; i ++ )
      result += pow( fabs( v[ i ] ), p );
   return pow( result, 1.0 / p );*/
    return 0;
}

template< typename Vector >
typename Vector :: RealType tnlVectorOperations< tnlMIC > :: getVectorSum( const Vector& v )
{
    /*
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;
   tnlAssert( v. getSize() > 0, );

   Real result( 0.0 );
   const Index n = v. getSize();
       
   for( Index i = 0; i < n; i ++ )
      result += v[ i ];
   return result;*/
    return 0;
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlMIC> :: getVectorDifferenceMax( const Vector1& v1,
                                                                                       const Vector2& v2 )
{
   /*typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;
   tnlAssert( v1. getSize() > 0, );
   tnlAssert( v1. getSize() == v2. getSize(), );
   Real result = v1. getElement( 0 ) - v2. getElement( 0 );
   const Index n = v1. getSize();
   for( Index i = 1; i < n; i ++ )
      result =  Max( result, v1. getElement( i ) - v2. getElement( i ) );
   return result;*/
    return 0;
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlMIC > :: getVectorDifferenceMin( const Vector1& v1,
                                                                                       const Vector2& v2 )
{
    /*
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( v1. getSize() > 0, );
   tnlAssert( v1. getSize() == v2. getSize(), );

   Real result = v1. getElement( 0 ) - v2. getElement( 0 );
   const Index n = v1. getSize();
   for( Index i = 1; i < n; i ++ )
      result =  Min( result, v1. getElement( i ) - v2. getElement( i ) );
   return result;
     */
    return 0;
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlMIC > :: getVectorDifferenceAbsMax( const Vector1& v1,
                                                                                          const Vector2& v2 )
{
    /*
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( v1. getSize() > 0, );
   tnlAssert( v1. getSize() == v2. getSize(), );

   Real result = fabs( v1. getElement( 0 ) - v2. getElement( 0 ) );
   const Index n = v1. getSize();
   for( Index i = 1; i < n; i ++ )
      result =  Max( result, ( Real ) fabs( v1. getElement( i ) - v2. getElement( i ) ) );
   return result;
     */
    return 0;
}

template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlMIC > :: getVectorDifferenceAbsMin( const Vector1& v1,
                                                                                          const Vector2& v2 )
{
    /*
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( v1. getSize() > 0, );
   tnlAssert( v1. getSize() == v2. getSize(), );

   Real result = fabs( v1[ 0 ] - v2[ 0 ] );
   const Index n = v1. getSize();
   for( Index i = 1; i < n; i ++ )
      result =  Min( result, ( Real ) fabs( v1[ i ] - v2[ i ] ) );
   return result;
     */
    return 0;
}

template< typename Vector1, typename Vector2 >
typename Vector1::RealType
tnlVectorOperations< tnlMIC >::
getVectorDifferenceL1Norm( const Vector1& v1,
                           const Vector2& v2 )
{
    /*
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( v1. getSize() > 0, );
   tnlAssert( v1. getSize() == v2. getSize(), );

   Real result( 0.0 );
   const Index n = v1. getSize();
      
   for( Index i = 0; i < n; i ++ )
      result += fabs( v1[ i ] - v2[ i ] );
   return result;
     */
    return 0;
}

template< typename Vector1, typename Vector2 >
typename Vector1::RealType
tnlVectorOperations< tnlMIC >::
getVectorDifferenceL2Norm( const Vector1& v1,
                           const Vector2& v2 )
{
    /*
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( v1. getSize() > 0, );
   tnlAssert( v1. getSize() == v2. getSize(), );

   Real result( 0.0 );
   const Index n = v1. getSize();
      
   for( Index i = 0; i < n; i ++ )
   {
      Real aux = fabs( v1[ i ] - v2[ i ] );
      result += aux * aux;
   }
   return sqrt( result );
     */
    return 0;
}


template< typename Vector1, typename Vector2 >
typename Vector1::RealType
tnlVectorOperations< tnlMIC >::
getVectorDifferenceLpNorm( const Vector1& v1,
                           const Vector2& v2,
                           const typename Vector1::RealType& p )
{
    /*
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( p > 0.0,
              cerr << " p = " << p );
   tnlAssert( v1. getSize() > 0, );
   tnlAssert( v1. getSize() == v2. getSize(), );

   if( p == 1.0 )
      return getVectorDifferenceL1Norm( v1, v2 );
   if( p == 2.0 )
      return getVectorDifferenceL2Norm( v1, v2 );

   Real result( 0.0 );
   const Index n = v1. getSize();
       
   for( Index i = 0; i < n; i ++ )
      result += pow( fabs( v1. getElement( i ) - v2. getElement( i ) ), p );
   return pow( result, 1.0 / p );
     */
    return 0;
}

template< typename Vector1, typename Vector2 >
typename Vector1::RealType tnlVectorOperations< tnlMIC > :: getVectorDifferenceSum( const Vector1& v1,
                                                                                     const Vector2& v2 )
{
    /*
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( v1. getSize() > 0, );
   tnlAssert( v1. getSize() == v2. getSize(), );

   Real result( 0.0 );
   const Index n = v1. getSize();
        
   for( Index i = 0; i < n; i ++ )
      result += v1. getElement( i ) - v2. getElement( i );
   return result;
     */
    return 0;
}


template< typename Vector >
void tnlVectorOperations< tnlMIC > :: vectorScalarMultiplication( Vector& v,
                                                                   const typename Vector :: RealType& alpha )
{
    /*
   typedef typename Vector :: RealType Real;
   typedef typename Vector :: IndexType Index;

   tnlAssert( v. getSize() > 0, );

   const Index n = v. getSize();
       
   for( Index i = 0; i < n; i ++ )
      v[ i ] *= alpha;
     */

}


template< typename Vector1, typename Vector2 >
typename Vector1 :: RealType tnlVectorOperations< tnlMIC > :: getScalarProduct( const Vector1& v1,
                                                                                 const Vector2& v2 )
{
    /*
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( v1. getSize() > 0, );
   tnlAssert( v1. getSize() == v2. getSize(), );

   Real result( 0.0 );
   const Index n = v1. getSize();
   
   for( Index i = 0; i < n; i++ )
      result += v1[ i ] * v2[ i ];
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
   /*return result;*/
   return 0;        
}

template< typename Vector1, typename Vector2 >
void tnlVectorOperations< tnlMIC > :: addVector( Vector1& y,
                                                  const Vector2& x,
                                                  const typename Vector2::RealType& alpha,
                                                  const typename Vector1::RealType& thisMultiplicator )
{
   /*typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( x. getSize() > 0, );
   tnlAssert( x. getSize() == y. getSize(), );

   const Index n = y. getSize();
   if( thisMultiplicator == 1.0 )
          
      for( Index i = 0; i < n; i ++ )
         y[ i ] += alpha * x[ i ];
   else
         
      for( Index i = 0; i < n; i ++ )
         y[ i ] = thisMultiplicator * y[ i ] + alpha * x[ i ];
    */
    return 0;
}

template< typename Vector1,
          typename Vector2,
          typename Vector3 >
void
tnlVectorOperations< tnlMIC >::
addVectors( Vector1& v,
            const Vector2& v1,
            const typename Vector2::RealType& multiplicator1,
            const Vector3& v2,
            const typename Vector3::RealType& multiplicator2,
            const typename Vector1::RealType& thisMultiplicator )
{
    /*
   typedef typename Vector1 :: RealType Real;
   typedef typename Vector1 :: IndexType Index;

   tnlAssert( v.getSize() > 0, );
   tnlAssert( v.getSize() == v1.getSize(), );
   tnlAssert( v.getSize() == v2.getSize(), );
   
   const Index n = v.getSize();
   if( thisMultiplicator == 1.0 )
          
      for( Index i = 0; i < n; i ++ )
         v[ i ] += multiplicator1 * v1[ i ] + multiplicator2 * v2[ i ];
   else
      
      for( Index i = 0; i < n; i ++ )
         v[ i ] = thisMultiplicator * v[ i ] * multiplicator1 * v1[ i ] + multiplicator2 * v2[ i ];
     */
}

template< typename Vector >
void tnlVectorOperations< tnlMIC >::computePrefixSum( Vector& v,
                                                       typename Vector::IndexType begin,
                                                       typename Vector::IndexType end )
{
    /*
   typedef typename Vector::IndexType Index;
   for( Index i = begin + 1; i < end; i++ )
      v[ i ] += v[ i - 1 ];
     */
}

template< typename Vector >
void tnlVectorOperations< tnlMIC >::computeExclusivePrefixSum( Vector& v,
                                                                typename Vector::IndexType begin,
                                                                typename Vector::IndexType end )
{
    /*
   typedef typename Vector::IndexType Index;
   typedef typename Vector::RealType Real;
   Real aux( v[ begin ] );
   v[ begin ] = 0.0;
   for( Index i = begin + 1; i < end; i++ )
   {
      Real x = v[ i ];
      v[ i ] = aux;
      aux += x;
   }
     */
}

#endif /* TNLVECTOROPERATIONSMIC_IMPL_H */

