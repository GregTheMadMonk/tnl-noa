/***************************************************************************
                          StaticFor.h  -  description
                             -------------------
    begin                : Feb 23, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< typename IndexType, IndexType val >
class StaticForIndexTag
{
public:
   static const IndexType value = val;

   typedef StaticForIndexTag<IndexType, val - 1> Decrement;
};


template< typename IndexType,
          typename Begin,
          typename N,
          template< IndexType > class LoopBody >
class StaticForExecutor
{
   public:

   __cuda_callable__
   static void exec()
   {
      StaticForExecutor< IndexType, Begin, typename N::Decrement, LoopBody >::exec();
      LoopBody< Begin::value + N::value - 1 >::exec();
   }

   template< typename T >
   __cuda_callable__
   static void exec( T& p )
   {
      StaticForExecutor< IndexType, Begin, typename N::Decrement, LoopBody >::exec( p );
      LoopBody< Begin::value + N::value - 1 >::exec( p );
   }

   template< typename T0,
             typename T1 >
   __cuda_callable__
   static void exec( T0& p0, T1& p1 )
   {
      StaticForExecutor< IndexType, Begin, typename N::Decrement, LoopBody >::exec( p0, p1 );
      LoopBody< Begin::value + N::value - 1 >::exec( p0, p1 );
   }

   template< typename T0,
             typename T1,
             typename T2 >
   __cuda_callable__
   static void exec( T0& p0, T1& p1, T2& p2 )
   {
      StaticForExecutor< IndexType, Begin, typename N::Decrement, LoopBody >::exec( p0, p1, p2 );
      LoopBody< Begin::value + N::value - 1 >::exec( p0, p1, p2 );
   }

   template< typename T0,
             typename T1,
             typename T2,
             typename T3 >
   __cuda_callable__
   static void exec( T0& p0, T1& p1, T2& p2, T3& p3 )
   {
      StaticForExecutor< IndexType, Begin, typename N::Decrement, LoopBody >::exec( p0, p1, p2, p3 );
      LoopBody< Begin::value + N::value - 1 >::exec( p0, p1, p2, p3 );
   }
};

template< typename IndexType,
          typename Begin,
          template< IndexType > class LoopBody >
class StaticForExecutor< IndexType,
                         Begin,
                         StaticForIndexTag< IndexType, 0 >,
                         LoopBody >
{
   public:

   __cuda_callable__
   static void exec() {}

   template< typename T >
   __cuda_callable__
   static void exec( T& p ) {}

   template< typename T0,
             typename T1 >
   __cuda_callable__
   static void exec( T0& p0, T1& p1 ) {}

   template< typename T0,
             typename T1,
             typename T2 >
   __cuda_callable__
   static void exec( T0& p0, T1& p1, T2& p2 ) {}

   template< typename T0,
             typename T1,
             typename T2,
             typename T3 >
   __cuda_callable__
   static void exec( T0& p0, T1& p1, T2& p2, T3& p3 ) {}
};

template< typename IndexType,
          IndexType begin,
          IndexType end,
          template< IndexType > class LoopBody >
class StaticFor
{
   public:

   __cuda_callable__
   static void exec()
   {
#ifndef HAVE_ICPC
      StaticForExecutor< IndexType,
                         StaticForIndexTag< IndexType, begin >,
                         StaticForIndexTag< IndexType, end - begin >,
                         LoopBody >::exec();
#else
     Assert( false, );
#endif
   }

   template< typename T >
   __cuda_callable__
   static void exec( T &p )
   {
#ifndef HAVE_ICPC
      StaticForExecutor< IndexType,
                         StaticForIndexTag< IndexType, begin >,
                         StaticForIndexTag< IndexType, end - begin >,
                         LoopBody >::exec( p );
#else
     Assert( false, );
#endif
   }

   template< typename T0,
             typename T1 >
   __cuda_callable__
   static void exec( T0& p0, T1& p1 )
   {
#ifndef HAVE_ICPC
      StaticForExecutor< IndexType,
                         StaticForIndexTag< IndexType, begin >,
                         StaticForIndexTag< IndexType, end - begin >,
                         LoopBody >::exec( p0, p1 );
#else
     Assert( false, );
#endif
   }

   template< typename T0,
             typename T1,
             typename T2 >
   __cuda_callable__
   static void exec( T0& p0, T1& p1, T2& p2 )
   {
#ifndef HAVE_ICPC
      StaticForExecutor< IndexType,
                         StaticForIndexTag< IndexType, begin >,
                         StaticForIndexTag< IndexType, end - begin >,
                         LoopBody >::exec( p0, p1, p2 );
#else
     Assert( false, );
#endif
   }

   template< typename T0,
             typename T1,
             typename T2,
             typename T3 >
   __cuda_callable__
   static void exec( T0& p0, T1& p1, T2& p2, T3& p3 )
   {
#ifndef HAVE_ICPC
      StaticForExecutor< IndexType,
                         StaticForIndexTag< IndexType, begin >,
                         StaticForIndexTag< IndexType, end - begin >,
                         LoopBody >::exec( p0, p1, p2, p3 );
#else
     Assert( false, );
#endif
   }
};


} // namespace TNL
