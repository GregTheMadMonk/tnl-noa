/***************************************************************************
                          tnlCommunicatorTester.h  -  description
                             -------------------
    begin                : Feb 6, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
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

#ifndef TNLCOMMUNICATORTESTER_H_
#define TNLCOMMUNICATORTESTER_H_
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <core/tnlCommunicator.h>
#include <core/tnlFile.h>

template< tnlDevice Device > class tnlCommunicatorTester : public CppUnit :: TestCase
{
   public:
   tnlCommunicatorTester(){};

   virtual
   ~tnlCommunicatorTester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite* suiteOfTests = new CppUnit :: TestSuite( "tnlCommunicatorTester" );
      CppUnit :: TestResult result;
      /*suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCommunicatorTester< Device > >(
                               "testCommunicatorInitiation",
                               & tnlCommunicatorTester< Device > :: testCommunicatorInitiation )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCommunicatorTester< Device > >(
                               "testCommunicatorSendReceive",
                               & tnlCommunicatorTester< Device > :: testCommunicatorSendReceive )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCommunicatorTester< Device > >(
                               "testCommunicatorSendReceiveOnLongVector",
                               & tnlCommunicatorTester< Device > :: testCommunicatorSendReceiveOnLongVector )
                              );

      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCommunicatorTester< Device > >(
                               "testCommunicatorBroadcast",
                               & tnlCommunicatorTester< Device > :: testCommunicatorBroadcast )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCommunicatorTester< Device > >(
                               "testCommunicatorReduction",
                               & tnlCommunicatorTester< Device > :: testCommunicatorReduction )
                              );*/
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCommunicatorTester< Device > >(
                               "testCommunicatorScatter",
                               & tnlCommunicatorTester< Device > :: testCommunicatorScatter )
                              );
      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCommunicatorTester< Device > >(
                               "testCommunicatorGather",
                               & tnlCommunicatorTester< Device > :: testCommunicatorGather )
                              );

      suiteOfTests -> addTest( new CppUnit :: TestCaller< tnlCommunicatorTester< Device > >(
                               "testCommunicatorBarrier",
                               & tnlCommunicatorTester< Device > :: testCommunicatorBarrier )
                              );



      return suiteOfTests;
   };

   void testCommunicatorInitiation()
   {
      /*tnlCommunicator< tnlHost > com;
      com. setCommunicationGroupSize( 8 );
      com. start();*/
   };

   void testCommunicatorSendReceive()
   {
      tnlCommunicator< 1, tnlHost > com;
      com. setCommunicationGroupSize( 2 );
      com. start();
      int dataSend, dataReceive;
      if( com. getDeviceId() == 0 )
      {
         dataSend = 721;
         com. send( &dataSend, 1 );
         com. receive( &dataReceive, 1 );
         CPPUNIT_ASSERT( dataSend == dataReceive );
      }
      if( com. getDeviceId() == 1 )
      {
         com. receive( &dataReceive, 0 );
         cerr << "Received data = " << dataReceive << endl;
         CPPUNIT_ASSERT( dataReceive == 721 );
         com. send( &dataReceive, 0 );
      }
   };

   void testCommunicatorSendReceiveOnLongVector()
   {
      tnlCommunicator< 1, tnlHost > com;
      com. setCommunicationGroupSize( 2 );
      com. start();
      tnlLongVector< double > sendingLongVector( "sendingLongVector" ), receivingLongVector( "receivingLongVector" );
      sendingLongVector. setSize( 100 );
      receivingLongVector. setSize( 100 );
      if( com. getDeviceId() == 0 )
      {
         sendingLongVector. setValue( 721.0 );
         com. send( sendingLongVector, 1 );
         com. receive( receivingLongVector, 1 );
         CPPUNIT_ASSERT( sendingLongVector == receivingLongVector );
      }
      if( com. getDeviceId() == 1 )
      {
         com. receive( receivingLongVector, 0 );
         com. send( receivingLongVector, 0 );
      }
   };

   void testCommunicatorBroadcast()
   {
      tnlCommunicator< 1, tnlHost > com;
      com. setCommunicationGroupSize( 16 );
      com. start();
      double d( 3.14 );
      if( com. getDeviceId() == 0 )
         d = 2.73;
      com. broadcast( &d, 0 );

      CPPUNIT_ASSERT( d == 2.73 );
   };

   void testCommunicatorBroadcastLongVector()
   {
      tnlCommunicator< 1, tnlHost > com;
      com. setCommunicationGroupSize( 4 );
      com. start();
      tnlLongVector< double, tnlHost > v( "broadcast-vector", 100 );
      v. setValue( 3.14 );
      if( com. getDeviceId() == 0 )
         v. setValue( 2.73 );
      com. broadcast( v, 0 );

      CPPUNIT_ASSERT( v[ 0 ] == 2.73 );
   };

   void testCommunicatorReduction()
   {
      tnlCommunicator< 1, tnlHost > com;
      const int groupSize = 16;
      com. setCommunicationGroupSize( groupSize );
      com. start();
      double d( 1.0 );
      com. reduction( &d, tnlSumReduction, 0 );

      if( com. getDeviceId() == 0 )
         CPPUNIT_ASSERT( d == groupSize );
   };

   void testCommunicatorScatter()
   {
      tnlCommunicator< 1, tnlHost > com;
      const int groupSize = 4;
      com. setCommunicationGroupSize( groupSize );
      com. start();
      tnlLongVector< double, tnlHost > originalData( "originalData" );
      if( com. getDeviceId() == 0 )
      {
         originalData. setSize( groupSize );
         for( int i = 0; i < groupSize; i ++ )
            originalData[ i ] = i;
      }
      tnlLongVector< double, tnlHost > scatteredData( "scatteredData" );
      scatteredData. setSize( 1 );
      com. scatter( originalData,
                    scatteredData,
                    0 );

      CPPUNIT_ASSERT( scatteredData[ 0 ] == com. getDeviceId() );
   };

   void testCommunicatorGather()
   {
      tnlCommunicator< 1, tnlHost > com;
      const int groupSize = 16;
      com. setCommunicationGroupSize( groupSize );
      com. start();
      tnlLongVector< double, tnlHost > originalData( "originalData" );
      originalData. setSize( 1 );
      originalData. setValue( com. getDeviceId() );
      tnlLongVector< double, tnlHost > gatheredData( "gatheredData" );
      if( com. getDeviceId() == 0 )
         gatheredData. setSize( com. getCommunicationGroupSize() );
      com. gather( originalData,
                   gatheredData,
                   0 );

      if( com. getDeviceId() == 0 )
      {
         cerr << gatheredData << endl;
         for( int i = 0; i < groupSize; i ++ )
            CPPUNIT_ASSERT( gatheredData[ i ] == i );
      }
   };

   void testCommunicatorBarrier()
   {
      tnlCommunicator< 1, tnlHost > com;
      const int groupSize = 16;
      com. setCommunicationGroupSize( groupSize );
      com. start();
      com. barrier();
   };
};


#endif /* TNLCOMMUNICATORTESTER_H_ */