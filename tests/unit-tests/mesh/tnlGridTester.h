/***************************************************************************
                          tnlGridTester.h  -  description
                             -------------------
    begin                : Jul 28, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */
#ifndef TNLGRIDTESTER_H_
#define TNLGRIDTESTER_H_


#ifdef HAVE_CPPUNIT
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/Message.h>
#include <mesh/tnlGrid.h>


template< int Dimensions, typename RealType, typename Device, typename IndexType >
class tnlGridTester{};

#include "tnlGrid1DTester.h"
#include "tnlGrid2DTester.h"
#include "tnlGrid3DTester.h"

#endif /* HAVE_CPPUNIT */

#endif /* TNLGRIDTESTER_H_ */
