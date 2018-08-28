set( headers tnlGridTester.h )

ADD_EXECUTABLE( tnlGridTest ${headers} tnlGridTest.cpp )
TARGET_LINK_LIBRARIES( tnlGridTest ${CPPUNIT_LIBRARIES} tnl )
