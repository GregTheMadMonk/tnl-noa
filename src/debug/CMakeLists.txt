set (EXTRA_DIST tnlDebugParser.y 
	         tnlDebugScanner.l 
             create-tnl-debug-parser
     )      
set (sources tnlDebug.cpp 
          tnlDebugGroup.cpp 
          tnlDebugStructure.cpp 
	      tnlDebugParser.cpp 
	      tnlDebugScanner.cpp 
	      parse.cc
	)
set (headers tnlDbgDataElement.h 
	      tnlDebugEntry.h 
	      tnlDebugGroup.h 
	      tnlDebugParserbase.h 
	      tnlDebugParser.h 
       tnlDebug.h 
    tnlDebugScanner.h 
  tnlDebugStructure.h
	)

ADD_LIBRARY(libtnldebug${mpiExt}${debugExt}-0.1 SHARED ${sources})	#Mozna jeste prekladaci ${OPTIMISECXXFLAGS}

add_executable ( tnl-debug-test tnl-debug-test.cpp)
target_link_libraries ( tnl-debug-test  libtnldebug-0.1) 