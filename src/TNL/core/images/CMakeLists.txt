set( headers tnlDicomHeader.h
             tnlDicomHeader_impl.h
             tnlDicomImageInfo.h
             tnlDicomImageInfo_impl.h
             tnlDicomSeriesInfo.h
             tnlDicomSeriesInfo_impl.h
             tnlDicomSeries.h
             tnlDicomSeries_impl.h
             tnlDicomPatientInfo.h
             tnlDicomPatientInfo_impl.h
             tnlImage.h
             tnlJPEGImage.h
             tnlJPEGImage_impl.h
             tnlPGMImage.h
             tnlPGMImage_impl.h
             tnlPNGImage.h
             tnlPNGImage_impl.h            
             tnlRegionOfInterest.h                                                    
             tnlRegionOfInterest_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/core/images )

####
# DCMTK currently does not support shared libraries. When it does _impl.h files
# could be replaced with .cpp and DCMTK might be link with TNL.
#
if( HAVE_DCMTK )
    set( common_SOURCES )       
else()
   set( common_SOURCES )
endif()

IF( BUILD_CUDA )
   set( tnl_core_images_CUDA__SOURCES
        ${common_SOURCES}             
        PARENT_SCOPE )
ELSE()
    set( tnl_core_images_SOURCES     
         ${common_SOURCES}          
         PARENT_SCOPE )
ENDIF()    

        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/core/images )