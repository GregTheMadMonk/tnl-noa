set( headers DicomHeader.h
             ImageInfoObj.h
             SeriesInfoObj.h
             tnlJPEGImage.h
             tnlPGMImage.h
             tnlPNGImage.h
             tnlRegionOfInterest.h
             DicomSeries.h
             PatientInfoObj.h
             tnlImage.h
             tnlJPEGImage_impl.h
             tnlPGMImage_impl.h
             tnlPNGImage_impl.h
             tnlRegionOfInterest_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/core/io )

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

        
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/core/io )