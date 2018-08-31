set( headers DicomHeader.h
             DicomHeader_impl.h
             DicomImageInfo.h
             DicomImageInfo_impl.h
             DicomSeriesInfo.h
             DicomSeriesInfo_impl.h
             DicomSeries.h
             DicomSeries_impl.h
             DicomPatientInfo.h
             DicomPatientInfo_impl.h
             Image.h
             JPEGImage.h
             JPEGImage_impl.h
             PGMImage.h
             PGMImage_impl.h
             PNGImage.h
             PNGImage_impl.h            
             RegionOfInterest.h                                                    
             RegionOfInterest_impl.h )

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Images )

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
   set( tnl_images_CUDA__SOURCES
        ${common_SOURCES}             
        PARENT_SCOPE )
ELSE()
    set( tnl_images_SOURCES     
         ${common_SOURCES}          
         PARENT_SCOPE )
ENDIF()    

        
INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Images )