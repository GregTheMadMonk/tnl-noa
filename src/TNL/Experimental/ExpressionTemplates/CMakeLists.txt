set( headers StaticVectorExpressions.h
	     VectorExpressions.h
)

#IF( BUILD_CUDA )
#    CUDA_ADD_EXECUTABLE( tnl-expression-templates expression-templates.cu )
#ELSE(  BUILD_CUDA )
#    ADD_EXECUTABLE( tnl-expression-templates expression-templates.cpp )
#    ADD_EXECUTABLE( tnl-expression-templates-static expression-templates-static.cpp )
#ENDIF( BUILD_CUDA )
#
#INSTALL( TARGETS tnl-expression-templates
#         RUNTIME DESTINATION bin
#         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
#
#INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Experimental/ExpressionTemplates )
