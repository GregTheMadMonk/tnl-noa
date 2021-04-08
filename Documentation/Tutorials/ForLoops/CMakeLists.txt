ADD_EXECUTABLE( StaticForExample StaticForExample_ug.cpp )
ADD_CUSTOM_COMMAND( COMMAND StaticForExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticForExample.out OUTPUT StaticForExample.out )

ADD_EXECUTABLE( TemplateStaticForExample TemplateStaticForExample_ug.cpp )
ADD_CUSTOM_COMMAND( COMMAND TemplateStaticForExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TemplateStaticForExample.out OUTPUT TemplateStaticForExample.out )

ADD_CUSTOM_TARGET( ForLoops ALL DEPENDS
   StaticForExample.out
   TemplateStaticForExample.out
)
