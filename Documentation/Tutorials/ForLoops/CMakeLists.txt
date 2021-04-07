ADD_EXECUTABLE( TemplateStaticForExample TemplateStaticForExample_ug.cpp )
ADD_CUSTOM_COMMAND( COMMAND TemplateStaticForExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TemplateStaticForExample.out OUTPUT TemplateStaticForExample.out )

ADD_CUSTOM_TARGET( ForLoops ALL DEPENDS
   TemplateStaticForExample.out
)
