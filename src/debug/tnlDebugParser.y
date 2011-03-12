%filenames tnlDebugParser
%class-name tnlDebugParser

%union{
   char* s_val;
   char c_val;
   int i_val;
   double d_val;
   bool b_val;
}

%token __CLASS__
%token __DEBUG__
%token __DEFAULT__
%token <b_val> __BOOL_TRUE__
%token <b_val> __BOOL_FALSE__
%token <s_val> __IDENTIFIER__

%start tnl_debug_entries

%%
tnl_debug_entries: tnl_debug_entry
                | tnl_debug_entry tnl_debug_entries
;
tnl_debug_entry: class_entry
              | function_entry
;
class_entry:__CLASS__ class_identifier class_debug_setting '{''}'  { AddCurrentGroup(); } 
          | __CLASS__ class_identifier class_debug_setting '{' function_entries '}' { AddCurrentGroup(); }
;
function_entry: function_identifier function_debug_setting { AddCurrentEntry(); }
;
class_identifier: __IDENTIFIER__ { SetCurrentClassId( $1 ); }
;
function_identifier: __IDENTIFIER__ { SetCurrentFunctionId( $1 ); }
;
function_entries: function_entry
                  | function_entry function_entries
;
class_debug_setting: '[' debug_entry ']' { SetClassDebugSettings(); }
               | '[' debug_entry ',' default_entry ']' { SetClassDebugSettings(); }
               | '[' default_entry ',' debug_entry ']' { SetClassDebugSettings(); }
;
function_debug_setting: '[' debug_entry ']'
;
debug_entry: __DEBUG__ '=' boolean_value { SetDebugValue(); }
;
default_entry: __DEFAULT__ '=' boolean_value { SetDefaultDebugValue(); }
;
boolean_value: __BOOL_TRUE__ { SetBool( $1 ); } 
             | __BOOL_FALSE__ { SetBool( $1 ); }
;
%%