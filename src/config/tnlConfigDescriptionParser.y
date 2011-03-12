%filenames tnlConfigDescriptionParser
%class-name tnlConfigDescriptionParser

%union{
   char* s_val;
   char c_val;
   int i_val;
   double d_val;
   bool b_val;
}

%token GROUP
%token STRING_KEYWORD
%token INTEGER_KEYWORD
%token REAL_KEYWORD
%token BOOLEAN_KEYWORD
%token LIST_OF_KEYWORD
%token <s_val> IDENTIFIER
%token <i_val> INTEGER
%token <d_val> REAL
%token <s_val> STRING
%token <s_val> COMMENT
%token <b_val> BOOLEAN
%type <s_val> keyword_identifier

%start config_entries

%%
config_entries: config_entry
                | config_entry config_entries
;
config_entry: keyword
              | group
;
group: GROUP group_identifier '{' keywords '}' ',' group_comment ';' { AddCurrentGroup(); }
;
keywords: keyword
          | keyword keywords
;
keyword: string_keyword |
         integer_keyword |
         real_keyword |
         boolean_keyword |
         list_of_keyword string_keyword |
         list_of_keyword integer_keyword |
         list_of_keyword real_keyword |
         list_of_keyword boolean_keyword
;
string_keyword: STRING_KEYWORD keyword_identifier';' {
                     SetCurrentEntryTypeName( "string" );
                     AddCurrentEntry( false );}           
                | STRING_KEYWORD keyword_identifier'(' '!' ')'';' {
                      SetCurrentEntryTypeName( "string" );
                      AddCurrentEntry( true );}           
                | STRING_KEYWORD keyword_identifier'(' string_value ')'';'{
                      SetCurrentEntryTypeName( "string" );
                      AddCurrentEntryWithDefaultValue();}           
                | STRING_KEYWORD keyword_identifier keyword_comment ';'{
                      SetCurrentEntryTypeName( "string" );
                      AddCurrentEntry( false );}           
                | STRING_KEYWORD keyword_identifier'(''!'')' keyword_comment ';'{
                      SetCurrentEntryTypeName( "string" );
                      AddCurrentEntry( true );}           
                | STRING_KEYWORD keyword_identifier'(' string_value ')' keyword_comment';'{
                      SetCurrentEntryTypeName( "string" );
                      AddCurrentEntryWithDefaultValue();}
;
integer_keyword: INTEGER_KEYWORD keyword_identifier';'{
                       SetCurrentEntryTypeName( "integer" );
                       AddCurrentEntry( false );}           
                 | INTEGER_KEYWORD keyword_identifier'(''!'')'';'{
                       SetCurrentEntryTypeName( "integer" );
                       AddCurrentEntry( true );}           
                 | INTEGER_KEYWORD keyword_identifier'(' integer_value ')'';'{
                       SetCurrentEntryTypeName( "integer" );
                       AddCurrentEntryWithDefaultValue();}           
                 | INTEGER_KEYWORD keyword_identifier keyword_comment ';'{
                       SetCurrentEntryTypeName( "integer" );
                       AddCurrentEntry( false );}           
                 | INTEGER_KEYWORD keyword_identifier'(''!'')' keyword_comment ';'{
                       SetCurrentEntryTypeName( "integer" );
                       AddCurrentEntry( true );}           
                 | INTEGER_KEYWORD keyword_identifier'(' integer_value ')' keyword_comment';'{
                       SetCurrentEntryTypeName( "integer" );
                       AddCurrentEntryWithDefaultValue();}
;
real_keyword: REAL_KEYWORD keyword_identifier ';'{
                    SetCurrentEntryTypeName( "real" );
                    AddCurrentEntry( false );}           
              | REAL_KEYWORD keyword_identifier '(''!'')'';'{
                    SetCurrentEntryTypeName( "real" );
                    AddCurrentEntry( true );}           
              | REAL_KEYWORD keyword_identifier'(' real_value ')'';'{
                    SetCurrentEntryTypeName( "real" );
                    AddCurrentEntryWithDefaultValue();}           
              | REAL_KEYWORD keyword_identifier keyword_comment ';'{
                    SetCurrentEntryTypeName( "real" );
                    AddCurrentEntry( false );}           
              | REAL_KEYWORD keyword_identifier '(''!'')'keyword_comment ';'{
                    SetCurrentEntryTypeName( "real" );
                    AddCurrentEntry( true );}           
              | REAL_KEYWORD keyword_identifier'(' real_value ')' keyword_comment';'{
                    SetCurrentEntryTypeName( "real" );
                    AddCurrentEntryWithDefaultValue();}           
;
boolean_keyword:  BOOLEAN_KEYWORD keyword_identifier ';'{
                        SetCurrentEntryTypeName( "bool" );
                        AddCurrentEntry( false );}           
                  | BOOLEAN_KEYWORD keyword_identifier '(''!'')'';'{
                        SetCurrentEntryTypeName( "bool" );
                        AddCurrentEntry( true );}           
                  | BOOLEAN_KEYWORD keyword_identifier'(' boolean_value ')'';'{
                        SetCurrentEntryTypeName( "bool" );
                        AddCurrentEntryWithDefaultValue();}           
                  | BOOLEAN_KEYWORD keyword_identifier keyword_comment ';'{
                        SetCurrentEntryTypeName( "bool" );
                        AddCurrentEntry( false );}           
                  | BOOLEAN_KEYWORD keyword_identifier '(''!'')'keyword_comment ';'{
                        SetCurrentEntryTypeName( "bool" );
                        AddCurrentEntry( true );}           
                  | BOOLEAN_KEYWORD keyword_identifier'(' boolean_value ')' keyword_comment';'{
                        SetCurrentEntryTypeName( "bool" );
                        AddCurrentEntryWithDefaultValue();}           
;
list_of_keyword: LIST_OF_KEYWORD { SetCurrentEntryTypeIsList( true ); }
;
group_identifier: IDENTIFIER { SetCurrentGroupId( $1 ); }
;
group_comment: COMMENT { SetCurrentGroupComment( $1 ); }
;
keyword_identifier: IDENTIFIER { SetCurrentEntryId( $1 ); }
;
keyword_comment: COMMENT { SetCurrentEntryComment( $1 ); }
;
string_value: STRING { string_default_value. SetString( $1, 1, 1 );}
;
real_value: REAL { real_default_value = $1; }
;
integer_value: INTEGER { integer_default_value = $1; }
;
boolean_value: BOOLEAN { bool_default_value = $1; }
;
%%
