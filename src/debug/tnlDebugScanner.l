%{
   //#define _SKIP_YYFLEXLEXER_
   //#include "scanner.ih"
   
   #include "tnlDebugParser.h"
   #include <iostream>

   int yywrap();
%}

%option c++ outfile="tnlDebugScanner.cpp" header-file="tnlDebugScanner.h" prefix="tnlDebug" debug  


BOOL_TRUE  "YES"|"Yes"|"yes"|"TRUE"|"True"|"true"
BOOL_FALSE "NO"|"No"|"no"|"FALSE"|"False"|"false"
ID         [A-Za-z][A-Za-z0-9-]*

%%
class                                         { //cout << "CLASS" << endl; 
                                                return tnlDebugParser :: __CLASS__; }
debug                                         { //cout << "DEBUG" << endl;
                                                return tnlDebugParser :: __DEBUG__; }
default-debug                                 { //cout << "DEFAULT" << endl;
                                                return tnlDebugParser :: __DEFAULT__; }
{BOOL_TRUE}                                   { // cout << "BOOL TRUE" << endl;
                                                tnlDebugParser :: current_parser -> 
                                                   setBVal( true );
                                                return tnlDebugParser :: __BOOL_TRUE__;}

{BOOL_FALSE}                                  { //cout << "BOOL FALSE" << endl;
                                                tnlDebugParser :: current_parser ->
                                                   setBVal( false );
                                                return tnlDebugParser :: __BOOL_FALSE__;}
{ID}                                          { //cout << "IDENTIFIER " << yytext <<  endl;
                                                tnlDebugParser :: current_parser -> 
                                                   setSVal( yytext );
                                                return tnlDebugParser :: __IDENTIFIER__; }
\n                                            { tnlDebugParser :: current_parser -> newLine(); }
#.*\n                                         { tnlDebugParser :: current_parser -> newLine(); }
[ \t]                            
.                                             { return yytext[ 0 ]; }
%%

int tnlDebugFlexLexer::yywrap()
{
   return 1;
}

extern int yywrap();
/*{
   return 1;
}*/
