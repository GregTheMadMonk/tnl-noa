%{
   //#define _SKIP_YYFLEXLEXER_
   //#include "scanner.ih"
   
   #include <legacy/config/tnlConfigDescriptionParser.h>
   #include <iostream>
   int yywrap();
%}

%option c++ outfile="tnlConfigDescriptionScanner.cpp" header-file="tnlConfigDescriptionScanner.h" prefix="mCDS" debug


BOOL_TRUE  "YES"|"Yes"|"yes"|"TRUE"|"True"|"true"
BOOL_FALSE "NO"|"No"|"no"|"FALSE"|"False"|"false"
EXP        "E"|"e"
DIGIT      [0-9]
DECIMAL    "."
SIGN       "+"|"-"
ID         [A-Za-z][A-Za-z0-9-]*
STRING     .*

%%
group                                         { //cout << "GROUP" << endl; 
                                                return tnlConfigDescriptionParser :: GROUP; }
string                                        { //cout << "STRING_KEYWORD" << endl;
                                                return tnlConfigDescriptionParser :: STRING_KEYWORD; }
integer                                       { //cout << "INTEGER_KEYWORD" << endl;
                                                return tnlConfigDescriptionParser :: INTEGER_KEYWORD; }
real                                          { //cout << "REAL_KEYWORD" << endl;
                                                return tnlConfigDescriptionParser :: REAL_KEYWORD; }
bool                                          { //cout << "REAL_KEYWORD" << endl;
                                                return tnlConfigDescriptionParser :: BOOLEAN_KEYWORD; }
"list of"                                     { return tnlConfigDescriptionParser :: LIST_OF_KEYWORD; }
{BOOL_TRUE}                                   { tnlConfigDescriptionParser :: current_parser -> setBVal( true );
                                                return tnlConfigDescriptionParser :: BOOLEAN; }
{BOOL_FALSE}                                  { tnlConfigDescriptionParser :: current_parser -> setBVal( false );
                                                return tnlConfigDescriptionParser :: BOOLEAN; }
\"{STRING}\"                                    { //cout << "STRING " << yytext << endl;
                                                tnlConfigDescriptionParser :: current_parser -> setSVal( yytext );
                                                return tnlConfigDescriptionParser :: STRING; }
\[.*\]                                        { //cout << "COMMENT " << yytext << endl;
                                                tnlConfigDescriptionParser :: current_parser -> 
                                                   setSVal( yytext );
                                                return tnlConfigDescriptionParser :: COMMENT; }
{ID}                                          { //cout << "IDENTIFIER " << yytext <<  endl;
                                                tnlConfigDescriptionParser :: current_parser -> 
                                                   setSVal( yytext );
                                                return tnlConfigDescriptionParser :: IDENTIFIER; }
{DIGIT}+                                      { //cout << "INTEGER " << yytext << endl;
                                                tnlConfigDescriptionParser :: current_parser -> 
                                                   setIVal( yytext ); 
                                                return tnlConfigDescriptionParser :: INTEGER; }
{DIGIT}+{DECIMAL}{DIGIT}*                     { //cout << "REAL " << yytext << endl;
                                                tnlConfigDescriptionParser :: current_parser -> 
                                                   setDVal( yytext );
                                                return tnlConfigDescriptionParser :: REAL; }
{DIGIT}+{DECIMAL}{DIGIT}*{EXP}{SIGN}?{DIGIT}+ { //cout << "REAL " << yytext << endl;
                                                tnlConfigDescriptionParser :: current_parser -> 
                                                   setDVal( yytext );
                                                return tnlConfigDescriptionParser :: REAL; }
\n                                            { tnlConfigDescriptionParser :: current_parser -> newLine(); }
#.*\n                                         { tnlConfigDescriptionParser :: current_parser -> newLine(); }
[ \t]                            
.                                             { return yytext[ 0 ]; }
%%


//extern int yywrap();
/*int mCDSFlexLexer :: yywrap()
{
   return 1;
}*/

int yywrap()
{
   return 1;
}
