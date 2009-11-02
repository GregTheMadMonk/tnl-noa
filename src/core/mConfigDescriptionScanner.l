%{
   //#define _SKIP_YYFLEXLEXER_
   //#include "scanner.ih"
   
   #include "mConfigDescriptionParser.h"
   #include <iostream>
   int yywrap();
%}

%option c++ outfile="mConfigDescriptionScanner.cpp" header-file="mConfigDescriptionScanner.h" prefix="mCDS" debug


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
                                                return mConfigDescriptionParser :: GROUP; }
string                                        { //cout << "STRING_KEYWORD" << endl;
                                                return mConfigDescriptionParser :: STRING_KEYWORD; }
integer                                       { //cout << "INTEGER_KEYWORD" << endl;
                                                return mConfigDescriptionParser :: INTEGER_KEYWORD; }
real                                          { //cout << "REAL_KEYWORD" << endl;
                                                return mConfigDescriptionParser :: REAL_KEYWORD; }
bool                                          { //cout << "REAL_KEYWORD" << endl;
                                                return mConfigDescriptionParser :: BOOLEAN_KEYWORD; }
"list of"                                     { return mConfigDescriptionParser :: LIST_OF_KEYWORD; }
{BOOL_TRUE}                                   { mConfigDescriptionParser :: current_parser -> setBVal( true );
                                                return mConfigDescriptionParser :: BOOLEAN; }
{BOOL_FALSE}                                  { mConfigDescriptionParser :: current_parser -> setBVal( false );
                                                return mConfigDescriptionParser :: BOOLEAN; }
\"{STRING}\"                                    { //cout << "STRING " << yytext << endl;
                                                mConfigDescriptionParser :: current_parser -> setSVal( yytext );
                                                return mConfigDescriptionParser :: STRING; }
\[.*\]                                        { //cout << "COMMENT " << yytext << endl;
                                                mConfigDescriptionParser :: current_parser -> 
                                                   setSVal( yytext );
                                                return mConfigDescriptionParser :: COMMENT; }
{ID}                                          { //cout << "IDENTIFIER " << yytext <<  endl;
                                                mConfigDescriptionParser :: current_parser -> 
                                                   setSVal( yytext );
                                                return mConfigDescriptionParser :: IDENTIFIER; }
{DIGIT}+                                      { //cout << "INTEGER " << yytext << endl;
                                                mConfigDescriptionParser :: current_parser -> 
                                                   setIVal( yytext ); 
                                                return mConfigDescriptionParser :: INTEGER; }
{DIGIT}+{DECIMAL}{DIGIT}*                     { //cout << "REAL " << yytext << endl;
                                                mConfigDescriptionParser :: current_parser -> 
                                                   setDVal( yytext );
                                                return mConfigDescriptionParser :: REAL; }
{DIGIT}+{DECIMAL}{DIGIT}*{EXP}{SIGN}?{DIGIT}+ { //cout << "REAL " << yytext << endl;
                                                mConfigDescriptionParser :: current_parser -> 
                                                   setDVal( yytext );
                                                return mConfigDescriptionParser :: REAL; }
\n                                            { mConfigDescriptionParser :: current_parser -> newLine(); }
#.*\n                                         { mConfigDescriptionParser :: current_parser -> newLine(); }
[ \t]                            
.                                             { return yytext[ 0 ]; }
%%

int yywrap()
{
   return 1;
}
