#ifndef tnlConfigDescriptionParser_h_included
#define tnlConfigDescriptionParser_h_included

// for error()'s inline implementation
#include <iostream>

// $insert baseclass
#include "tnlConfigDescriptionParserbase.h"
#include "tnlConfigDescription.h"

#undef tnlConfigDescriptionParser

class mCDSFlexLexer;

using namespace std;

class tnlConfigDescriptionParser: public tnlConfigDescriptionParserBase
{
    public:
    tnlConfigDescriptionParser();

    void setScanner( istream* in_stream );

    int runParsing( tnlConfigDescription* conf_desc );
        
    int parse();

    // Methods called by the scanner
    void newLine();

    void setSVal( char* s );

    void setIVal( char* s );

    void setDVal( char* s );
    
    void setBVal( bool b );
      
    static tnlConfigDescriptionParser* current_parser;

    //Methods for the parsing
    void SetCurrentGroupId( const char* id );

    void SetCurrentGroupComment( const char* comment );

    void AddCurrentGroup();

    void SetCurrentEntryTypeName( const char* _basic_type );

    void SetCurrentEntryTypeIsList( const bool _list_entry );

    void SetCurrentEntryId( const char* id );
    
    void SetCurrentEntryComment( const char* comment );

    void AddCurrentEntry( bool required );
    
    void AddCurrentEntryWithDefaultValue();

    ~tnlConfigDescriptionParser();

    protected:
    
    mCDSFlexLexer* scanner;

    tnlConfigDescription* config_description;
    
    int line;

    tnlString current_group_name;
    
    tnlString current_group_comment;

    tnlString current_entry_name;

    tnlConfigEntryType current_entry_type;
    
    tnlString current_entry_comment;
    
    bool current_entry_is_list;

    bool bool_default_value;

    double real_default_value;

    int integer_default_value;

    tnlString string_default_value;

    bool parse_error;

    private:
        void error(char const *msg);    // called on (syntax) errors
        int lex();                      // returns the next token from the
                                        // lexical scanner. 
        void print__();                   // use, e.g., d_token, d_loc

        void print(){};
    // support functions for parse():
        void executeAction(int ruleNr);
        void errorRecovery();
        int lookup( bool recovery );
        void nextToken();
};

inline void tnlConfigDescriptionParser :: error(char const *msg)
{
    std::cerr << msg << " at line " << line << std::endl;
}

// $insert lex

//inline void tnlConfigDescriptionParser :: print__()      // use d_token, d_loc
//{}


#endif
