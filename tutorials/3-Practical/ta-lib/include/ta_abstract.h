/* TA-LIB Copyright (c) 1999-2007, Mario Fortier
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in
 *   the documentation and/or other materials provided with the
 *   distribution.
 *
 * - Neither name of author nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef TA_ABSTRACT_H
#define TA_ABSTRACT_H

#ifndef TA_COMMON_H
    #include "ta_common.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* This file defines the interface for calling all the TA functions without
 * knowing at priori the parameters.
 *
 * This capability is particularly useful for an application who needs
 * to support the complete list of TA functions without having to
 * re-write new code each time a new function is added to TA-Lib.
 *
 * Example 1:
 *        Lets say you are doing a charting software. When the user select
 *        a price bar, a side list offers blindly all the TA functions
 *        that could be applied to a price bar. The user selects one of
 *        these, then a dialog open for allowing to adjust some parameter
 *        (TA-LIB will tell your software which parameter are needed and the
 *        valid range for each). Once all the parameter are set, you can
 *        call blindly the corresponding TA function. The returned
 *        information can then also blindly be drawn on the chart (some
 *        output flags allows to get some hint about how the data shall be
 *        displayed).
 *        The same "abstract" logic apply to all the TA functions.
 *        Some TA Functions works only on volume, or can work indiferently
 *        with any time serie data (the open, close, another indicator...).
 *        All the applicable functions to the currently selected/available
 *        data can be determined through this "virtual" interface.
 *
 * Example 2:
 *        Let's say you do not like the direct interface for
 *        calling the TA Functions, you can write a code that
 *        re-generate a different interface. This is already
 *        done even for the 'C' interface (ta_func.h is generated).
 *
 * Example 3:
 *        With the abstract interface you can easily generate
 *        glue code. Like generating an interface that integrates
 *        well within Perl... see the project SWIG if you are
 *        interested by such things.
 *
 * The abstract interface is used within TA-Lib to perform at least 
 * the following:
 *   - used by gen_code to generate all the glue code.
 *   - used by the Excel interface to call all the TA functions.
 *   - used to generate a XML representation of the TA functions.
 */

/* The following functions are used to obtain the name of all the
 * TA function groups ("Market Strength", "Trend Indicator" etc...).
 *
 * On success, it becomes the responsibility of the caller to
 * call TA_GroupTableFree once the 'table' is no longuer needed.
 *
 * Example:
 * This code snippet will print out the name of all the supported
 * function group available:
 *
 *   TA_StringTable *table;
 *   TA_RetCode retCode;
 *   int i;
 *
 *   retCode = TA_GroupTableAlloc( &table );
 *
 *   if( retCode == TA_SUCCESS )
 *   {
 *      for( i=0; i < table->size; i++ )
 *         printf( "%s\n", table->string[i] );
 *
 *      TA_GroupTableFree( table );
 *   }
 */
TA_RetCode TA_GroupTableAlloc( TA_StringTable **table );
TA_RetCode TA_GroupTableFree ( TA_StringTable *table );

/* The following functions are used to obtain the name of all the
 * function provided in a certain group.
 *
 * On success, it becomes the responsibility of the caller to
 * call TA_FuncTableFree once the 'table' is no longuer needed.
 *
 * Passing NULL as the group string will return ALL the TA functions.
 * (Note: All TA_Functions have a unique string identifier even when in
 *        seperate group).
 *
 * Example:
 * This code snippet will print out the name of all the supported
 * function in the "market strength" category:
 *
 *   TA_StringTable *table;
 *   TA_RetCode retCode;
 *   int i;
 *
 *   retCode = TA_FuncTableAlloc( "Market Strength", &table );
 *
 *   if( retCode == TA_SUCCESS )
 *   {
 *      for( i=0; i < table->size; i++ )
 *         printf( "%s\n", table->string[i] );
 *
 *      TA_FuncTableFree( table );
 *   }
 */
TA_RetCode TA_FuncTableAlloc( const char *group, TA_StringTable **table );                              
TA_RetCode TA_FuncTableFree ( TA_StringTable *table );

/* Using the name, you can obtain an handle unique to this function.
 * This handle is further used for obtaining information on the
 * parameters needed and also for potentially calling this TA function.
 *
 * For convenience, this handle can also be found in
 * the TA_FuncInfo structure (see below).
 */
typedef unsigned int TA_FuncHandle;
TA_RetCode TA_GetFuncHandle( const char *name,
                             const TA_FuncHandle **handle );

/* Get some basic information about a function.
 *
 * A const pointer will be set on the corresponding TA_FuncInfo structure.
 * The whole structure are hard coded constants and it can be assumed they
 * will not change at runtime.
 *
 * Example:
 *   Print the number of inputs used by the MA (moving average) function.
 *
 *   TA_RetCode retCode;
 *   TA_FuncHandle *handle;
 *   const TA_FuncInfo *theInfo;
 *
 *   retCode = TA_GetFuncHandle( "MA", &handle );
 *
 *   if( retCode == TA_SUCCESS )
 *   {
 *      retCode = TA_GetFuncInfo( handle, &theInfo );
 *      if( retCode == TA_SUCCESS )
 *         printf( "Nb Input = %d\n", theInfo->nbInput );
 *   }
 *
 */
typedef int TA_FuncFlags;
#define TA_FUNC_FLG_OVERLAP   0x01000000 /* Output scale same as input data. */
#define TA_FUNC_FLG_VOLUME    0x04000000 /* Output shall be over the volume data. */
#define TA_FUNC_FLG_UNST_PER  0x08000000 /* Indicate if this function have an unstable 
                                          * initial period. Some additional code exist
                                          * for these functions for allowing to set that
                                          * unstable period. See Documentation.
                                          */
#define TA_FUNC_FLG_CANDLESTICK 0x10000000 /* Output shall be a candlestick */

typedef struct TA_FuncInfo
{
   /* Constant information about the function. The
    * information found in this structure is guarantee
    * to not change at runtime.
    */
   const char * name;
   const char * group;

   const char * hint;
   const char * camelCaseName;
   TA_FuncFlags flags;

   unsigned int nbInput;
   unsigned int nbOptInput;
   unsigned int nbOutput;

   const TA_FuncHandle *handle;
} TA_FuncInfo;

TA_RetCode TA_GetFuncInfo( const TA_FuncHandle *handle,
                           const TA_FuncInfo **funcInfo );


/* An alternate way to access all the functions is through the
 * use of the TA_ForEachFunc(). You can setup a function to be
 * call back for each TA function in the TA-Lib.
 *
 * Example:
 *  This code will print the group and name of all available functions.
 *
 *  void printFuncInfo( const TA_FuncInfo *funcInfo, void *opaqueData )
 *  {
 *     printf( "Group=%s Name=%s\n", funcInfo->group, funcInfo->name );
 *  }
 *
 *  void displayListOfTAFunctions( void )
 *  {
 *     TA_ForEachFunc( printFuncInfo, NULL );
 *  }
 */
typedef void (*TA_CallForEachFunc)(const TA_FuncInfo *funcInfo, void *opaqueData );

TA_RetCode TA_ForEachFunc( TA_CallForEachFunc functionToCall, void *opaqueData );

/* The next section includes the data structures and function allowing to
 * proceed with the call of a Tech. Analysis function.
 *
 * At first, it may seems a little bit complicated, but it is worth to take the
 * effort to learn about it. Once you succeed to interface with TA-Abstract you get
 * access to the complete library of TA functions at once without further effort.
 */

/* Structures representing extended information on a parameter. */

typedef struct TA_RealRange
{
   TA_Real     min;
   TA_Real     max;
   TA_Integer  precision; /* nb of digit after the '.' */

   /* The following suggested value are used by Tech. Analysis software
    * doing parameter "optimization". Can be ignored by most user.
    */
   TA_Real     suggested_start;
   TA_Real     suggested_end;
   TA_Real     suggested_increment;
} TA_RealRange;

typedef struct TA_IntegerRange
{
   TA_Integer  min;
   TA_Integer  max;

   /* The following suggested value are used by Tech. Analysis software
    * doing parameter "optimization". Can be ignored by most user.
    */
   TA_Integer  suggested_start;
   TA_Integer  suggested_end;
   TA_Integer  suggested_increment;
} TA_IntegerRange;

typedef struct TA_RealDataPair
{
   /* A TA_Real value and the corresponding string. */
   TA_Real     value;
   const char *string;
} TA_RealDataPair;

typedef struct TA_IntegerDataPair
{
   /* A TA_Integer value and the corresponding string. */
   TA_Integer  value;
   const char *string;
} TA_IntegerDataPair;

typedef struct TA_RealList
{
   const TA_RealDataPair *data;
   unsigned int nbElement;
} TA_RealList;

typedef struct TA_IntegerList
{
   const TA_IntegerDataPair *data;
   unsigned int nbElement;
} TA_IntegerList;

typedef enum
{
   TA_Input_Price,
   TA_Input_Real,
   TA_Input_Integer
} TA_InputParameterType;

typedef enum
{
   TA_OptInput_RealRange,
   TA_OptInput_RealList,
   TA_OptInput_IntegerRange,
   TA_OptInput_IntegerList
} TA_OptInputParameterType;

typedef enum
{
   TA_Output_Real,
   TA_Output_Integer
} TA_OutputParameterType;

/* When the input is a TA_Input_Price, the following 
 * TA_InputFlags indicates the required components.
 * These can be combined for functions requiring more
 * than one component.
 *
 * Example:
 *   (TA_IN_PRICE_OPEN|TA_IN_PRICE_HIGH)
 *   Indicates that the functions requires two inputs
 *   that must be specifically the open and the high.
 */
typedef int TA_InputFlags;
#define TA_IN_PRICE_OPEN         0x00000001
#define TA_IN_PRICE_HIGH         0x00000002
#define TA_IN_PRICE_LOW          0x00000004
#define TA_IN_PRICE_CLOSE        0x00000008
#define TA_IN_PRICE_VOLUME       0x00000010
#define TA_IN_PRICE_OPENINTEREST 0x00000020
#define TA_IN_PRICE_TIMESTAMP    0x00000040

/* The following are flags for optional inputs.
 *
 * TA_OPTIN_IS_PERCENT:  Input is a percentage.
 *
 * TA_OPTIN_IS_DEGREE:   Input is a degree (0-360).
 *
 * TA_OPTIN_IS_CURRENCY: Input is a currency.
 *
 * TA_OPTIN_ADVANCED:
 *    Used for parameters who are rarely changed.
 *    Most application can hide these advanced optional inputs to their
 *    end-user (or make it harder to change).
 */
typedef int TA_OptInputFlags;
#define TA_OPTIN_IS_PERCENT   0x00100000 /* Input is a percentage.  */
#define TA_OPTIN_IS_DEGREE    0x00200000 /* Input is a degree (0-360). */
#define TA_OPTIN_IS_CURRENCY  0x00400000 /* Input is a currency. */
#define TA_OPTIN_ADVANCED     0x01000000
 

/* The following are flags giving hint on what
 * could be done with the output.
 */
typedef int TA_OutputFlags;
#define TA_OUT_LINE              0x00000001 /* Suggest to display as a connected line. */
#define TA_OUT_DOT_LINE          0x00000002 /* Suggest to display as a 'dotted' line. */
#define TA_OUT_DASH_LINE         0x00000004 /* Suggest to display as a 'dashed' line. */
#define TA_OUT_DOT               0x00000008 /* Suggest to display with dots only. */
#define TA_OUT_HISTO             0x00000010 /* Suggest to display as an histogram. */
#define TA_OUT_PATTERN_BOOL      0x00000020 /* Indicates if pattern exists (!=0) or not (0) */
#define TA_OUT_PATTERN_BULL_BEAR 0x00000040 /* =0 no pattern, > 0 bullish, < 0 bearish */
#define TA_OUT_PATTERN_STRENGTH  0x00000080 /* =0 neutral, ]0..100] getting bullish, ]100..200] bullish, [-100..0[ getting bearish, [-200..100[ bearish */
#define TA_OUT_POSITIVE          0x00000100 /* Output can be positive */
#define TA_OUT_NEGATIVE          0x00000200 /* Output can be negative */
#define TA_OUT_ZERO              0x00000400 /* Output can be zero */
#define TA_OUT_UPPER_LIMIT       0x00000800 /* Indicates that the values represent an upper limit. */
#define TA_OUT_LOWER_LIMIT       0x00001000 /* Indicates that the values represent a lower limit. */


/* The following 3 structures will exist for each input, optional
 * input and output.
 *
 * These structures tells you everything you need to know for identifying
 * the parameters applicable to the function.
 */
typedef struct TA_InputParameterInfo
{
   TA_InputParameterType type;
   const char           *paramName;
   TA_InputFlags         flags;

} TA_InputParameterInfo;

typedef struct TA_OptInputParameterInfo
{
   TA_OptInputParameterType type;
   const char              *paramName;
   TA_OptInputFlags         flags;

   const char *displayName;
   const void *dataSet;
   TA_Real     defaultValue;
   const char *hint;
   const char *helpFile;

} TA_OptInputParameterInfo;

typedef struct TA_OutputParameterInfo
{
   TA_OutputParameterType type;
   const char            *paramName;
   TA_OutputFlags         flags;

} TA_OutputParameterInfo;

/* Functions to get a const ptr on the "TA_XXXXXXParameterInfo".
 * Max value for the 'paramIndex' can be found in the TA_FuncInfo structure.
 * The 'handle' can be obtained from TA_GetFuncHandle or from a TA_FuncInfo.
 *
 * Short example:
 *
 * void displayInputType( const TA_FuncInfo *funcInfo )
 * {
 *    unsigned int i;
 *    const TA_InputParameterInfo *paramInfo;
 *
 *    for( i=0; i < funcInfo->nbInput; i++ )
 *    {
 *       TA_GetInputParameterInfo( funcInfo->handle, i, &paramInfo );
 *       switch( paramInfo->type )
 *       {
 *       case TA_Input_Price:
 *          printf( "This function needs price bars as input\n" );
 *          break;
 *       case TA_Input_Real:
 *          printf( "This function needs an array of floats as input\n" );
 *          break;
 *        (... etc. ...)
 *       }
 *    }
 * }
 */
TA_RetCode TA_GetInputParameterInfo( const TA_FuncHandle *handle,
                                     unsigned int paramIndex,
                                     const TA_InputParameterInfo **info );

TA_RetCode TA_GetOptInputParameterInfo( const TA_FuncHandle *handle,
                                        unsigned int paramIndex,
                                        const TA_OptInputParameterInfo **info );

TA_RetCode TA_GetOutputParameterInfo( const TA_FuncHandle *handle,
                                      unsigned int paramIndex,
                                      const TA_OutputParameterInfo **info );

/* Alloc a structure allowing to build the list of parameters
 * for doing a call.
 *
 * All input and output parameters must be setup. If not, TA_BAD_PARAM
 * will be returned when TA_CallFunc is called.
 *
 * The optional input are not required to be setup. A default value
 * will always be used in that case.
 *
 * If there is an attempts to set a parameter with the wrong function
 * (and thus the wrong type), TA_BAD_PARAM will be immediatly returned.
 *
 * Although this mechanism looks complicated, it is written for being fairly solid.
 * If you provide a wrong parameter value, or wrong type, or wrong pointer etc. the
 * library shall return TA_BAD_PARAM or TA_BAD_OBJECT and not hang.
 */
typedef struct TA_ParamHolder
{
  /* Implementation is hidden. */
  void *hiddenData;
} TA_ParamHolder; 

TA_RetCode TA_ParamHolderAlloc( const TA_FuncHandle *handle,
                                TA_ParamHolder **allocatedParams );

TA_RetCode TA_ParamHolderFree( TA_ParamHolder *params );

/* Setup the values of the data input parameters. 
 *
 * paramIndex is zero for the first input.
 */
TA_RetCode TA_SetInputParamIntegerPtr( TA_ParamHolder *params,
                                       unsigned int paramIndex,
                                       const TA_Integer *value );

TA_RetCode TA_SetInputParamRealPtr( TA_ParamHolder *params,
                                    unsigned int paramIndex,
                                    const TA_Real *value );

TA_RetCode TA_SetInputParamPricePtr( TA_ParamHolder *params,
                                     unsigned int paramIndex,
                                     const TA_Real      *open,
                                     const TA_Real      *high,
                                     const TA_Real      *low,
                                     const TA_Real      *close,
                                     const TA_Real      *volume,
                                     const TA_Real      *openInterest );

/* Setup the values of the optional input parameters.
 * If an optional input is not set, a default value will be used.
 *
 * paramIndex is zero for the first optional input.
 */
TA_RetCode TA_SetOptInputParamInteger( TA_ParamHolder *params,
                                       unsigned int paramIndex,
                                       TA_Integer optInValue );

TA_RetCode TA_SetOptInputParamReal( TA_ParamHolder *params,
                                    unsigned int paramIndex,
                                    TA_Real optInValue );

/* Setup the parameters indicating where to store the output.
 *
 * The caller is responsible to allocate sufficient memory. A safe bet is to
 * always do: nb of output elements  == (endIdx-startIdx+1)
 *
 * paramIndex is zero for the first output.
 *
 */
TA_RetCode TA_SetOutputParamIntegerPtr( TA_ParamHolder *params,
                                        unsigned int paramIndex,
                                        TA_Integer     *out );

TA_RetCode TA_SetOutputParamRealPtr( TA_ParamHolder *params,
                                     unsigned int paramIndex,
                                     TA_Real        *out );

/* Once the optional parameter are set, it is possible to 
 * get the lookback for this call. This information can be
 * used to calculate the optimal size for the output buffers.
 * (See the documentation for method to calculate the output size).
 */
TA_RetCode TA_GetLookback( const TA_ParamHolder *params,
                           TA_Integer *lookback );

/* Finally, call the TA function with the parameters. 
 *
 * The TA function who is going to be called was specified
 * when the TA_ParamHolderAlloc was done.
 */
TA_RetCode TA_CallFunc( const TA_ParamHolder *params,
                        TA_Integer            startIdx,
                        TA_Integer            endIdx,
                        TA_Integer           *outBegIdx,
                        TA_Integer           *outNbElement );


/* Return XML representation of all the TA functions.
 * The returned array is the same as the ta_func_api.xml file.
 */
const char *TA_FunctionDescriptionXML( void );

#ifdef __cplusplus
}
#endif

#endif

