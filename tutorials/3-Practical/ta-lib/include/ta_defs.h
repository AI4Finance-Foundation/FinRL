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
#ifndef TA_DEFS_H
#define TA_DEFS_H

/*** The following block of code is to define:
 ***
 ***    UInt32  : 32 bits unsigned integer.
 ***    Int32   : 32 bits signed integer.
 ***    UInt64  : 64 bits unsigned integer.
 ***    Int64   : 64 bits signed integer.
 ***
 ***    INT_MIN : The minimal value for Int32
 ***    INT_MAX : The maximal value for Int32
 ***/
#ifndef FD_DEFS_H
  #if defined( _MANAGED )
    /* Int32, UInt32, Int64 and UInt64 are built-in for .NET */	
    #define INT_MIN (Int32::MinValue)
    #define INT_MAX (Int32::MaxValue)
  #elif defined( _JAVA )
    #define INT_MIN Integer.MIN_VALUE
    #define INT_MAX Integer.MAX_VALUE
  #else
    #include <limits.h>

    /* Identify if 64 bits platform with __64BIT__.
     * Can also be done from compiler command line. 
     */
    #if defined(_WIN64)
       #define __64BIT__ 1
    #endif

    #if defined( __LP64__ ) || defined( _LP64 )
       #define __64BIT__ 1
    #endif

    /* Check also for some known GCC def for 64 bits platform. */
    #if defined(__alpha__)\
        ||defined(__ia64__)\
        ||defined(__ppc64__)\
        ||defined(__s390x__)\
        ||defined(__x86_64__)
       #define __64BIT__ 1
    #endif		  
		   
    #if !defined(__MACTYPES__)
        typedef signed int   Int32;
        typedef unsigned int UInt32;

        #if defined(_WIN32) || defined(_WIN64)
           /* See "Windows Data Types". Platform SDK. MSDN documentation. */
           typedef signed __int64   Int64;
           typedef unsigned __int64 UInt64;
        #else
           #if defined(__64BIT__)
              /* Standard LP64 model for 64 bits Unix platform. */
              typedef signed long   Int64;
              typedef unsigned long UInt64;
           #else
              /* Standard ILP32 model for 32 bits Unix platform. */
              typedef signed long long   Int64;
              typedef unsigned long long UInt64;
           #endif
         #endif
    #endif
  #endif
#endif

/* Enumeration and macros to abstract syntax differences
 * between C, C++, managed C++ and Java.
 */
#if defined( _MANAGED )

  /* CMJ is the "CManagedJava" macro. It allows to write variant
   * for the 3 different languages.
   */
  #define CMJ(c,managed,java) managed

  /* Enumeration abstraction */
  #define ENUM_BEGIN(w) enum class w {  
  #define ENUM_DEFINE(x,y) y
  #define ENUM_VALUE(w,x,y) (w::y)
  #define ENUM_CASE(w,x,y)  (w::y)
  #define ENUM_DECLARATION(w) w
  #define ENUM_END(w) };  

  /* Structure abstraction */
  #define STRUCT_BEGIN(x) struct x {
  #define STRUCT_END(x) };

  /* Pointer/reference abstraction */
  #define VALUE_HANDLE_INT(name)           int name
  #define VALUE_HANDLE_DEREF(name)         name
  #define VALUE_HANDLE_DEREF_TO_ZERO(name) name = 0
  #define VALUE_HANDLE_OUT(name)           name

  #define VALUE_HANDLE_GET(name)         name
  #define VALUE_HANDLE_SET(name,x)       name = x

  /* Misc. */
  #define CONSTANT_DOUBLE(x) const double x
  #define NAMESPACE(x) x::
  #define UNUSED_VARIABLE(x) (void)x

#elif defined( _JAVA )
  #define CMJ(c,managed,java) java

  #define ENUM_BEGIN(w) public enum w {
  #define ENUM_DEFINE(x,y) y
  #define ENUM_VALUE(w,x,y) w.y
  #define ENUM_CASE(w,x,y) y
  #define ENUM_DECLARATION(w) w
  #define ENUM_END(w) };

  #define STRUCT_BEGIN(x) public class x {
  #define STRUCT_END(x) };

  #define VALUE_HANDLE_INT(name)            MInteger name = new MInteger()
  #define VALUE_HANDLE_DEREF(name)          name.value
  #define VALUE_HANDLE_DEREF_TO_ZERO(name)  name.value = 0
  #define VALUE_HANDLE_OUT(name)            name

  #define VALUE_HANDLE_GET(name)         name.value
  #define VALUE_HANDLE_SET(name,x)       name.value = x

  #define CONSTANT_DOUBLE(x) final double x
  #define NAMESPACE(x) x.
  #define UNUSED_VARIABLE(x)

#else

  #define CMJ(c,managed,java) c

  #define ENUM_BEGIN(w) typedef enum {
  #define ENUM_DEFINE(x,y) x
  #define ENUM_VALUE(w,x,y) x
  #define ENUM_CASE(w,x,y) x
  #define ENUM_DECLARATION(w) TA_##w
  #define ENUM_END(w) } TA_##w;

  #define STRUCT_BEGIN(x) typedef struct {
  #define STRUCT_END(x) } x;

  #define VALUE_HANDLE_INT(name)           int name
  #define VALUE_HANDLE_DEREF(name)         (*name)
  #define VALUE_HANDLE_DEREF_TO_ZERO(name) (*name) = 0
  #define VALUE_HANDLE_OUT(name)           &name

  #define VALUE_HANDLE_GET(name)          name
  #define VALUE_HANDLE_SET(name,x)        name = x

  #define CONSTANT_DOUBLE(x) const double x
  #define NAMESPACE(x)
  #define UNUSED_VARIABLE(x) (void)x
#endif

/* Abstraction of function calls within the library.
 * Needed because Java/.NET allows overloading, while for C the
 * TA_PREFIX allows to select variant of the same function.
 */
#define FUNCTION_CALL(x)        TA_PREFIX(x)
#define FUNCTION_CALL_DOUBLE(x) TA_##x
#define LOOKBACK_CALL(x)        TA_##x##_Lookback

/* min/max value for a TA_Integer */
#define TA_INTEGER_MIN (INT_MIN+1)
#define TA_INTEGER_MAX (INT_MAX)

/* min/max value for a TA_Real 
 *
 * Use fix value making sense in the
 * context of TA-Lib (avoid to use DBL_MIN
 * and DBL_MAX standard macro because they
 * are troublesome with some compiler).
 */
#define TA_REAL_MIN (-3e+37)
#define TA_REAL_MAX (3e+37)

/* A value outside of the min/max range 
 * indicates an undefined or default value.
 */
#define TA_INTEGER_DEFAULT (INT_MIN)
#define TA_REAL_DEFAULT    (-4e+37)

/* Part of this file is generated by gen_code */

ENUM_BEGIN( RetCode )    
    /*      0 */  ENUM_DEFINE( TA_SUCCESS, Success ),            /* No error */
    /*      1 */  ENUM_DEFINE( TA_LIB_NOT_INITIALIZE, LibNotInitialize ), /* TA_Initialize was not sucessfully called */
    /*      2 */  ENUM_DEFINE( TA_BAD_PARAM, BadParam ), /* A parameter is out of range */
    /*      3 */  ENUM_DEFINE( TA_ALLOC_ERR, AllocErr ), /* Possibly out-of-memory */
    /*      4 */  ENUM_DEFINE( TA_GROUP_NOT_FOUND, GroupNotFound ),
    /*      5 */  ENUM_DEFINE( TA_FUNC_NOT_FOUND, FuncNotFound ),
    /*      6 */  ENUM_DEFINE( TA_INVALID_HANDLE, InvalidHandle ),
    /*      7 */  ENUM_DEFINE( TA_INVALID_PARAM_HOLDER, InvalidParamHolder ),
    /*      8 */  ENUM_DEFINE( TA_INVALID_PARAM_HOLDER_TYPE, InvalidParamHolderType ),
    /*      9 */  ENUM_DEFINE( TA_INVALID_PARAM_FUNCTION, InvalidParamFunction ),
    /*     10 */  ENUM_DEFINE( TA_INPUT_NOT_ALL_INITIALIZE, InputNotAllInitialize ),
    /*     11 */  ENUM_DEFINE( TA_OUTPUT_NOT_ALL_INITIALIZE, OutputNotAllInitialize ),
    /*     12 */  ENUM_DEFINE( TA_OUT_OF_RANGE_START_INDEX, OutOfRangeStartIndex ),
    /*     13 */  ENUM_DEFINE( TA_OUT_OF_RANGE_END_INDEX, OutOfRangeEndIndex ),
    /*     14 */  ENUM_DEFINE( TA_INVALID_LIST_TYPE, InvalidListType ),
    /*     15 */  ENUM_DEFINE( TA_BAD_OBJECT, BadObject ),
    /*     16 */  ENUM_DEFINE( TA_NOT_SUPPORTED, NotSupported ),
    /*   5000 */  ENUM_DEFINE( TA_INTERNAL_ERROR, InternalError ) = 5000,
    /* 0xFFFF */  ENUM_DEFINE( TA_UNKNOWN_ERR, UnknownErr ) = 0xFFFF
ENUM_END( RetCode )

ENUM_BEGIN( Compatibility )    
    ENUM_DEFINE( TA_COMPATIBILITY_DEFAULT, Default ),
    ENUM_DEFINE( TA_COMPATIBILITY_METASTOCK, Metastock )
ENUM_END( Compatibility )

ENUM_BEGIN( MAType )
   ENUM_DEFINE( TA_MAType_SMA,   Sma   ) =0,
   ENUM_DEFINE( TA_MAType_EMA,   Ema   ) =1,
   ENUM_DEFINE( TA_MAType_WMA,   Wma   ) =2,
   ENUM_DEFINE( TA_MAType_DEMA,  Dema  ) =3,
   ENUM_DEFINE( TA_MAType_TEMA,  Tema  ) =4,
   ENUM_DEFINE( TA_MAType_TRIMA, Trima ) =5,
   ENUM_DEFINE( TA_MAType_KAMA,  Kama  ) =6,
   ENUM_DEFINE( TA_MAType_MAMA,  Mama  ) =7,
   ENUM_DEFINE( TA_MAType_T3,    T3    ) =8
ENUM_END( MAType )


/**** START GENCODE SECTION 1 - DO NOT DELETE THIS LINE ****/
/* Generated */ 
/* Generated */ ENUM_BEGIN( FuncUnstId )
/* Generated */     /* 000 */  ENUM_DEFINE( TA_FUNC_UNST_ADX, Adx),
/* Generated */     /* 001 */  ENUM_DEFINE( TA_FUNC_UNST_ADXR, Adxr),
/* Generated */     /* 002 */  ENUM_DEFINE( TA_FUNC_UNST_ATR, Atr),
/* Generated */     /* 003 */  ENUM_DEFINE( TA_FUNC_UNST_CMO, Cmo),
/* Generated */     /* 004 */  ENUM_DEFINE( TA_FUNC_UNST_DX, Dx),
/* Generated */     /* 005 */  ENUM_DEFINE( TA_FUNC_UNST_EMA, Ema),
/* Generated */     /* 006 */  ENUM_DEFINE( TA_FUNC_UNST_HT_DCPERIOD, HtDcPeriod),
/* Generated */     /* 007 */  ENUM_DEFINE( TA_FUNC_UNST_HT_DCPHASE, HtDcPhase),
/* Generated */     /* 008 */  ENUM_DEFINE( TA_FUNC_UNST_HT_PHASOR, HtPhasor),
/* Generated */     /* 009 */  ENUM_DEFINE( TA_FUNC_UNST_HT_SINE, HtSine),
/* Generated */     /* 010 */  ENUM_DEFINE( TA_FUNC_UNST_HT_TRENDLINE, HtTrendline),
/* Generated */     /* 011 */  ENUM_DEFINE( TA_FUNC_UNST_HT_TRENDMODE, HtTrendMode),
/* Generated */     /* 012 */  ENUM_DEFINE( TA_FUNC_UNST_KAMA, Kama),
/* Generated */     /* 013 */  ENUM_DEFINE( TA_FUNC_UNST_MAMA, Mama),
/* Generated */     /* 014 */  ENUM_DEFINE( TA_FUNC_UNST_MFI, Mfi),
/* Generated */     /* 015 */  ENUM_DEFINE( TA_FUNC_UNST_MINUS_DI, MinusDI),
/* Generated */     /* 016 */  ENUM_DEFINE( TA_FUNC_UNST_MINUS_DM, MinusDM),
/* Generated */     /* 017 */  ENUM_DEFINE( TA_FUNC_UNST_NATR, Natr),
/* Generated */     /* 018 */  ENUM_DEFINE( TA_FUNC_UNST_PLUS_DI, PlusDI),
/* Generated */     /* 019 */  ENUM_DEFINE( TA_FUNC_UNST_PLUS_DM, PlusDM),
/* Generated */     /* 020 */  ENUM_DEFINE( TA_FUNC_UNST_RSI, Rsi),
/* Generated */     /* 021 */  ENUM_DEFINE( TA_FUNC_UNST_STOCHRSI, StochRsi),
/* Generated */     /* 022 */  ENUM_DEFINE( TA_FUNC_UNST_T3, T3),
/* Generated */                ENUM_DEFINE( TA_FUNC_UNST_ALL, FuncUnstAll),
/* Generated */                ENUM_DEFINE( TA_FUNC_UNST_NONE, FuncUnstNone) = -1
/* Generated */ ENUM_END( FuncUnstId )
/* Generated */ 
/**** END GENCODE SECTION 1 - DO NOT DELETE THIS LINE ****/

/* The TA_RangeType enum specifies the types of range that can be considered 
 * when to compare a part of a candle to other candles
 */

ENUM_BEGIN( RangeType )
   ENUM_DEFINE( TA_RangeType_RealBody, RealBody ),
   ENUM_DEFINE( TA_RangeType_HighLow, HighLow ),
   ENUM_DEFINE( TA_RangeType_Shadows, Shadows )
ENUM_END( RangeType )

/* The TA_CandleSettingType enum specifies which kind of setting to consider;
 * the settings are based on the parts of the candle and the common words
 * indicating the length (short, long, very long)
 */
ENUM_BEGIN( CandleSettingType )
    ENUM_DEFINE( TA_BodyLong, BodyLong ),
    ENUM_DEFINE( TA_BodyVeryLong, BodyVeryLong ),
    ENUM_DEFINE( TA_BodyShort, BodyShort ),
    ENUM_DEFINE( TA_BodyDoji, BodyDoji ),
    ENUM_DEFINE( TA_ShadowLong, ShadowLong ),
    ENUM_DEFINE( TA_ShadowVeryLong, ShadowVeryLong ),
    ENUM_DEFINE( TA_ShadowShort, ShadowShort ),
    ENUM_DEFINE( TA_ShadowVeryShort, ShadowVeryShort ),
    ENUM_DEFINE( TA_Near, Near ),
    ENUM_DEFINE( TA_Far, Far ),
    ENUM_DEFINE( TA_Equal, Equal ),
    ENUM_DEFINE( TA_AllCandleSettings, AllCandleSettings )
ENUM_END( CandleSettingType )

#endif
