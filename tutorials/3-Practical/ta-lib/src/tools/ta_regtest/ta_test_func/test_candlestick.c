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

/* List of contributors:
 *
 *  Initial  Name/description
 *  -------------------------------------------------------------------
 *  MF       Mario Fortier
 *
 *
 * Change history:
 *
 *  MMDDYY BY   Description
 *  -------------------------------------------------------------------
 *  082304 MF   First version.
 *  041305 MF   Add latest list of function.
 */

/* Description:
 *     Test functions for candlestick. 
 */

/**** Headers ****/
#include <stdio.h>
#include <string.h>

#include "ta_test_priv.h"
#include "ta_test_func.h"
#include "ta_utility.h"

/**** External functions declarations. ****/
/* None */

/**** External variables declarations. ****/
/* None */

/**** Global variables definitions.    ****/
/* None */

/**** Local declarations.              ****/
#define MAX_OPTIN_PARAM    5
#define MAX_TESTED_OUTPUT  3

TA_RetCode TA_SetCandleSettings( TA_CandleSettingType settingType, 
                                 TA_RangeType rangeType, 
                                 int avgPeriod, 
                                 double factor );

typedef struct
{
   TA_RangeType bodyLong_type;
   int          bodyLong_avg;
   double       bodyLong_factor;
   TA_RangeType bodyVeryLong_type;
   int          bodyVeryLong_avg;
   double       bodyVeryLong_factor;
   TA_RangeType bodyShort_type;
   int          bodyShort_avg;
   double       bodyShort_factor;
   TA_RangeType bodyDoji_type;
   int          bodyDoji_avg;
   double       bodyDoji_factor;
   TA_RangeType shadowLong_type;
   int          shadowLong_avg;
   double       shadowLong_factor;
   TA_RangeType shadowVeryLong_type;
   int          shadowVeryLong_avg;
   double       shadowVeryLong_factor;
   TA_RangeType shadowShort_type;
   int          shadowShort_avg;
   double       shadowShort_factor;
   TA_RangeType shadowVeryShort_type;
   int          shadowVeryShort_avg;
   double       shadowVeryShort_factor;
   TA_RangeType near_type;
   int          near_avg;
   double       near_factor;
   TA_RangeType far_type;
   int          far_avg;
   double       far_factor;
} TA_CDLGlobals;

typedef struct
{
   int index;
   int value;  
} TA_ExpectedOutput;


typedef struct
{
   /* Indicate which function will be called */
   const char *name;

   /* Indicate if ranging test should be done. 
    * (These tests are very time consuming).
    */
   int doRangeTestFlag;

   /* Range for the function call. 
    * When both value are -1 a series of automated range 
    * tests are performed.
    */
   TA_Integer startIdx;
   TA_Integer endIdx;

   /* Up to 5 parameters depending of functions. 
    * Will be converted to integer when input is integer.
    */
   TA_Real params[MAX_OPTIN_PARAM];

   /* The expected return code. */   
   TA_RetCode expectedRetCode;

   /* When return code is TA_SUCCESS, the following output's
    * element are verified.
    */
   TA_ExpectedOutput output[MAX_TESTED_OUTPUT];
} TA_Test;


typedef struct
{
   /* Allows to pass key information as an 
    * opaque parameter for doRangeTest.
    */
   const TA_Test *test;
   const TA_Real *open;
   const TA_Real *high;
   const TA_Real *low;
   const TA_Real *close;

   TA_ParamHolder *paramHolder;
} TA_RangeTestParam;

/**** Local functions declarations.    ****/
static ErrorNumber do_test( const TA_History *history,
                            const TA_Test *test );

static ErrorNumber callCandlestick( TA_ParamHolder **paramHolderPtr,
                                    const char   *name,
                                    int           startIdx,
                                    int           endIdx,
                                    const double *inOpen,
                                    const double *inHigh,
                                    const double *inLow,
                                    const double *inClose,
                                    const double  optInArray[],
                                    int          *outBegIdx,
                                    int          *outNbElement,                                    
                                    int           outInteger[],
                                    int          *lookback,
                                    TA_RetCode   *retCode );
/**** Local variables definitions.     ****/

/* Some set of globals */

/* List of test to perform. */
static TA_Test tableTest[] =
{
   { "CDL2CROWS",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDL3BLACKCROWS",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDL3INSIDE",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDL3LINESTRIKE",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDL3OUTSIDE",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDL3STARSINSOUTH",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDL3WHITESOLDIERS",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLABANDONEDBABY",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLADVANCEBLOCK",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLBELTHOLD",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLBREAKAWAY",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLCLOSINGMARUBOZU",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLCONCEALBABYSWALL",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLCOUNTERATTACK",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLDARKCLOUDCOVER",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLDOJI",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLDOJISTAR",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLDRAGONFLYDOJI",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLENGULFING",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLEVENINGDOJISTAR",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLEVENINGSTAR",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLGAPSIDESIDEWHITE",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLGRAVESTONEDOJI",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLHAMMER",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLHANGINGMAN",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLHARAMI",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLHARAMICROSS",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLHIKKAKE",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLHIKKAKEMOD",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLHIGHWAVE",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLHOMINGPIGEON",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLIDENTICAL3CROWS",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLINNECK",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLINVERTEDHAMMER",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLKICKING",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLKICKINGBYLENGTH",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLLADDERBOTTOM",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLLONGLEGGEDDOJI",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLLONGLINE",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLMARUBOZU",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLMATCHINGLOW",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLMATHOLD",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLMORNINGDOJISTAR",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLMORNINGSTAR",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLONNECK",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLPIERCING",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLRICKSHAWMAN",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLRISEFALL3METHODS",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLSEPARATINGLINES",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLSHOOTINGSTAR",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLSHORTLINE",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLSPINNINGTOP",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLSTALLEDPATTERN",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLSTICKSANDWICH",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLTAKURI",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLTASUKIGAP",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLTHRUSTING",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLTRISTAR",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLUNIQUE3RIVER",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLUPSIDEGAP2CROWS",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }},
   { "CDLXSIDEGAP3METHODS",1, 0, 0, {0.0,0.0}, TA_SUCCESS, { {0,0}, {1,1} }}
};

#define NB_TEST (sizeof(tableTest)/sizeof(TA_Test))

/**** Global functions definitions.   ****/
ErrorNumber test_candlestick( TA_History *history )
{
   unsigned int i;
   ErrorNumber retValue;

   /* Initialize all the unstable period with a large number that would
    * break the logic if a candlestick unexpectably use a function affected
    * by an unstable period.
    */
   TA_SetUnstablePeriod( TA_FUNC_UNST_ALL, 20000 );

   /* Perform sequentialy all the tests. */
   for( i=0; i < NB_TEST; i++ )
   {
      retValue = do_test( history, &tableTest[i] );
      if( retValue != 0 )
      {
         printf( "Failed Test #%d for %s (retValue=%d)\n", i, tableTest[i].name, retValue );
         return retValue;
      }
   }

   /* Re-initialize all the unstable period to zero. */
   TA_SetUnstablePeriod( TA_FUNC_UNST_ALL, 0 );

   /* All tests succeed. */
   return TA_TEST_PASS; 
}

/**** Local functions definitions.     ****/

/* Abstract call for all candlestick functions.
 *
 * Call the function by 'name'.
 * 
 * Optional inputs are pass as an array of double.
 * Elements will be converted to integer as needed.
 *
 * All outputs are returned in the remaining parameters.
 *
 * 'lookback' is the return value of the corresponding Lookback function.
 * taFuncRetCode is the return code from the call of the TA function.
 *
 */
static ErrorNumber callCandlestick( TA_ParamHolder **paramHolderPtr,
                                    const char   *name,
                                    int           startIdx,
                                    int           endIdx,
                                    const double *inOpen,
                                    const double *inHigh,
                                    const double *inLow,
                                    const double *inClose,
                                    const double  optInArray[],
                                    int          *outBegIdx,
                                    int          *outNbElement,                                    
                                    int           outInteger[],
                                    int          *lookback,
                                    TA_RetCode   *taFuncRetCode )
{

   /* Use the abstract interface to call the function by name. */
   TA_ParamHolder *paramHolder;
   const TA_FuncHandle *handle;
   const TA_FuncInfo *funcInfo;
   const TA_InputParameterInfo *inputInfo;
   const TA_OutputParameterInfo *outputInfo;

   TA_RetCode retCode;

   (void)optInArray;
   
   /* Speed optimization if paramHolder is already initialized. */   
   paramHolder = *paramHolderPtr;
   if( !paramHolder )
   {
      retCode = TA_GetFuncHandle( name, &handle );
      if( retCode != TA_SUCCESS )
      {
         printf( "Can't get the function handle [%d]\n", retCode );
         return TA_TSTCDL_GETFUNCHANDLE_FAIL;   
      }
                             
      retCode = TA_ParamHolderAlloc( handle, &paramHolder );
      if( retCode != TA_SUCCESS )
      {
         printf( "Can't allocate the param holder [%d]\n", retCode );
         return TA_TSTCDL_PARAMHOLDERALLOC_FAIL;
      }

      *paramHolderPtr = paramHolder;
      TA_GetFuncInfo( handle, &funcInfo );

      /* Verify that the input are only OHLC. */
      if( funcInfo->nbInput != 1 )
      {
         printf( "Candlestick are expected to use only OHLC as input.\n" );
         return TA_TSTCDL_NBINPUT_WRONG;
      }

      TA_GetInputParameterInfo( handle, 0, &inputInfo );

      if( inputInfo->type != TA_Input_Price )
      {
         printf( "Candlestick are expected to use only OHLC as input.\n" );
         return TA_TSTCDL_INPUT_TYPE_WRONG;
      }
   
      if( inputInfo->flags != (TA_IN_PRICE_OPEN |
                               TA_IN_PRICE_HIGH |
                               TA_IN_PRICE_LOW  |
                               TA_IN_PRICE_CLOSE) )
      {
         printf( "Candlestick are expected to use only OHLC as input.\n" );
         return TA_TSTCDL_INPUT_FLAG_WRONG;
      }
    
      /* Set the optional inputs. */
   
      /* Verify that there is only one output. */
      if( funcInfo->nbOutput != 1 )
      {
         printf( "Candlestick are expected to have only one output array.\n" );
         return TA_TSTCDL_NBOUTPUT_WRONG;
      }

      TA_GetOutputParameterInfo( handle, 0, &outputInfo );
      if( outputInfo->type != TA_Output_Integer )
      {
         printf( "Candlestick are expected to have only one output array of type integer.\n" );
         return TA_TSTCDL_OUTPUT_TYPE_WRONG;
      }

      /* !!!!!!!!!!!!! TO BE DONE !!!!!!!!!!!!!!!!!! 
       * For now all candlestick functions will be called with default optional parameter.
       */
   }

   /* Set the input buffers. */
   TA_SetInputParamPricePtr( paramHolder, 0,
                             inOpen, inHigh, inLow, inClose, NULL, NULL );

   TA_SetOutputParamIntegerPtr(paramHolder,0,outInteger);


   /* Do the function call. */
   *taFuncRetCode = TA_CallFunc(paramHolder,startIdx,endIdx,outBegIdx,outNbElement);

   if( *taFuncRetCode != TA_SUCCESS )
   {
      printf( "TA_CallFunc() failed [%d]\n", *taFuncRetCode );
      TA_ParamHolderFree( paramHolder );
      return TA_TSTCDL_CALLFUNC_FAIL;
   }      

   /* Do the lookback function call. */
   retCode = TA_GetLookback( paramHolder, lookback );
   if( retCode != TA_SUCCESS )
   {
      printf( "TA_GetLookback() failed [%d]\n", retCode );
      TA_ParamHolderFree( paramHolder );
      return TA_TSTCDL_GETLOOKBACK_FAIL;
   }

   return TA_TEST_PASS;   
}

/* rangeTestFunction is a different way to call any of 
 * the TA function.
 *
 * This is called by doRangeTest found in test_util.c
 *
 * The doRangeTest verifies behavior that should be common
 * for ALL TA functions. It detects bugs like:
 *   - outBegIdx, outNbElement and lookback inconsistency.
 *   - off-by-one writes to output.
 *   - output inconsistency for different start/end index.
 *   - ... many other limit cases...
 *
 * In the case of candlestick, the output is integer and 
 * should be put in outputBufferInt, and outputBuffer is
 * ignored.
 */
static TA_RetCode rangeTestFunction( TA_Integer   startIdx,
                                     TA_Integer   endIdx,
                                     TA_Real     *outputBuffer,
                                     TA_Integer  *outputBufferInt,
                                     TA_Integer  *outBegIdx,
                                     TA_Integer  *outNbElement,
                                     TA_Integer  *lookback,
                                     void        *opaqueData,
                                     unsigned int outputNb,
                                     unsigned int *isOutputInteger )
{
   TA_RangeTestParam *testParam1;
   const TA_Test *testParam2;
   ErrorNumber errNb;

   TA_RetCode retCode;

   (void)outputBuffer;
   (void)outputNb;

   testParam1 = (TA_RangeTestParam *)opaqueData;
   testParam2 = (const TA_Test *)testParam1->test;

   *isOutputInteger = 1; /* Must be != 0 */

   retCode = TA_INTERNAL_ERROR(166);

   /* Call the TA function by name */
   errNb = callCandlestick( &testParam1->paramHolder,
                            testParam2->name,
                            startIdx, endIdx,
                            testParam1->open,
                            testParam1->high,
                            testParam1->low,
                            testParam1->close,
                            testParam2->params,
                            outBegIdx,
                            outNbElement,                                    
                            outputBufferInt,
                            lookback,
                            &retCode );
 
   if( errNb != TA_TEST_PASS )
      retCode = TA_INTERNAL_ERROR(168);

   return retCode;
}

static ErrorNumber do_test( const TA_History *history,
                            const TA_Test *test )
{
   TA_RangeTestParam testParam;
   ErrorNumber errNb;
   TA_RetCode retCode;

   (void)test;

   /* Set to NAN all the elements of the gBuffers.  */
   clearAllBuffers();

   /* Build the input. */
   setInputBuffer( 0, history->open,  history->nbBars );
   setInputBuffer( 1, history->high,  history->nbBars );
   setInputBuffer( 2, history->low,   history->nbBars );
   setInputBuffer( 3, history->close, history->nbBars );
      
   
#if 0
   /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
   /* Test for specific value not yet implemented */

   /* Make a simple first call. */
   switch( test->theFunction )
   {
   case TA_CCI_TEST:
      retCode = TA_CCI( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        gBuffer[1].in,
                        gBuffer[2].in,
                        test->optInTimePeriod,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[0].out0 );
      break;

   case TA_WILLR_TEST:
      retCode = TA_WILLR( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          gBuffer[1].in,
                          gBuffer[2].in,
                          test->optInTimePeriod,
                          &outBegIdx,
                          &outNbElement,
                          gBuffer[0].out0 );
      break;

   default:
      retCode = TA_INTERNAL_ERROR(133);
   }

   /* Check that the input were preserved. */
   errNb = checkDataSame( gBuffer[0].in, history->high,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[1].in, history->low, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[2].in, history->close,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[0].out0, 0 );

   outBegIdx = outNbElement = 0;

   /* Make another call where the input and the output are the
    * same buffer.
    */
   switch( test->theFunction )
   {
   case TA_CCI_TEST:
      retCode = TA_CCI( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        gBuffer[1].in,
                        gBuffer[2].in,
                        test->optInTimePeriod,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[0].in );
      break;
   case TA_WILLR_TEST:
      retCode = TA_WILLR( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          gBuffer[1].in,
                          gBuffer[2].in,
                          test->optInTimePeriod,
                          &outBegIdx,
                          &outNbElement,
                          gBuffer[0].in );
      break;
   default:
      retCode = TA_INTERNAL_ERROR(134);
   }

   /* Check that the input were preserved. */
   errNb = checkDataSame( gBuffer[1].in, history->low, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[2].in, history->close,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   /* The previous call to TA_MA should have the same output
    * as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent( gBuffer[0].out0, gBuffer[0].in );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[0].in, 0 );
   setInputBuffer( 0, history->high,  history->nbBars );

   /* Make another call where the input and the output are the
    * same buffer.
    */
   switch( test->theFunction )
   {
   case TA_CCI_TEST:
      retCode = TA_CCI( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        gBuffer[1].in,
                        gBuffer[2].in,
                        test->optInTimePeriod,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[1].in );
      break;
   case TA_WILLR_TEST:
      retCode = TA_WILLR( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          gBuffer[1].in,
                          gBuffer[2].in,
                          test->optInTimePeriod,
                          &outBegIdx,
                          &outNbElement,
                          gBuffer[1].in );
      break;
   default:
      retCode = TA_INTERNAL_ERROR(135);
   }

   /* Check that the input were preserved. */
   errNb = checkDataSame( gBuffer[0].in, history->high,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[2].in, history->close,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   /* The previous call should have the same output as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent( gBuffer[0].out0, gBuffer[1].in );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[1].in, 0 );
   setInputBuffer( 1, history->low,   history->nbBars );

   /* Make another call where the input and the output are the
    * same buffer.
    */
   switch( test->theFunction )
   {
   case TA_CCI_TEST:
      retCode = TA_CCI( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        gBuffer[1].in,
                        gBuffer[2].in,
                        test->optInTimePeriod,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[2].in );
      break;
   case TA_WILLR_TEST:
      retCode = TA_WILLR( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          gBuffer[1].in,
                          gBuffer[2].in,
                          test->optInTimePeriod,
                          &outBegIdx,
                          &outNbElement,
                          gBuffer[2].in );
      break;
   default:
      retCode = TA_INTERNAL_ERROR(136);
   }

   /* Check that the input were preserved. */
   errNb = checkDataSame( gBuffer[0].in, history->high,history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[1].in, history->low, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   /* The previous call to TA_MA should have the same output
    * as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent( gBuffer[0].out0, gBuffer[2].in );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[2].in, 0 );
   setInputBuffer( 2, history->close, history->nbBars );
#endif

   /* Do a systematic test of most of the
    * possible startIdx/endIdx range.
    */
   testParam.test  = test;
   testParam.open  = history->open;
   testParam.high  = history->high;
   testParam.low   = history->low;
   testParam.close  = history->close;
   testParam.paramHolder = NULL;

   if( test->doRangeTestFlag )
   {
      
      errNb = doRangeTest( rangeTestFunction, 
                           TA_FUNC_UNST_NONE,
                           (void *)&testParam, 1, 0 );

      if( testParam.paramHolder )
      {
         retCode = TA_ParamHolderFree( testParam.paramHolder );
         if( retCode != TA_SUCCESS )
         {
            printf( "TA_ParamHolderFree failed [%d]\n", retCode );
            return TA_TSTCDL_PARAMHOLDERFREE_FAIL;
         }
      }
       
      if( errNb != TA_TEST_PASS )
         return errNb;
   }

   return TA_TEST_PASS;
}
