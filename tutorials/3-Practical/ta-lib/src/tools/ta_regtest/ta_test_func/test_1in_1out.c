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
 *  020203 MF   First version.
 *
 */

/* Description:
 *
 *     Test functions which have the following
 *     characterisic: 
 *      - have one input and one output
 *      - there is no optional parameters
 */

/**** Headers ****/
#include <stdio.h>
#include <string.h>

#include "ta_test_priv.h"
#include "ta_test_func.h"
#include "ta_utility.h"
#include "ta_memory.h"

/**** External functions declarations. ****/
/* None */

/**** External variables declarations. ****/
/* None */

/**** Global variables definitions.    ****/
/* None */

/**** Local declarations.              ****/
typedef enum {
  TA_HT_DCPERIOD_TEST,
  TA_HT_DCPHASE_TEST,
  TA_HT_TRENDLINE_TEST,
  TA_HT_TRENDMODE_TEST,
  TA_SIN_TEST
} TA_TestId;

typedef struct
{
   TA_Integer  doRangeTestFlag;

   TA_TestId   theFunction;
   TA_Integer  unstablePeriod;

   TA_Integer startIdx;
   TA_Integer endIdx;
   
   TA_RetCode expectedRetCode;

   TA_Integer oneOfTheExpectedOutRealIndex0;
   TA_Real    oneOfTheExpectedOutReal0;

   TA_Integer expectedBegIdx;
   TA_Integer expectedNbElement;
} TA_Test;

typedef struct
{
   const TA_Test *test;
   const TA_Real *price;
} TA_RangeTestParam;

/**** Local functions declarations.    ****/
static ErrorNumber do_test( const TA_History *history,
                            const TA_Test *test );

/**** Local variables definitions.     ****/

static TA_Test tableTest[] =
{
   /********************************/
   /* Some Hilbert Transform Tests */
   /********************************/
   { 1, TA_HT_TRENDMODE_TEST, 0, 0, 251, TA_SUCCESS,      0,  1.0, 63,  252-63 }, /* First Value */

   { 1, TA_HT_TRENDLINE_TEST, 0, 0, 251, TA_SUCCESS,      0,  88.257, 63,  252-63 }, /* First Value */
   { 0, TA_HT_TRENDLINE_TEST, 0, 0, 251, TA_SUCCESS,      0,  88.257, 63,  252-63 },
   { 0, TA_HT_TRENDLINE_TEST, 0, 0, 251, TA_SUCCESS, 252-66, 109.69, 63,  252-63 },
   { 0, TA_HT_TRENDLINE_TEST, 0, 0, 251, TA_SUCCESS, 252-65, 110.18, 63,  252-63 },
   { 0, TA_HT_TRENDLINE_TEST, 0, 0, 251, TA_SUCCESS, 252-64, 110.46, 63,  252-63 }, /* Last Value */

   { 1, TA_HT_DCPHASE_TEST, 0, 0, 251, TA_SUCCESS,      0, 22.1495, 63,  252-63 }, /* First Value */
   { 0, TA_HT_DCPHASE_TEST, 0, 0, 251, TA_SUCCESS, 252-66, -31.182, 63,  252-63 },
   { 0, TA_HT_DCPHASE_TEST, 0, 0, 251, TA_SUCCESS, 252-65, 23.2691, 63,  252-63 },
   { 0, TA_HT_DCPHASE_TEST, 0, 0, 251, TA_SUCCESS, 252-64, 47.2765, 63,  252-63 }, /* Last Value */

   { 1, TA_HT_DCPERIOD_TEST, 0, 0, 251, TA_SUCCESS,      0, 15.5527, 32,  252-32 }, /* First Value */
   { 0, TA_HT_DCPERIOD_TEST, 0, 0, 251, TA_SUCCESS, 252-33, 18.6140, 32,  252-32 },  /* Last Value */

   /*********************************/
   /* Trigonometric and Vector Math */
   /*********************************/
   { 1, TA_SIN_TEST, 0, 0, 251, TA_SUCCESS, 0, -0.38371, 0,  252 }, /* First Value */
   { 0, TA_SIN_TEST, 0, 0, 251, TA_SUCCESS, 251, 0.870319, 0,  252 }  /* Last Value */

};

#define NB_TEST (sizeof(tableTest)/sizeof(TA_Test))

/**** Global functions definitions.   ****/
ErrorNumber test_func_1in_1out( TA_History *history )
{
   unsigned int i;
   ErrorNumber retValue;

   for( i=0; i < NB_TEST; i++ )
   {
      /* Re-initialize all the unstable period to zero. */
      TA_SetUnstablePeriod( TA_FUNC_UNST_ALL, 0 );

      if( (int)tableTest[i].expectedNbElement > (int)history->nbBars )
      {
         printf( "Failed Bad Parameter for Test #%d (%d,%d)\n",
                 i, tableTest[i].expectedNbElement, history->nbBars );
         return TA_TESTUTIL_TFRR_BAD_PARAM;
      }

      retValue = do_test( history, &tableTest[i] );
      if( retValue != 0 )
      {
         printf( "Failed Test #%d (Code=%d)\n", i, retValue );
         return retValue;
      }
   }

   /* Re-initialize all the unstable period to zero. */
   TA_SetUnstablePeriod( TA_FUNC_UNST_ALL, 0 );

   /* All test succeed. */
   return TA_TEST_PASS; 
}

/**** Local functions definitions.     ****/
static TA_RetCode rangeTestFunction( TA_Integer    startIdx,
                                     TA_Integer    endIdx,
                                     TA_Real      *outputBuffer,
                                     TA_Integer   *outputBufferInt,
                                     TA_Integer   *outBegIdx,
                                     TA_Integer   *outNbElement,
                                     TA_Integer   *lookback,
                                     void         *opaqueData,
                                     unsigned int  outputNb,
                                     unsigned int *isOutputInteger )
{
   TA_RetCode retCode;
   TA_RangeTestParam *testParam;
   TA_Integer *intBuffer;
   int size, i;

   (void)outputNb;
   (void)outputBufferInt;

   *isOutputInteger = 0;
  
   testParam = (TA_RangeTestParam *)opaqueData;   

   switch( testParam->test->theFunction )
   {
   case TA_HT_DCPERIOD_TEST:
      retCode = TA_HT_DCPERIOD( startIdx,
                                endIdx,
                                testParam->price,
                                outBegIdx,
                                outNbElement,                          
                                outputBuffer );
      *lookback = TA_HT_DCPERIOD_Lookback();
      break;
   case TA_HT_DCPHASE_TEST:
      retCode = TA_HT_DCPHASE( startIdx,
                               endIdx,
                               testParam->price,
                               outBegIdx,
                               outNbElement,
                               outputBuffer );

      *lookback = TA_HT_DCPHASE_Lookback();
      break;
   case TA_HT_TRENDLINE_TEST:
      retCode = TA_HT_TRENDLINE( startIdx,
                                 endIdx,
                                 testParam->price,
                                 outBegIdx,
                                 outNbElement,
                                 outputBuffer );
      *lookback = TA_HT_TRENDLINE_Lookback();
      break;
   case TA_HT_TRENDMODE_TEST:
      /* Trendmode returns integers, but this test
       * is comparing real, so a translation is done
       * here.
       */
      #define PRE_SENTINEL  ((TA_Integer)0xABABFEDC)
      #define POST_SENTINEL ((TA_Integer)0xEFABCDFF)
      #define ALLOC_INT_BUFFER(varSize)  \
      { \
         intBuffer = TA_Malloc(sizeof(TA_Integer)*(varSize+2)); \
         intBuffer[0]      = PRE_SENTINEL; \
         intBuffer[varSize+1] = POST_SENTINEL; \
      }

      size = endIdx-startIdx+1; \
      ALLOC_INT_BUFFER(size);
      retCode = TA_HT_TRENDMODE( startIdx,
                                 endIdx,
                                 testParam->price,
                                 outBegIdx,
                                 outNbElement,
                                 &intBuffer[1] );
      *lookback = TA_HT_TRENDMODE_Lookback();

      #define FREE_INT_BUFFER( destBuffer, varNbElement ) \
      { \
         if( intBuffer[0] != PRE_SENTINEL ) \
         { \
            retCode = TA_INTERNAL_ERROR(138); \
         } \
         else if( intBuffer[size+1] != POST_SENTINEL ) \
         { \
            retCode = TA_INTERNAL_ERROR(139); \
         } \
         else \
         { \
            for( i=0; i < varNbElement; i++ ) \
               destBuffer[i] = (double)intBuffer[i+1]; \
         } \
         TA_Free( intBuffer ); \
      }

      FREE_INT_BUFFER( outputBuffer, *outNbElement );
      break;
   case TA_SIN_TEST:
      retCode = TA_SIN( startIdx,
                        endIdx,
                        testParam->price,
                        outBegIdx,
                        outNbElement,                          
                        outputBuffer );
      *lookback = TA_SIN_Lookback();
      break;
   default:
      retCode = TA_INTERNAL_ERROR(132);
   }

   return retCode;
}

static ErrorNumber do_test( const TA_History *history,
                            const TA_Test *test )
{
   TA_RetCode retCode;
   ErrorNumber errNb;
   TA_Integer outBegIdx;
   TA_Integer outNbElement;
   TA_RangeTestParam testParam;
   const TA_Real *referenceInput;

   TA_Integer *intBuffer;
   int size, i;

   /* Set to NAN all the elements of the gBuffers.  */
   clearAllBuffers();

   /* Build the input. */
   setInputBuffer( 0, history->close,  history->nbBars );

   /* Change the input to MEDPRICE for some tests. */
   switch( test->theFunction )
   {
   case TA_HT_DCPERIOD_TEST:
   case TA_HT_DCPHASE_TEST:
   case TA_HT_TRENDLINE_TEST:
   case TA_HT_TRENDMODE_TEST:
      TA_MEDPRICE( 0, history->nbBars-1, history->high, history->low,
                   &outBegIdx, &outNbElement, gBuffer[0].in );

      /* Will be use as reference */
      TA_MEDPRICE( 0, history->nbBars-1, history->high, history->low,
                   &outBegIdx, &outNbElement, gBuffer[1].in );

      referenceInput = gBuffer[1].in;
      break;
   default:
      referenceInput = history->close;
   }

   /* Make a simple first call. */
   size = (test->endIdx-test->startIdx)+1;

   switch( test->theFunction )
   {
   case TA_HT_DCPERIOD_TEST:
      retCode = TA_HT_DCPERIOD( test->startIdx,
                                test->endIdx,
                                gBuffer[0].in,
                                &outBegIdx,
                                &outNbElement,
                                gBuffer[0].out0 );
      break;

   case TA_HT_DCPHASE_TEST:
      retCode = TA_HT_DCPHASE( test->startIdx,
                               test->endIdx,
                               gBuffer[0].in,
                               &outBegIdx,
                               &outNbElement,
                               gBuffer[0].out0 );
      break;
   case TA_HT_TRENDLINE_TEST:
      retCode = TA_HT_TRENDLINE( test->startIdx,
                                 test->endIdx,
                                 gBuffer[0].in,
                                 &outBegIdx,
                                 &outNbElement,
                                 gBuffer[0].out0 );
      break;
   case TA_HT_TRENDMODE_TEST:
      ALLOC_INT_BUFFER(size);
      retCode = TA_HT_TRENDMODE( test->startIdx,
                                 test->endIdx,
                                 gBuffer[0].in,
                                 &outBegIdx,
                                 &outNbElement,
                                 &intBuffer[1] );
      FREE_INT_BUFFER( gBuffer[0].out0, outNbElement );
      break;
   case TA_SIN_TEST:
      retCode = TA_SIN( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[0].out0 );
	   break;
   default:
      retCode = TA_INTERNAL_ERROR(133);
   }

   /* Check that the input were preserved. */
   errNb = checkDataSame( gBuffer[0].in, referenceInput, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[0].out0, 0 );

   outBegIdx = outNbElement = 0;

   /* Make another call where the input and the output 
    * are the same buffer.
    */
   switch( test->theFunction )
   {
   case TA_HT_DCPERIOD_TEST:
      retCode = TA_HT_DCPERIOD( test->startIdx,
                                test->endIdx,
                                gBuffer[0].in,
                                &outBegIdx,
                                &outNbElement,
                                gBuffer[0].in
                              );
      break;

   case TA_HT_DCPHASE_TEST:
      retCode = TA_HT_DCPHASE( test->startIdx,
                               test->endIdx,
                               gBuffer[0].in,
                               &outBegIdx,
                               &outNbElement,
                               gBuffer[0].in
                              );
      break;
   case TA_HT_TRENDLINE_TEST:
      retCode = TA_HT_TRENDLINE( test->startIdx,
                                 test->endIdx,
                                 gBuffer[0].in,
                                 &outBegIdx,
                                 &outNbElement,
                                 gBuffer[0].in
                                );
      break;
   case TA_HT_TRENDMODE_TEST:
      ALLOC_INT_BUFFER(size);
      retCode = TA_HT_TRENDMODE( test->startIdx,
                                 test->endIdx,
                                 gBuffer[0].in,
                                 &outBegIdx,
                                 &outNbElement,
                                 &intBuffer[1]
                                );
      FREE_INT_BUFFER( gBuffer[0].in, outNbElement );
      break;
   case TA_SIN_TEST:
      retCode = TA_SIN( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[0].in
                        );
      break;
   default:
      retCode = TA_INTERNAL_ERROR(134);
   }

   /* The previous call should have the same output
    * as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent( gBuffer[0].out0, gBuffer[0].in );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[0].in, 0 );

   /* Do a systematic test of most of the
    * possible startIdx/endIdx range.
    */
   testParam.test  = test;
   testParam.price = referenceInput;

   if( test->doRangeTestFlag )
   {
      switch( test->theFunction )
      {
      case TA_HT_DCPERIOD_TEST:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_HT_DCPERIOD,
                              (void *)&testParam, 1, 0 );
         break;

      case TA_HT_DCPHASE_TEST:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_HT_DCPHASE,
                              (void *)&testParam, 1, 360 );
         break;

      case TA_HT_TRENDLINE_TEST:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_HT_TRENDLINE,
                              (void *)&testParam, 1, 0 );
         break;

      case TA_HT_TRENDMODE_TEST:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_HT_TRENDMODE,
                              (void *)&testParam, 1, 0 );
         break;

      default:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_NONE,
                              (void *)&testParam, 1, 0 );
      }
      if( errNb != TA_TEST_PASS )
         return errNb;
   }

   return TA_TEST_PASS;
}

