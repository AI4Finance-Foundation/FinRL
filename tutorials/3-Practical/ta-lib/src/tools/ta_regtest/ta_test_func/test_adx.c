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
 *  112400 MF   First version.
 *
 */

/* Description:
 *     Test all the directional movement functions.
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
typedef enum
{
   TST_MINUS_DM,
   TST_MINUS_DI,
   TST_PLUS_DM,
   TST_PLUS_DI,
   TST_DX,
   TST_ADX,
   TST_ADXR
} TestId;

typedef struct
{
   TestId id;

   TA_Integer doRangeTestFlag;

   TA_Integer unstablePeriod;

   TA_Integer startIdx;
   TA_Integer endIdx;

   TA_Integer optInTimePeriod;

   TA_RetCode expectedRetCode;

   TA_Integer oneOfTheExpectedOutRealIndex0;
   TA_Real    oneOfTheExpectedOutReal0;

   TA_Integer expectedBegIdx;
   TA_Integer expectedNbElement;
} TA_Test;

typedef struct
{
   const TA_Test *test;
   const TA_Real *high;
   const TA_Real *low;
   const TA_Real *close;
} TA_RangeTestParam;

/**** Local functions declarations.    ****/
static ErrorNumber do_test( const TA_History *history,
                            const TA_Test *test );

/**** Local variables definitions.     ****/

static TA_Test tableTest[] =
{
   /* Test the fixes for Bug#1089506 */
   { TST_MINUS_DI, 1, 0, 0, 251, 1, TA_SUCCESS, 0,   0.0, 1,  252-1 },
   { TST_PLUS_DI,  1, 0, 0, 251, 1, TA_SUCCESS, 0,  0.478,  1,  252-1 },
   { TST_MINUS_DM, 1, 0, 0, 251, 1, TA_SUCCESS, 0,   0.0,      1,  252-1 },
   { TST_PLUS_DM,  1, 0, 0, 251, 1, TA_SUCCESS, 0,   1.69,     1,  252-1 },
   
   /* Normal regression tests. */
   { TST_ADXR,1, 0, 0, 251, 14, TA_SUCCESS, 0,   19.8666,   40,  252-40 }, /* First Value */
   { TST_ADXR,0, 0, 0, 251, 14, TA_SUCCESS, 1,   18.9092,   40,  252-40 },

   { TST_ADXR,0, 0, 0, 251, 14, TA_SUCCESS, 210, 21.5972,   40,  252-40 },
   { TST_ADXR,0, 0, 0, 251, 14, TA_SUCCESS, 211, 20.4920,   40,  252-40 }, /* Last Value */

   { TST_PLUS_DM, 1, 0, 0, 251, 14, TA_SUCCESS, 0,   10.28,  13,  252-13 }, /* First Value */
   { TST_PLUS_DM, 0, 0, 0, 251, 14, TA_SUCCESS, 237, 10.317, 13,  252-13 },
   { TST_PLUS_DM, 0, 0, 0, 251, 14, TA_SUCCESS, 238,  9.58,  13,  252-13 }, /* Last Value */

   { TST_PLUS_DI, 1, 0, 0, 251, 14, TA_SUCCESS, 0,   20.3781,   14,  252-14 }, /* First Value */

   { TST_PLUS_DI, 0, 0, 0, 251, 14, TA_SUCCESS, 13,  22.1073,   14,  252-14 },
   { TST_PLUS_DI, 0, 0, 0, 251, 14, TA_SUCCESS, 14,  20.3746,   14,  252-14 },
   { TST_PLUS_DI, 0, 0, 0, 251, 14, TA_SUCCESS, 237, 21.0000,   14,  252-14 }, /* Last Value */

   { TST_MINUS_DM, 1, 0, 0, 251, 14, TA_SUCCESS, 0,   12.995,  13,  252-13 }, /* First Value */
   { TST_MINUS_DM, 0, 0, 0, 251, 14, TA_SUCCESS, 237,  8.33,   13,  252-13 },
   { TST_MINUS_DM, 0, 0, 0, 251, 14, TA_SUCCESS, 238,  9.672,  13,  252-13 }, /* Last Value */

   { TST_MINUS_DI, 1, 0, 0, 251, 14, TA_SUCCESS, 0,   30.1684,   14,  252-14 }, /* First Value */
   { TST_MINUS_DI, 0, 0, 0, 251, 14, TA_SUCCESS, 14,  24.969182,   14,  252-14 },
   { TST_MINUS_DI, 0, 0, 0, 251, 14, TA_SUCCESS, 237, 21.1988,   14,  252-14 }, /* Last Value */

   { TST_DX, 1, 0, 0, 251, 14, TA_SUCCESS, 0,   19.3689,   14,  252-14 }, /* First Value */
   { TST_DX, 0, 0, 0, 251, 14, TA_SUCCESS, 1,    9.7131,   14,  252-14 }, 
   { TST_DX, 0, 0, 0, 251, 14, TA_SUCCESS, 2,   17.2905,   14,  252-14 }, 
   { TST_DX, 0, 0, 0, 251, 14, TA_SUCCESS, 236, 10.6731,   14,  252-14 },
   { TST_DX, 0, 0, 0, 251, 14, TA_SUCCESS, 237,  0.4722,   14,  252-14 }, /* Last Value */

   { TST_ADX, 1, 0, 0, 251, 14, TA_SUCCESS, 0,   23.0000,   27,  252-27 }, /* First Value */
   { TST_ADX, 0, 0, 0, 251, 14, TA_SUCCESS, 1,   22.0802,   27,  252-27 },
   { TST_ADX, 0, 0, 0, 251, 14, TA_SUCCESS, 223, 16.6840,   27,  252-27 },
   { TST_ADX, 0, 0, 0, 251, 14, TA_SUCCESS, 224, 15.5260,   27,  252-27 }  /* Last Value */

#if 0
   /*These were the values when using integer rounding in the logic */
   { TST_ADXR,1, 0, 0, 251, 14, TA_SUCCESS, 0,   19.0,   40,  252-40 }, /* First Value */
   { TST_ADXR,0, 0, 0, 251, 14, TA_SUCCESS, 1,   18.0,   40,  252-40 },
   { TST_ADXR,0, 0, 0, 251, 14, TA_SUCCESS, 210, 22.0,   40,  252-40 },
   { TST_ADXR,0, 0, 0, 251, 14, TA_SUCCESS, 211, 21.0,   40,  252-40 }, /* Last Value */

   { TST_PLUS_DI, 1, 0, 0, 251, 14, TA_SUCCESS, 0,   20.0,   14,  252-14 }, /* First Value */
   { TST_PLUS_DI, 0, 0, 0, 251, 14, TA_SUCCESS, 13,  22.0,   14,  252-14 },
   { TST_PLUS_DI, 0, 0, 0, 251, 14, TA_SUCCESS, 14,  20.0,   14,  252-14 },
   { TST_PLUS_DI, 0, 0, 0, 251, 14, TA_SUCCESS, 237, 21.0,   14,  252-14 }, /* Last Value */

   { TST_MINUS_DI, 1, 0, 0, 251, 14, TA_SUCCESS, 0,   30.0,   14,  252-14 }, /* First Value */
   { TST_MINUS_DI, 0, 0, 0, 251, 14, TA_SUCCESS, 14,  25.0,   14,  252-14 },
   { TST_MINUS_DI, 0, 0, 0, 251, 14, TA_SUCCESS, 237, 21.0,   14,  252-14 }, /* Last Value */

   { TST_DX, 1, 0, 0, 251, 14, TA_SUCCESS, 0,   20.0,   14,  252-14 }, /* First Value */
   { TST_DX, 0, 0, 0, 251, 14, TA_SUCCESS, 1,    9.0,   14,  252-14 }, 
   { TST_DX, 0, 0, 0, 251, 14, TA_SUCCESS, 2,   18.0,   14,  252-14 }, 
   { TST_DX, 0, 0, 0, 251, 14, TA_SUCCESS, 236, 10.0,   14,  252-14 }, 
   { TST_DX, 0, 0, 0, 251, 14, TA_SUCCESS, 237,  0.0,   14,  252-14 }, /* Last Value */

   { TST_ADX, 1, 0, 0, 251, 14, TA_SUCCESS, 0,   23.0,   27,  252-27 }, /* First Value */
   { TST_ADX, 0, 0, 0, 251, 14, TA_SUCCESS, 1,   22.0,   27,  252-27 },
   { TST_ADX, 0, 0, 0, 251, 14, TA_SUCCESS, 223, 17.0,   27,  252-27 },
   { TST_ADX, 0, 0, 0, 251, 14, TA_SUCCESS, 224, 16.0,   27,  252-27 }  /* Last Value */
#endif
};

#define NB_TEST (sizeof(tableTest)/sizeof(TA_Test))

/**** Global functions definitions.   ****/
ErrorNumber test_func_adx( TA_History *history )
{
   unsigned int i;
   ErrorNumber retValue;

   /* Re-initialize all the unstable period to zero. */
   TA_SetUnstablePeriod( TA_FUNC_UNST_ALL, 0 );

   for( i=0; i < NB_TEST; i++ )
   {
      if( (int)tableTest[i].expectedNbElement > (int)history->nbBars )
      {
         printf( "%s Failed Bad Parameter for Test #%d (%d,%d)\n", __FILE__,
                 i, tableTest[i].expectedNbElement, history->nbBars );
         return TA_TESTUTIL_TFRR_BAD_PARAM;
      }

      retValue = do_test( history, &tableTest[i] );
      if( retValue != 0 )
      {
         printf( "%s Failed Test #%d (Code=%d)\n", __FILE__,
                 i, retValue );
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

   (void)outputNb;
   (void)outputBufferInt;
  
   *isOutputInteger = 0;

   testParam = (TA_RangeTestParam *)opaqueData;   

   switch( testParam->test->id )
   {
   case TST_MINUS_DM:
      retCode = TA_MINUS_DM( startIdx,
                             endIdx,
                             testParam->high,
                             testParam->low,
                             testParam->test->optInTimePeriod,
                             outBegIdx,
                             outNbElement,
                             outputBuffer );

      *lookback = TA_MINUS_DM_Lookback( testParam->test->optInTimePeriod );
      break;

   case TST_MINUS_DI:
      retCode = TA_MINUS_DI( startIdx,
                             endIdx,
                             testParam->high,
                             testParam->low,
                             testParam->close,
                             testParam->test->optInTimePeriod,
                             outBegIdx,
                             outNbElement,
                             outputBuffer );

      *lookback = TA_MINUS_DI_Lookback( testParam->test->optInTimePeriod );
      break;

   case TST_DX:
      retCode = TA_DX( startIdx,
                       endIdx,
                       testParam->high,
                       testParam->low,
                       testParam->close,
                       testParam->test->optInTimePeriod,
                       outBegIdx,
                       outNbElement,
                       outputBuffer );

      *lookback = TA_DX_Lookback( testParam->test->optInTimePeriod );
      break;

   case TST_ADX:
      retCode = TA_ADX( startIdx,
                        endIdx,
                        testParam->high,
                        testParam->low,
                        testParam->close,
                        testParam->test->optInTimePeriod,
                        outBegIdx,
                        outNbElement,
                        outputBuffer );

      *lookback = TA_ADX_Lookback( testParam->test->optInTimePeriod );
      break;

   case TST_PLUS_DM:
      retCode = TA_PLUS_DM( startIdx,
                            endIdx,
                            testParam->high,
                            testParam->low,
                            testParam->test->optInTimePeriod,
                            outBegIdx,
                            outNbElement,
                            outputBuffer );

      *lookback = TA_PLUS_DM_Lookback( testParam->test->optInTimePeriod );
      break;

   case TST_PLUS_DI:
      retCode = TA_PLUS_DI( startIdx,
                            endIdx,
                            testParam->high,
                            testParam->low,
                            testParam->close,
                            testParam->test->optInTimePeriod,
                            outBegIdx,
                            outNbElement,
                            outputBuffer );

      *lookback = TA_PLUS_DI_Lookback( testParam->test->optInTimePeriod );
      break;

   case TST_ADXR:
      retCode = TA_ADXR( startIdx,
                         endIdx,
                         testParam->high,
                         testParam->low,
                         testParam->close,
                         testParam->test->optInTimePeriod,
                         outBegIdx,
                         outNbElement,
                         outputBuffer );

      *lookback = TA_ADXR_Lookback( testParam->test->optInTimePeriod );
      break;
   default:
      retCode = TA_BAD_PARAM;
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

   /* Set to NAN all the elements of the gBuffers.  */
   clearAllBuffers();

   /* Build the input. */
   setInputBuffer( 0, history->high,  history->nbBars );
   setInputBuffer( 1, history->low,   history->nbBars );
   setInputBuffer( 2, history->close, history->nbBars );
   setInputBuffer( 3, history->high,  history->nbBars );

   /* Make a simple first call. */
   switch( test->id )
   {
   case TST_MINUS_DM:
      retCode = TA_SetUnstablePeriod(
                                      TA_FUNC_UNST_MINUS_DM,
                                      test->unstablePeriod );
      if( retCode != TA_SUCCESS )
         return TA_TEST_TFRR_SETUNSTABLE_PERIOD_FAIL;

      retCode = TA_MINUS_DM( test->startIdx,
                             test->endIdx,
                             gBuffer[0].in,
                             gBuffer[1].in,
                             test->optInTimePeriod,
                             &outBegIdx,
                             &outNbElement,
                             gBuffer[0].out0 );
      break;

   case TST_MINUS_DI:
      retCode = TA_SetUnstablePeriod( TA_FUNC_UNST_MINUS_DI,
                                      test->unstablePeriod );
      if( retCode != TA_SUCCESS )
         return TA_TEST_TFRR_SETUNSTABLE_PERIOD_FAIL;

      retCode = TA_MINUS_DI( test->startIdx,
                             test->endIdx,
                             gBuffer[0].in,
                             gBuffer[1].in,
                             gBuffer[2].in,
                             test->optInTimePeriod,
                             &outBegIdx,
                             &outNbElement,
                             gBuffer[0].out0 );
      break;

   case TST_DX:
      retCode = TA_SetUnstablePeriod( TA_FUNC_UNST_DX,
                                      test->unstablePeriod );
      if( retCode != TA_SUCCESS )
         return TA_TEST_TFRR_SETUNSTABLE_PERIOD_FAIL;

      retCode = TA_DX( test->startIdx,
                       test->endIdx,
                       gBuffer[0].in,
                       gBuffer[1].in,
                       gBuffer[2].in,
                       test->optInTimePeriod,
                       &outBegIdx,
                       &outNbElement,
                       gBuffer[0].out0 );
      break;

   case TST_ADX:
      retCode = TA_SetUnstablePeriod( TA_FUNC_UNST_ADX,
                                      test->unstablePeriod );
      if( retCode != TA_SUCCESS )
         return TA_TEST_TFRR_SETUNSTABLE_PERIOD_FAIL;

      retCode = TA_ADX( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        gBuffer[1].in,
                        gBuffer[2].in,
                        test->optInTimePeriod,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[0].out0 );
      break;

   case TST_PLUS_DM:
      retCode = TA_SetUnstablePeriod( TA_FUNC_UNST_PLUS_DM,
                                      test->unstablePeriod );
      if( retCode != TA_SUCCESS )
         return TA_TEST_TFRR_SETUNSTABLE_PERIOD_FAIL;

      retCode = TA_PLUS_DM( test->startIdx,
                            test->endIdx,
                            gBuffer[0].in,
                            gBuffer[1].in,
                            test->optInTimePeriod,
                            &outBegIdx,
                            &outNbElement,
                            gBuffer[0].out0 );
      break;

   case TST_PLUS_DI:
      retCode = TA_SetUnstablePeriod( TA_FUNC_UNST_PLUS_DI,
                                      test->unstablePeriod );
      if( retCode != TA_SUCCESS )
         return TA_TEST_TFRR_SETUNSTABLE_PERIOD_FAIL;

      retCode = TA_PLUS_DI( test->startIdx,
                            test->endIdx,
                            gBuffer[0].in,
                            gBuffer[1].in,
                            gBuffer[2].in,
                            test->optInTimePeriod,
                            &outBegIdx,
                            &outNbElement,
                            gBuffer[0].out0 );
      break;

   case TST_ADXR:
      retCode = TA_SetUnstablePeriod( TA_FUNC_UNST_ADX,
                                      test->unstablePeriod );
      if( retCode != TA_SUCCESS )
         return TA_TEST_TFRR_SETUNSTABLE_PERIOD_FAIL;

      retCode = TA_ADXR( test->startIdx,
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
      retCode = TA_BAD_PARAM;
   }

   /* Verify that the inputs were preserved. */
   errNb = checkDataSame( gBuffer[0].in, history->high, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[1].in, history->low, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[2].in, history->close, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[0].out0, 0 );

   outBegIdx = outNbElement = 0;

   /* Make another call where the input and the output are the
    * same buffer.
    */
   switch( test->id )
   {
   case TST_MINUS_DM:
      retCode = TA_MINUS_DM( test->startIdx,
                             test->endIdx,
                             gBuffer[3].in,
                             gBuffer[1].in,
                             test->optInTimePeriod,
                             &outBegIdx,
                             &outNbElement,
                             gBuffer[3].in );
      break;

   case TST_MINUS_DI:
      retCode = TA_MINUS_DI( test->startIdx,
                             test->endIdx,
                             gBuffer[3].in,
                             gBuffer[1].in,
                             gBuffer[2].in,
                             test->optInTimePeriod,
                             &outBegIdx,
                             &outNbElement,
                             gBuffer[3].in );
      break;

   case TST_DX:
      retCode = TA_DX( test->startIdx,
                       test->endIdx,
                       gBuffer[3].in,
                       gBuffer[1].in,
                       gBuffer[2].in,
                       test->optInTimePeriod,
                       &outBegIdx,
                       &outNbElement,
                       gBuffer[3].in );
      break;

   case TST_ADX:
      retCode = TA_ADX( test->startIdx,
                        test->endIdx,
                        gBuffer[3].in,
                        gBuffer[1].in,
                        gBuffer[2].in,
                        test->optInTimePeriod,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[3].in );
      break;

   case TST_PLUS_DM:
      retCode = TA_PLUS_DM( test->startIdx,
                            test->endIdx,
                            gBuffer[3].in,
                            gBuffer[1].in,
                            test->optInTimePeriod,
                            &outBegIdx,
                            &outNbElement,
                            gBuffer[3].in );
      break;

   case TST_PLUS_DI:
      retCode = TA_PLUS_DI( test->startIdx,
                            test->endIdx,
                            gBuffer[3].in,
                            gBuffer[1].in,
                            gBuffer[2].in,
                            test->optInTimePeriod,
                            &outBegIdx,
                            &outNbElement,
                            gBuffer[3].in );
      break;

   case TST_ADXR:
      retCode = TA_ADXR( test->startIdx,
                         test->endIdx,
                         gBuffer[3].in,
                         gBuffer[1].in,
                         gBuffer[2].in,
                         test->optInTimePeriod,
                         &outBegIdx,
                         &outNbElement,
                         gBuffer[3].in );
      break;
   default:
      retCode = TA_BAD_PARAM;
   }

   /* Verify that the inputs were preserved. */
   errNb = checkDataSame( gBuffer[1].in, history->low, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;
   errNb = checkDataSame( gBuffer[2].in, history->close, history->nbBars );
   if( errNb != TA_TEST_PASS )
      return errNb;

   /* The previous call should have the same output as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent( gBuffer[0].out0, gBuffer[3].in );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[3].in, 0 );

   if( errNb != TA_TEST_PASS )
      return errNb;

   /* Do a systematic test of most of the
    * possible startIdx/endIdx range.
    */
   testParam.test  = test;
   testParam.high  = history->high;
   testParam.low   = history->low;
   testParam.close = history->close;

   if( test->doRangeTestFlag )
   {
      switch( test->id )
      {
      case TST_MINUS_DM:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_MINUS_DM,
                              (void *)&testParam, 1, 0 );
         break;

      case TST_MINUS_DI:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_MINUS_DI,
                              (void *)&testParam, 1, 2 );
         break;

      case TST_DX:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_DX,
                              (void *)&testParam, 1, 2 );
         break;

      case TST_ADX:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_ADX,
                              (void *)&testParam, 1, 2 );
         break;

      case TST_PLUS_DM:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_PLUS_DM,
                              (void *)&testParam, 1, 0 );
         break;

      case TST_PLUS_DI:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_PLUS_DI,
                              (void *)&testParam, 1, 2 );
         break;

      case TST_ADXR:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_ADX,
                              (void *)&testParam, 1, 2 );
         break;
      }

      if( errNb != TA_TEST_PASS )
         return errNb;
   }

   return TA_TEST_PASS;
}
