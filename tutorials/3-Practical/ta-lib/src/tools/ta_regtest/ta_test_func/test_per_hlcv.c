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
 *  AC       Angelo Ciceri
 *
 *
 * Change history:
 *
 *  MMDDYY BY   Description
 *  -------------------------------------------------------------------
 *  011103 MF   First version.
 *  111705 MF   Add test for Fix#1359452 (AD Function).
 *  110206 AC   Change volume and open interest to double
 */

/* Description:
 *
 *     Test functions which have the following
 *     characterisic: 
 *      - the inputs are high,low, close and volume.
 *      - have one output of type real.
 *      - might have an optional parameter.
 *        
 *     
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
typedef enum {
TA_MFI_TEST,
TA_AD_TEST,
TA_ADOSC_3_10_TEST,
TA_ADOSC_5_2_TEST,
} TA_TestId;

typedef struct
{
   TA_Integer doRangeTestFlag;

   TA_TestId  theFunction;

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
   const TA_Test    *test;
   const TA_Real    *high;
   const TA_Real    *low;
   const TA_Real    *close;
   const TA_Real    *volume;
} TA_RangeTestParam;

/**** Local functions declarations.    ****/
static ErrorNumber do_test( const TA_History *history,
                            const TA_Test *test );

/**** Local variables definitions.     ****/

static TA_Test tableTest[] =
{
   /*************/
   /* AD TEST   */
   /*************/
   /* Note: the period field is ignored. The period is irrelevant */
   { 1, TA_AD_TEST,  0, 251, -1, TA_SUCCESS,      0, -1631000.00,  0,  252 }, /* First Value */
   { 0, TA_AD_TEST,  0, 251, -1, TA_SUCCESS,      1, 2974412.02,   0,  252 },
   { 0, TA_AD_TEST,  0, 251, -1, TA_SUCCESS,    250, 8707691.07,   0,  252 },
   { 0, TA_AD_TEST,  0, 251, -1, TA_SUCCESS,    251, 8328944.54,   0,  252 }, /* Last Value */

   /****************/
   /* ADOSC TEST   */
   /****************/
   /* Note: the period field is ignored. The periods are always 3 and 10 */
   { 1, TA_ADOSC_3_10_TEST, 0, 251, -1, TA_SUCCESS,      0,  841238.32,  9,  243 }, /* First Value */
   { 0, TA_ADOSC_3_10_TEST, 0, 251, -1, TA_SUCCESS,      1,  2255663.07, 9,  243 },
   { 0, TA_ADOSC_3_10_TEST, 0, 251, -1, TA_SUCCESS,    241,  -526700.32, 9,  243 },
   { 0, TA_ADOSC_3_10_TEST, 0, 251, -1, TA_SUCCESS,    242, -1139932.729, 9,  243 }, /* Last Value */

   /* Note: the period field is ignored. The periods are always 2 and 5 */
   { 1, TA_ADOSC_5_2_TEST, 0, 251, -1, TA_SUCCESS,      0, 585361.28,  4,  248 }, /* First Value */

   /**************/
   /* MFI TEST   */
   /**************/
   { 0, TA_MFI_TEST,  0, 251, 14, TA_SUCCESS,      0,    42.8923,  14,  252-14 }, /* First Value */
   { 0, TA_MFI_TEST,  0, 251, 14, TA_SUCCESS,      1,    45.6072,  14,  252-14 },
   { 0, TA_MFI_TEST,  0, 251, 14, TA_SUCCESS, 252-15,    53.1997,  14,  252-14 }, /* Last Value */

   { 1, TA_MFI_TEST,  0, 251, 49, TA_SUCCESS,      0,    44.7902,  49,  252-49 }, /* First Value */
   { 0, TA_MFI_TEST,  0, 251, 49, TA_SUCCESS,      1,    43.1963,  49,  252-49 },
   { 0, TA_MFI_TEST,  0, 251, 49, TA_SUCCESS, 252-50,    57.4806,  49,  252-49 }, /* Last Value */

   { 1, TA_MFI_TEST,  0, 251, 50, TA_SUCCESS,      0,    44.2414,  50,  252-50 }, /* First Value */
   { 0, TA_MFI_TEST,  0, 251, 50, TA_SUCCESS,      1,    42.1108,  50,  252-50 },
   { 0, TA_MFI_TEST,  0, 251, 50, TA_SUCCESS, 252-51,    50.5905,  50,  252-50 }, /* Last Value */

   { 1, TA_MFI_TEST,  0, 251, 51, TA_SUCCESS,      0,    43.1496,  51,  252-51 }, /* First Value */
   { 0, TA_MFI_TEST,  0, 251, 51, TA_SUCCESS,      1,    40.7692,  51,  252-51 },
   { 0, TA_MFI_TEST,  0, 251, 51, TA_SUCCESS, 252-52,    51.7265,  51,  252-51 }, /* Last Value */

   { 1, TA_MFI_TEST,  0, 251, 100, TA_SUCCESS,       0,  50.0166,  100,  252-100 }, /* First Value */
   { 0, TA_MFI_TEST,  0, 251, 100, TA_SUCCESS,       1,  50.2648,  100,  252-100 },
   { 0, TA_MFI_TEST,  0, 251, 100, TA_SUCCESS, 252-101,  48.4264,  100,  252-100 } /* Last Value */

};

#define NB_TEST (sizeof(tableTest)/sizeof(TA_Test))

/**** Global functions definitions.   ****/
ErrorNumber test_func_per_hlcv( TA_History *history )
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

   (void)outputNb;
   (void)outputBufferInt;
  
   *isOutputInteger = 0;

   testParam = (TA_RangeTestParam *)opaqueData;   

   switch( testParam->test->theFunction )
   {
   case TA_MFI_TEST:
      retCode = TA_MFI( startIdx,
                        endIdx,
                        testParam->high,
                        testParam->low,
                        testParam->close,
                        testParam->volume,
                        testParam->test->optInTimePeriod,
                        outBegIdx,
                        outNbElement,
                        outputBuffer );
      *lookback = TA_MFI_Lookback( testParam->test->optInTimePeriod );
      break;

   case TA_AD_TEST:
      retCode = TA_AD( startIdx,
                       endIdx,
                       testParam->high,
                       testParam->low,
                       testParam->close,
                       testParam->volume,
                       outBegIdx,
                       outNbElement,
                       outputBuffer );
      *lookback = TA_AD_Lookback();
      break;

   case TA_ADOSC_3_10_TEST:
      retCode = TA_ADOSC( startIdx,
                       endIdx,
                       testParam->high,
                       testParam->low,
                       testParam->close,
                       testParam->volume,
                       3, 10,
                       outBegIdx,
                       outNbElement,
                       outputBuffer );
      *lookback = TA_ADOSC_Lookback(3,10);
      break;

   case TA_ADOSC_5_2_TEST:
      retCode = TA_ADOSC( startIdx,
                       endIdx,
                       testParam->high,
                       testParam->low,
                       testParam->close,
                       testParam->volume,
                       5, 2,
                       outBegIdx,
                       outNbElement,
                       outputBuffer );
      *lookback = TA_ADOSC_Lookback(5,2);
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

   /* Set to NAN all the elements of the gBuffers.  */
   clearAllBuffers();

   /* Build the input. */
   setInputBuffer( 0, history->high,  history->nbBars );
   setInputBuffer( 1, history->low,   history->nbBars );
   setInputBuffer( 2, history->close, history->nbBars );

   /* Clear the unstable periods from previous tests. */
   retCode = TA_SetUnstablePeriod( TA_FUNC_UNST_MFI, 0 );
   if( retCode != TA_SUCCESS )
      return TA_TEST_TFRR_SETUNSTABLE_PERIOD_FAIL;      

   /* Make a simple first call. */
   switch( test->theFunction )
   {
   case TA_MFI_TEST:
      retCode = TA_MFI( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        gBuffer[1].in,
                        gBuffer[2].in,
                        history->volume,
                        test->optInTimePeriod,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[0].out0 );
      break;
   case TA_AD_TEST:
      retCode = TA_AD( test->startIdx,
                       test->endIdx,
                       gBuffer[0].in,
                       gBuffer[1].in,
                       gBuffer[2].in,
                       history->volume,
                       &outBegIdx,
                       &outNbElement,
                       gBuffer[0].out0 );
      break;

   case TA_ADOSC_3_10_TEST:
      retCode = TA_ADOSC( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          gBuffer[1].in,
                          gBuffer[2].in,
                          history->volume,
                          3, 10,
                          &outBegIdx,
                          &outNbElement,
                          gBuffer[0].out0 );
      break;

   case TA_ADOSC_5_2_TEST:
      retCode = TA_ADOSC( test->startIdx,
                          test->endIdx,
                          gBuffer[0].in,
                          gBuffer[1].in,
                          gBuffer[2].in,
                          history->volume,
                          5, 2,
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
   case TA_MFI_TEST:
      retCode = TA_MFI( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        gBuffer[1].in,
                        gBuffer[2].in,
                        history->volume,
                        test->optInTimePeriod,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[0].in );
      break;
   case TA_AD_TEST:
      retCode = TA_AD( test->startIdx,
                       test->endIdx,
                       gBuffer[0].in,
                       gBuffer[1].in,
                       gBuffer[2].in,
                       history->volume,
                       &outBegIdx,
                       &outNbElement,
                       gBuffer[0].in );
      break;
   case TA_ADOSC_3_10_TEST:
      retCode = TA_ADOSC( test->startIdx,
                       test->endIdx,
                       gBuffer[0].in,
                       gBuffer[1].in,
                       gBuffer[2].in,
                       history->volume,
                       3, 10,
                       &outBegIdx,
                       &outNbElement,                       
                       gBuffer[0].in );
      break;
   case TA_ADOSC_5_2_TEST:
      retCode = TA_ADOSC( test->startIdx,
                       test->endIdx,
                       gBuffer[0].in,
                       gBuffer[1].in,
                       gBuffer[2].in,
                       history->volume,
                       5, 2,
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

   /* The previous call should have the same output as this call.
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
   case TA_MFI_TEST:
      retCode = TA_MFI( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        gBuffer[1].in,
                        gBuffer[2].in,
                        history->volume,
                        test->optInTimePeriod,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[1].in );
      break;
   case TA_AD_TEST:
      retCode = TA_AD( test->startIdx,
                       test->endIdx,
                       gBuffer[0].in,
                       gBuffer[1].in,
                       gBuffer[2].in,
                       history->volume,
                       &outBegIdx,
                       &outNbElement,
                       gBuffer[1].in );
      break;
   case TA_ADOSC_3_10_TEST:
      retCode = TA_ADOSC( test->startIdx,
                       test->endIdx,
                       gBuffer[0].in,
                       gBuffer[1].in,
                       gBuffer[2].in,
                       history->volume,
                       3, 10,
                       &outBegIdx,
                       &outNbElement,
                       gBuffer[1].in );
      break;
   case TA_ADOSC_5_2_TEST:
      retCode = TA_ADOSC( test->startIdx,
                       test->endIdx,
                       gBuffer[0].in,
                       gBuffer[1].in,
                       gBuffer[2].in,
                       history->volume,
                       5, 2,
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

   /* The previous call to TA_MA should have the same output
    * as this call.
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
   case TA_MFI_TEST:
      retCode = TA_MFI( test->startIdx,
                        test->endIdx,
                        gBuffer[0].in,
                        gBuffer[1].in,
                        gBuffer[2].in,
                        history->volume,
                        test->optInTimePeriod,
                        &outBegIdx,
                        &outNbElement,
                        gBuffer[2].in );
      break;
   case TA_AD_TEST:
      retCode = TA_AD( test->startIdx,
                       test->endIdx,
                       gBuffer[0].in,
                       gBuffer[1].in,
                       gBuffer[2].in,
                       history->volume,
                       &outBegIdx,
                       &outNbElement,
                       gBuffer[2].in );
      break;
   case TA_ADOSC_3_10_TEST:
      retCode = TA_ADOSC( test->startIdx,
                       test->endIdx,
                       gBuffer[0].in,
                       gBuffer[1].in,
                       gBuffer[2].in,
                       history->volume,
                       3, 10,
                       &outBegIdx,
                       &outNbElement,
                       gBuffer[2].in );
      break;
   case TA_ADOSC_5_2_TEST:
      retCode = TA_ADOSC( test->startIdx,
                       test->endIdx,
                       gBuffer[0].in,
                       gBuffer[1].in,
                       gBuffer[2].in,
                       history->volume,
                       5, 2,
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

   /* The previous call should have the same output as this call.
    *
    * checkSameContent verify that all value different than NAN in
    * the first parameter is identical in the second parameter.
    */
   errNb = checkSameContent( gBuffer[0].out0, gBuffer[2].in );
   if( errNb != TA_TEST_PASS )
      return errNb;

   CHECK_EXPECTED_VALUE( gBuffer[2].in, 0 );
   setInputBuffer( 2, history->close, history->nbBars );

   if( test->doRangeTestFlag )
   {
      /* Do a systematic test of most of the
       * possible startIdx/endIdx range.
       */
      testParam.test   = test;
      testParam.high   = history->high;
      testParam.low    = history->low;
      testParam.close  = history->close;
      testParam.volume = history->volume;

      switch( test->theFunction )
      {
      case TA_MFI_TEST:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_MFI,
                              (void *)&testParam, 1, 0 );
         if( errNb != TA_TEST_PASS )
            return errNb;
         break;
      case TA_AD_TEST:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_NONE,
                              (void *)&testParam, 1,
                              TA_DO_NOT_COMPARE );
         if( errNb != TA_TEST_PASS )
            return errNb;
         break;
      case TA_ADOSC_3_10_TEST:
      case TA_ADOSC_5_2_TEST:
         errNb = doRangeTest( rangeTestFunction, 
                              TA_FUNC_UNST_EMA,
                              (void *)&testParam, 1,
                              TA_DO_NOT_COMPARE );
         if( errNb != TA_TEST_PASS )
            return errNb;
         break;
      default:
         break;
      }
   }

   /* Check for fix #1359452 - AD RAnge not working as expected. */
   if( test->theFunction == TA_AD_TEST )
   {
      gBuffer[0].out0[0] = -1.0;
      gBuffer[0].out0[1] = -1.0;
      retCode = TA_AD( 0,
                       0,
                       gBuffer[0].in,
                       gBuffer[1].in,
                       gBuffer[2].in,
                       history->volume,
                       &outBegIdx,
                       &outNbElement,
                       &gBuffer[0].out0[0] );
      if( retCode != TA_SUCCESS )
      {
         printf( "Failed AD call for fix #1359452 [%d]\n", retCode );
         return TA_TEST_FAIL_BUG1359452_1;
      }
      if( gBuffer[0].out0[0] == -1.0 )
      {
         printf( "Failed AD call for fix #1359452 out0[0] == -1\n" );
         return TA_TEST_FAIL_BUG1359452_2;
      }

      retCode = TA_AD( 1,
                       1,
                       gBuffer[0].in,
                       gBuffer[1].in,
                       gBuffer[2].in,
                       history->volume,
                       &outBegIdx,
                       &outNbElement,
                       &gBuffer[0].out0[1] );
      if( retCode != TA_SUCCESS )
      {
         printf( "Failed AD call for fix #1359452 [%d]\n", retCode );
         return TA_TEST_FAIL_BUG1359452_3;
      }
      if( gBuffer[0].out0[1] == -1.0 )
      {
         printf( "Failed AD call for fix #1359452 out0[1] == -1\n" );
         return TA_TEST_FAIL_BUG1359452_4;
      }

      /* The two values are to be different. */
      if( gBuffer[0].out0[1] == gBuffer[0].out0[0] )
      {
         printf( "Failed AD logic for fix #1359452\n" );
         return TA_TEST_FAIL_BUG1359452_5;
      }       
   }

   return TA_TEST_PASS;
}

