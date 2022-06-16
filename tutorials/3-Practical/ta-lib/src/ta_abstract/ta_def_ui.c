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
 *  130101 MF   First version.
 *  111503 MF   Add TA_DEF_UI_Input_Price_CV
 */

/* Description:
 *    Defines const structures representing how data can be
 *    input/output from the TA functions.
 *    These structure are mainly used to build the tables\table<a..z>.c files.
 */
#include <stdlib.h>
#include "ta_abstract.h"
#include "ta_def_ui.h"

const char TA_GroupId_MathOperatorsString[]        = "Math Operators";
const char TA_GroupId_MathTransformString[]        = "Math Transform";
const char TA_GroupId_OverlapStudiesString[]       = "Overlap Studies";
const char TA_GroupId_VolatilityIndicatorsString[] = "Volatility Indicators";
const char TA_GroupId_MomentumIndicatorsString[]   = "Momentum Indicators";
const char TA_GroupId_CycleIndicatorsString[]      = "Cycle Indicators";
const char TA_GroupId_VolumeIndicatorsString[]     = "Volume Indicators";
const char TA_GroupId_PatternRecognitionString[]   = "Pattern Recognition";
const char TA_GroupId_StatisticString[]            = "Statistic Functions";
const char TA_GroupId_PriceTransformString[]       = "Price Transform";

const char *TA_GroupString[TA_NB_GROUP_ID] =
{
   &TA_GroupId_MathOperatorsString[0],
   &TA_GroupId_MathTransformString[0],
   &TA_GroupId_OverlapStudiesString[0],
   &TA_GroupId_VolatilityIndicatorsString[0],
   &TA_GroupId_MomentumIndicatorsString[0],
   &TA_GroupId_CycleIndicatorsString[0],
   &TA_GroupId_VolumeIndicatorsString[0],
   &TA_GroupId_PatternRecognitionString[0],
   &TA_GroupId_StatisticString[0],
   &TA_GroupId_PriceTransformString[0]
};

/*************************************************************
 * Define from here the TA_InputParameterInfo
 * These shall be sufficient for all possible TA functions.
 *************************************************************/
const TA_InputParameterInfo TA_DEF_UI_Input_Price_OHLCV =
                                  { TA_Input_Price, "inPriceOHLCV",
                                    TA_IN_PRICE_OPEN   |
                                    TA_IN_PRICE_HIGH   |
                                    TA_IN_PRICE_LOW    |
                                    TA_IN_PRICE_CLOSE  |
                                    TA_IN_PRICE_VOLUME };

const TA_InputParameterInfo TA_DEF_UI_Input_Price_HLCV =
                                  { TA_Input_Price, "inPriceHLCV",
                                    TA_IN_PRICE_HIGH   |
                                    TA_IN_PRICE_LOW    |
                                    TA_IN_PRICE_CLOSE  |
                                    TA_IN_PRICE_VOLUME };

const TA_InputParameterInfo TA_DEF_UI_Input_Price_OHLC =
                                  { TA_Input_Price, "inPriceOHLC",
                                    TA_IN_PRICE_OPEN   |
                                    TA_IN_PRICE_HIGH   |
                                    TA_IN_PRICE_LOW    |
                                    TA_IN_PRICE_CLOSE  };

const TA_InputParameterInfo TA_DEF_UI_Input_Price_HLC =
                                  { TA_Input_Price, "inPriceHLC",
                                    TA_IN_PRICE_HIGH   |
                                    TA_IN_PRICE_LOW    |
                                    TA_IN_PRICE_CLOSE  };

const TA_InputParameterInfo TA_DEF_UI_Input_Price_HL =
                                  { TA_Input_Price, "inPriceHL",
                                    TA_IN_PRICE_HIGH   |
                                    TA_IN_PRICE_LOW    };

const TA_InputParameterInfo TA_DEF_UI_Input_Price_CV =
                                  { TA_Input_Price, "inPriceCV",
                                    TA_IN_PRICE_CLOSE  |
                                    TA_IN_PRICE_VOLUME };

const TA_InputParameterInfo TA_DEF_UI_Input_Price_V =
                                  { TA_Input_Price, "inPriceV",
                                    TA_IN_PRICE_VOLUME };

const TA_InputParameterInfo TA_DEF_UI_Input_Real =
                                  { TA_Input_Real, "inReal", 0 };

const TA_InputParameterInfo TA_DEF_UI_Input_Periods =
                                  { TA_Input_Real, "inPeriods", 0 };

const TA_InputParameterInfo TA_DEF_UI_Input_Real0 =
                                  { TA_Input_Real, "inReal0", 0 };

const TA_InputParameterInfo TA_DEF_UI_Input_Real1 =
                                  { TA_Input_Real, "inReal1", 0 };

const TA_InputParameterInfo TA_DEF_UI_Input_Integer =
                                  { TA_Input_Integer, "inInteger", 0 };


/*************************************************************
 * Define from here the TA_OutputParameterInfo
 * These shall be sufficient for most of the TA functions.
 *************************************************************/
const TA_OutputParameterInfo TA_DEF_UI_Output_Real =
                                  { TA_Output_Real, "outReal", TA_OUT_LINE };

const TA_OutputParameterInfo TA_DEF_UI_Output_Integer =
                                  { TA_Output_Integer, "outInteger", TA_OUT_LINE };

/*****************************************************
 * Define from here the TA_Integer ranges
 ****************************************************/

const TA_IntegerRange TA_DEF_TimePeriod_Positive =
{
   1,             /* min */
   100000,        /* max */
   1,             /* suggested start */
   200,           /* suggested end   */
   1              /* suggested increment */
};

const TA_IntegerRange TA_DEF_TimePeriod_Positive_Minimum5 =
{
   5,             /* min */
   100000,        /* max */
   5,             /* suggested start */
   200,           /* suggested end   */
   1              /* suggested increment */
};

const TA_IntegerRange TA_DEF_TimePeriod_Positive_Minimum2 =
{
   2,             /* min */
   100000,        /* max */
   4,             /* suggested start */
   200,           /* suggested end   */
   1              /* suggested increment */
};

const TA_IntegerRange TA_DEF_HorizontalShiftPeriod =
{
   -200,               /* min */
    200,               /* max   */
   0,                  /* suggested start */
   8,                  /* suggested end   */
   1                   /* suggested increment */
};


/*****************************************************
 * Define from here the TA_Real ranges
 ****************************************************/
const TA_RealRange TA_DEF_VerticalShiftPercent =
{
   -99.0,            /* min */
   99.0,             /* max   */
   1,                /* precision */
   -10.0,            /* suggested start */
   10.0,             /* suggested end   */
   0.5               /* suggested increment */
};

const TA_RealRange TA_DEF_NbDeviation =
{
   TA_REAL_MIN,    /* min */
   TA_REAL_MAX,    /* max */
   2,              /* precision */
   -2.0,           /* suggested start */
   2.0,            /* suggested end   */
   0.2             /* suggested increment */
};

const TA_RealRange TA_DEF_ZeroToOne =
{
   0.00,  /* min */
   1.00,  /* max */
   2,     /* precision */
   0.01,  /* suggested start */
   1.00,  /* suggested end   */
   0.05   /* suggested increment */
};

const TA_RealRange TA_DEF_RealPositive =
{
   0.00,  /* min */
   TA_REAL_MAX,  /* max */
   0,    /* precision */
   0.0,  /* suggested start */
   0.0,  /* suggested end   */
   0.0   /* suggested increment */
};

/*****************************************************
 * Define from here the TA_OptInputParameterInfo.
 ****************************************************/

const TA_OptInputParameterInfo TA_DEF_UI_MinPeriod =
{
   TA_OptInput_IntegerRange, /* type */
   "optInMinPeriod",        /* paramName */
   0,                        /* flags */

   "Minimum Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive_Minimum2, /* dataSet */
   2, /* defaultValue */
   "Value less than minimum will be changed to Minimum period", /* hint */

   NULL /* helpFile */
};

const TA_OptInputParameterInfo TA_DEF_UI_MaxPeriod =
{
   TA_OptInput_IntegerRange, /* type */
   "optInMaxPeriod",        /* paramName */
   0,                        /* flags */

   "Maximum Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive_Minimum2, /* dataSet */
   30, /* defaultValue */
   "Value higher than maximum will be changed to Maximum period", /* hint */

   NULL /* helpFile */
};

const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_30_MINIMUM2 =
{
   TA_OptInput_IntegerRange, /* type */
   "optInTimePeriod",        /* paramName */
   0,                        /* flags */

   "Time Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive_Minimum2, /* dataSet */
   30, /* defaultValue */
   "Number of period", /* hint */

   NULL /* helpFile */
};

const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_21_MINIMUM2 =
{
   TA_OptInput_IntegerRange, /* type */
   "optInTimePeriod",        /* paramName */
   0,                        /* flags */

   "Time Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive_Minimum2, /* dataSet */
   21, /* defaultValue */
   "Number of period", /* hint */

   NULL /* helpFile */
};

const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_14_MINIMUM2 =
{
   TA_OptInput_IntegerRange, /* type */
   "optInTimePeriod",        /* paramName */
   0,                        /* flags */

   "Time Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive_Minimum2, /* dataSet */
   14, /* defaultValue */
   "Number of period", /* hint */

   NULL /* helpFile */
};

const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_14_MINIMUM5 =
{
   TA_OptInput_IntegerRange, /* type */
   "optInTimePeriod",        /* paramName */
   0,                        /* flags */

   "Time Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive_Minimum5, /* dataSet */
   14, /* defaultValue */
   "Number of period", /* hint */

   NULL /* helpFile */
};

const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_10_MINIMUM2 =
{
   TA_OptInput_IntegerRange, /* type */
   "optInTimePeriod",        /* paramName */
   0,                        /* flags */

   "Time Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive_Minimum2, /* dataSet */
   10, /* defaultValue */
   "Number of period", /* hint */

   NULL /* helpFile */
};

const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_5_MINIMUM2 =
{
   TA_OptInput_IntegerRange, /* type */
   "optInTimePeriod",        /* paramName */
   0,                        /* flags */

   "Time Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive_Minimum2, /* dataSet */
   5, /* defaultValue */
   "Number of period", /* hint */

   NULL /* helpFile */
};

const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_30 =
{
   TA_OptInput_IntegerRange, /* type */
   "optInTimePeriod",        /* paramName */
   0,                        /* flags */

   "Time Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive, /* dataSet */
   30, /* defaultValue */
   "Number of period", /* hint */

   NULL /* helpFile */
};

const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_14 =
{
   TA_OptInput_IntegerRange, /* type */
   "optInTimePeriod",        /* paramName */
   0,                        /* flags */

   "Time Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive, /* dataSet */
   14, /* defaultValue */
   "Number of period", /* hint */

   NULL /* helpFile */
};

const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_10 =
{
   TA_OptInput_IntegerRange, /* type */
   "optInTimePeriod",        /* paramName */
   0,                        /* flags */

   "Time Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive, /* dataSet */
   10, /* defaultValue */
   "Number of period", /* hint */

   NULL /* helpFile */
};

const TA_OptInputParameterInfo TA_DEF_UI_TimePeriod_5 =
{
   TA_OptInput_IntegerRange, /* type */
   "optInTimePeriod",        /* paramName */
   0,                        /* flags */

   "Time Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive, /* dataSet */
   5, /* defaultValue */
   "Number of period", /* hint */

   NULL /* helpFile */
};


/* Use for the multiplier of standard deviations. */
const TA_OptInputParameterInfo TA_DEF_UI_NbDeviation =
{
   TA_OptInput_RealRange, /* type */
   "optInNbDev",          /* paramName */
   0,                     /* flags */

   "Deviations",          /* displayName */
   (const void *)&TA_DEF_NbDeviation, /* dataSet */
   1.0, /* defaultValue */
   "Nb of deviations", /* hint */

   NULL /* helpFile */
};

const TA_OptInputParameterInfo TA_DEF_UI_Penetration_30 =
{
   TA_OptInput_RealRange,     /* type */
   "optInPenetration",        /* paramName */
   0,                         /* flags */

   "Penetration",             /* displayName */
   (const void *)&TA_DEF_RealPositive, /* dataSet */
   0.3, /* defaultValue */
   "Percentage of penetration of a candle within another candle", /* hint */

   NULL /* helpFile */
};

const TA_OptInputParameterInfo TA_DEF_UI_Penetration_50 =
{
   TA_OptInput_RealRange,     /* type */
   "optInPenetration",        /* paramName */
   0,                         /* flags */

   "Penetration",             /* displayName */
   (const void *)&TA_DEF_RealPositive, /* dataSet */
   0.5, /* defaultValue */
   "Percentage of penetration of a candle within another candle", /* hint */

   NULL /* helpFile */
};

const TA_OptInputParameterInfo TA_DEF_UI_VerticalShift =
{
   TA_OptInput_RealRange,  /* type */
   "optInVertShift",       /* paramName */
   TA_OPTIN_IS_PERCENT,       /* flags */

   "Vertical Shift",       /* displayName */
   (const void *)&TA_DEF_VerticalShiftPercent, /* dataSet */
   0, /* defaultValue */
   "Positive number shift upwards, negative downwards", /* hint */

   NULL /* helpFile */
};

const TA_OptInputParameterInfo TA_DEF_UI_HorizontalShift =
{
   TA_OptInput_IntegerRange,  /* type */
   "optInHorizShift",         /* paramName */
   0,                         /* flags */

   "Horizontal Shift",        /* displayName */
   (const void *)&TA_DEF_HorizontalShiftPeriod, /* dataSet */
   0, /* defaultValue */
   "Positive number shift 'n' period to the right, negative shift to the left", /* hint */

   NULL /* helpFile */
};

/* Define the values available for the MA function. 
 * 
 * Many TA function using the MA internally may wish to
 * re-use this parameter as one of their own. That way a
 * new MA will provide automatically a new way of calculating
 * to a multitude of other TA functions.
 */
static const TA_IntegerDataPair TA_MA_TypeDataPair[] =
{
   {0,"SMA"},
   {1,"EMA"},
   {2,"WMA"},
   {3,"DEMA" },
   {4,"TEMA" },
   {5,"TRIMA"},
   {6,"KAMA" },
   {7,"MAMA" },
   {8,"T3"}
   /*{...,"Linear Regression"}*/
};

const TA_IntegerList TA_MA_TypeList =
{
   &TA_MA_TypeDataPair[0],
   sizeof(TA_MA_TypeDataPair)/sizeof(TA_IntegerDataPair)
};

const TA_OptInputParameterInfo TA_DEF_UI_MA_Method =
{
   TA_OptInput_IntegerList, /* type */
   "optInMAType",           /* paramName */
   0,                       /* flags */

   "MA Type",                /* displayName */
   (const void *)&TA_MA_TypeList, /* dataSet */
   0, /* defaultValue = simple average */
   "Type of Moving Average", /* hint */

   NULL /* helpFile */
};

/* When a TA function needs two period (often called
 * a fast and slow period).
 */
const TA_OptInputParameterInfo TA_DEF_UI_Fast_Period =
{
   TA_OptInput_IntegerRange, /* type */
   "optInFastPeriod",        /* paramName */
   0,                        /* flags */

   "Fast Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive_Minimum2, /* dataSet */
   12, /* defaultValue */
   "Number of period for the fast MA", /* hint */

   NULL /* helpFile */
};

const TA_OptInputParameterInfo TA_DEF_UI_Slow_Period =
{
   TA_OptInput_IntegerRange, /* type */
   "optInSlowPeriod",        /* paramName */
   0,                        /* flags */

   "Slow Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive_Minimum2, /* dataSet */
   26, /* defaultValue */
   "Number of period for the slow MA", /* hint */

   NULL /* helpFile */
};

