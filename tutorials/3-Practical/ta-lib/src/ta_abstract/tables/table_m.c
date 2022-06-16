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

/*********************************************************************
 * This file contains only TA functions starting with the letter 'M' *
 *********************************************************************/
#include <stddef.h>
#include "ta_abstract.h"
#include "ta_def_ui.h"

/* Follow the 3 steps defined below for adding a new TA Function to this
 * file.
 */

/****************************************************************************
 * Step 1 - Define here the interface to your TA functions with
 *          the macro DEF_FUNCTION.
 *
 ****************************************************************************/

const TA_OptInputParameterInfo TA_DEF_UI_Signal_Period =
{
   TA_OptInput_IntegerRange, /* type */
   "optInSignalPeriod",      /* paramName */
   0,                        /* flags */

   "Signal Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive, /* dataSet */
   9, /* defaultValue */
   "Smoothing for the signal line (nb of period)", /* hint */

   NULL /* CamelCase name */
};

const TA_OptInputParameterInfo TA_DEF_UI_Slow_MA_Type =
{
   TA_OptInput_IntegerList, /* type */
   "optInSlowMAType",       /* paramName */
   0,                       /* flags */

   "Slow MA",                /* displayName */
   (const void *)&TA_MA_TypeList, /* dataSet */
   0, /* defaultValue = simple average */
   "Type of Moving Average for slow MA", /* hint */

   NULL /* CamelCase name */
};

const TA_OptInputParameterInfo TA_DEF_UI_Fast_MA_Type =
{
   TA_OptInput_IntegerList, /* type */
   "optInFastMAType",       /* paramName */
   0,                       /* flags */

   "Fast MA",                /* displayName */
   (const void *)&TA_MA_TypeList, /* dataSet */
   0, /* defaultValue = simple average */
   "Type of Moving Average for fast MA", /* hint */

   NULL /* CamelCase name */
};

const TA_OptInputParameterInfo TA_DEF_UI_Signal_MA_Type =
{
   TA_OptInput_IntegerList, /* type */
   "optInSignalMAType",     /* paramName */
   0,                       /* flags */
   "Signal MA",             /* displayName */
   (const void *)&TA_MA_TypeList, /* dataSet */
   0, /* defaultValue = simple average */
   "Type of Moving Average for signal line", /* hint */

   NULL /* CamelCase name */
};

/* MA BEGIN */
static const TA_InputParameterInfo    *TA_MA_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_MA_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_MA_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_30,
  &TA_DEF_UI_MA_Method,
  NULL
};

DEF_FUNCTION( MA,                         /* name */
              TA_GroupId_OverlapStudies,  /* groupId */
              "Moving average",       /* hint */
              "MovingAverage",            /* CamelCase name */
              TA_FUNC_FLG_OVERLAP         /* flags */
             );
/* MA END */

/* MACD BEGIN */
const TA_OutputParameterInfo TA_DEF_UI_Output_Real_MACD =
                               { TA_Output_Real, "outMACD", TA_OUT_LINE };

const TA_OutputParameterInfo TA_DEF_UI_Output_Real_MACDSignal =
                               { TA_Output_Real, "outMACDSignal", TA_OUT_DASH_LINE };

const TA_OutputParameterInfo TA_DEF_UI_Output_Real_MACDHist =
                                { TA_Output_Real, "outMACDHist", TA_OUT_HISTO };

static const TA_InputParameterInfo    *TA_MACD_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo *TA_MACD_Outputs[]   =
{
  &TA_DEF_UI_Output_Real_MACD,
  &TA_DEF_UI_Output_Real_MACDSignal,
  &TA_DEF_UI_Output_Real_MACDHist,
  NULL
};

static const TA_OptInputParameterInfo *TA_MACD_OptInputs[] =
{ &TA_DEF_UI_Fast_Period,
  &TA_DEF_UI_Slow_Period,
  &TA_DEF_UI_Signal_Period,
  NULL
};

DEF_FUNCTION( MACD,                       /* name */
              TA_GroupId_MomentumIndicators,  /* groupId */
              "Moving Average Convergence/Divergence", /* hint */
              "Macd",                     /* CamelCase name */
              0                           /* flags */
             );
/* MACD END */

/* MACDEXT BEGIN */
static const TA_InputParameterInfo    *TA_MACDEXT_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo *TA_MACDEXT_Outputs[]   =
{
  &TA_DEF_UI_Output_Real_MACD,
  &TA_DEF_UI_Output_Real_MACDSignal,
  &TA_DEF_UI_Output_Real_MACDHist,
  NULL
};

static const TA_OptInputParameterInfo *TA_MACDEXT_OptInputs[] =
{ &TA_DEF_UI_Fast_Period,
  &TA_DEF_UI_Fast_MA_Type,
  &TA_DEF_UI_Slow_Period,
  &TA_DEF_UI_Slow_MA_Type,
  &TA_DEF_UI_Signal_Period,
  &TA_DEF_UI_Signal_MA_Type,
  NULL
};

DEF_FUNCTION( MACDEXT,                     /* name */
              TA_GroupId_MomentumIndicators,  /* groupId */
              "MACD with controllable MA type", /* hint */
              "MacdExt",                  /* CamelCase name */
              0                           /* flags */
             );
/* MACDEXT END */

/* MACDFIX BEGIN */
static const TA_InputParameterInfo    *TA_MACDFIX_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_MACDFIX_Outputs[]   =
{
  &TA_DEF_UI_Output_Real_MACD,
  &TA_DEF_UI_Output_Real_MACDSignal,
  &TA_DEF_UI_Output_Real_MACDHist,
  NULL
};

static const TA_OptInputParameterInfo *TA_MACDFIX_OptInputs[] =
{ &TA_DEF_UI_Signal_Period,
  NULL
};

DEF_FUNCTION( MACDFIX,                    /* name */
              TA_GroupId_MomentumIndicators,  /* groupId */
              "Moving Average Convergence/Divergence Fix 12/26", /* hint */
              "MacdFix",                  /* CamelCase name */
              0                           /* flags */
             );
/* MACDFIX END */

/* MAMA BEGIN */
const TA_RealRange TA_DEF_MAMA_FastLimit =
{
   0.01,          /* min */
   0.99,          /* max */
   2,            /* precision */
   0.21,         /* suggested start */
   0.80,         /* suggested end   */
   0.01          /* suggested increment */
};

const TA_RealRange TA_DEF_MAMA_SlowLimit =
{
   0.01,         /* min */
   0.99,         /* max   */
   2,            /* precision */
   0.01,         /* suggested start */
   0.60,         /* suggested end   */
   0.01          /* suggested increment */
};

const TA_OptInputParameterInfo TA_DEF_UI_MAMA_FastLimit =
{
   TA_OptInput_RealRange, /* type */
   "optInFastLimit",      /* paramName */
   0,                        /* flags */
   "Fast Limit",            /* displayName */
   (const void *)&TA_DEF_MAMA_FastLimit, /* dataSet */
   0.5, /* defaultValue */
   "Upper limit use in the adaptive algorithm", /* hint */
   NULL /* CamelCase name */
};

const TA_OptInputParameterInfo TA_DEF_UI_MAMA_SlowLimit =
{
   TA_OptInput_RealRange, /* type */
   "optInSlowLimit",      /* paramName */
   0,                        /* flags */

   "Slow Limit",            /* displayName */
   (const void *)&TA_DEF_MAMA_SlowLimit, /* dataSet */
   0.05, /* defaultValue */
   "Lower limit use in the adaptive algorithm", /* hint */
   NULL /* CamelCase name */
};

const TA_OutputParameterInfo TA_DEF_UI_Output_Real_MAMA =
                               { TA_Output_Real, "outMAMA", TA_OUT_LINE };

const TA_OutputParameterInfo TA_DEF_UI_Output_Real_FAMA =
                               { TA_Output_Real, "outFAMA", TA_OUT_DASH_LINE };

static const TA_InputParameterInfo    *TA_MAMA_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_MAMA_Outputs[]   =
{
  &TA_DEF_UI_Output_Real_MAMA,
  &TA_DEF_UI_Output_Real_FAMA,
  NULL
};

static const TA_OptInputParameterInfo *TA_MAMA_OptInputs[] =
{ &TA_DEF_UI_MAMA_FastLimit,
  &TA_DEF_UI_MAMA_SlowLimit,
  NULL
};

DEF_FUNCTION( MAMA,                         /* name */
              TA_GroupId_OverlapStudies,  /* groupId */
              "MESA Adaptive Moving Average",       /* hint */
              "Mama",                       /* CamelCase name */
              TA_FUNC_FLG_UNST_PER|TA_FUNC_FLG_OVERLAP /* flags */
             );
/* MAMA END */

/* MAVP BEGIN */
static const TA_InputParameterInfo    *TA_MAVP_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  &TA_DEF_UI_Input_Periods,
  NULL
};

static const TA_OutputParameterInfo   *TA_MAVP_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_MAVP_OptInputs[] =
{ 
  &TA_DEF_UI_MinPeriod,
  &TA_DEF_UI_MaxPeriod,
  &TA_DEF_UI_MA_Method,
  NULL
};

DEF_FUNCTION( MAVP,                         /* name */
              TA_GroupId_OverlapStudies,  /* groupId */
              "Moving average with variable period",  /* hint */
              "MovingAverageVariablePeriod", /* CamelCase name */
              TA_FUNC_FLG_OVERLAP         /* flags */
             );
/* MAVP END */

/* MAX BEGIN */
static const TA_InputParameterInfo    *TA_MAX_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_MAX_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_MAX_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_30_MINIMUM2,
  NULL
};

DEF_FUNCTION( MAX,                       /* name */
              TA_GroupId_MathOperators,  /* groupId */
              "Highest value over a specified period", /* hint */
              "Max",                     /* CamelCase name */
              TA_FUNC_FLG_OVERLAP        /* flags */
             );
/* MAX END */

/* MAXINDEX BEGIN */
static const TA_InputParameterInfo    *TA_MAXINDEX_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_MAXINDEX_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_MAXINDEX_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_30_MINIMUM2,
  NULL
};

DEF_FUNCTION( MAXINDEX,                  /* name */
              TA_GroupId_MathOperators,  /* groupId */
              "Index of highest value over a specified period", /* hint */
              "MaxIndex",                /* CamelCase name */
              0                          /* flags */
             );
/* MAXINDEX END */

/* MEDPRICE BEGIN */
static const TA_InputParameterInfo    *TA_MEDPRICE_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HL,
  NULL
};

static const TA_OutputParameterInfo   *TA_MEDPRICE_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_MEDPRICE_OptInputs[] = { NULL };

DEF_FUNCTION( MEDPRICE,                   /* name */
              TA_GroupId_PriceTransform,  /* groupId */
              "Median Price",             /* hint */
              "MedPrice",                 /* CamelCase name */
              TA_FUNC_FLG_OVERLAP         /* flags */
             );

/* MEDPRICE END */

/* MFI BEGIN */
static const TA_InputParameterInfo    *TA_MFI_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HLCV,
  NULL
};

static const TA_OutputParameterInfo   *TA_MFI_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_MFI_OptInputs[] = 
{
  &TA_DEF_UI_TimePeriod_14_MINIMUM2,
  NULL
};

DEF_FUNCTION( MFI,                   /* name */
              TA_GroupId_MomentumIndicators,  /* groupId */
              "Money Flow Index",         /* hint */
              "Mfi",                      /* CamelCase name */
              TA_FUNC_FLG_UNST_PER        /* flags */
             );
/* MFI END */

/* MIDPRICE BEGIN */
static const TA_InputParameterInfo    *TA_MIDPRICE_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HL,
  NULL
};

static const TA_OutputParameterInfo   *TA_MIDPRICE_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_MIDPRICE_OptInputs[] = 
{ &TA_DEF_UI_TimePeriod_14_MINIMUM2,
  NULL
};

DEF_FUNCTION( MIDPRICE,                    /* name */
              TA_GroupId_OverlapStudies,   /* groupId */
              "Midpoint Price over period",/* hint */
              "MidPrice",                  /* CamelCase name */
              TA_FUNC_FLG_OVERLAP          /* flags */
             );

/* MIDPRICE END */

/* MIDPOINT BEGIN */
static const TA_InputParameterInfo    *TA_MIDPOINT_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_MIDPOINT_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_MIDPOINT_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_14_MINIMUM2,
  NULL
};

DEF_FUNCTION( MIDPOINT,                   /* name */
              TA_GroupId_OverlapStudies,  /* groupId */
              "MidPoint over period",     /* hint */
              "MidPoint",                 /* CamelCase name */
              TA_FUNC_FLG_OVERLAP         /* flags */
             );
/* MIDPOINT END */

/* MIN BEGIN */
static const TA_InputParameterInfo    *TA_MIN_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_MIN_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_MIN_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_30_MINIMUM2,
  NULL
};

DEF_FUNCTION( MIN,                       /* name */
              TA_GroupId_MathOperators,  /* groupId */
              "Lowest value over a specified period", /* hint */
              "Min",                     /* CamelCase name */
              TA_FUNC_FLG_OVERLAP        /* flags */
             );
/* MIN END */

/* MININDEX BEGIN */
static const TA_InputParameterInfo    *TA_MININDEX_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_MININDEX_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer,
  NULL
};

static const TA_OptInputParameterInfo *TA_MININDEX_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_30_MINIMUM2,
  NULL
};

DEF_FUNCTION( MININDEX,                  /* name */
              TA_GroupId_MathOperators,  /* groupId */
              "Index of lowest value over a specified period", /* hint */
              "MinIndex",                /* CamelCase name */
              0                          /* flags */
             );
/* MININDEX END */

/* MINMAX BEGIN */
const TA_OutputParameterInfo TA_DEF_UI_Output_Real_Min =
                               { TA_Output_Real, "outMin", TA_OUT_LINE };

const TA_OutputParameterInfo TA_DEF_UI_Output_Real_Max =
                               { TA_Output_Real, "outMax", TA_OUT_LINE };

static const TA_InputParameterInfo    *TA_MINMAX_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_MINMAX_Outputs[]   =
{
  &TA_DEF_UI_Output_Real_Min,
  &TA_DEF_UI_Output_Real_Max,
  NULL
};

static const TA_OptInputParameterInfo *TA_MINMAX_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_30_MINIMUM2,
  NULL
};

DEF_FUNCTION( MINMAX,                    /* name */
              TA_GroupId_MathOperators,  /* groupId */
              "Lowest and highest values over a specified period", /* hint */
              "MinMax",                  /* CamelCase name */
              TA_FUNC_FLG_OVERLAP        /* flags */
             );
/* MINMAX END */

/* MINMAXINDEX BEGIN */
const TA_OutputParameterInfo TA_DEF_UI_Output_Integer_MinIdx =
                               { TA_Output_Integer, "outMinIdx", TA_OUT_LINE };

const TA_OutputParameterInfo TA_DEF_UI_Output_Integer_MaxIdx =
                               { TA_Output_Integer, "outMaxIdx", TA_OUT_LINE };

static const TA_InputParameterInfo    *TA_MINMAXINDEX_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_MINMAXINDEX_Outputs[]   =
{
  &TA_DEF_UI_Output_Integer_MinIdx,
  &TA_DEF_UI_Output_Integer_MaxIdx,
  NULL
};

static const TA_OptInputParameterInfo *TA_MINMAXINDEX_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_30_MINIMUM2,
  NULL
};

DEF_FUNCTION( MINMAXINDEX,               /* name */
              TA_GroupId_MathOperators,  /* groupId */
              "Indexes of lowest and highest values over a specified period", /* hint */
              "MinMaxIndex",             /* CamelCase name */
              0                          /* flags */
             );
/* MINMAXINDEX END */

/* MINUS_DI BEGIN */
static const TA_InputParameterInfo    *TA_MINUS_DI_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_MINUS_DI_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_MINUS_DI_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_14,
  NULL
};

DEF_FUNCTION( MINUS_DI,                      /* name */
              TA_GroupId_MomentumIndicators,    /* groupId */
              "Minus Directional Indicator", /* hint */
              "MinusDI",                     /* CamelCase name */
              TA_FUNC_FLG_UNST_PER           /* flags */              
             );

/* MINUS_DI END */

/* MINUS_DM BEGIN */
static const TA_InputParameterInfo    *TA_MINUS_DM_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HL,
  NULL
};

static const TA_OutputParameterInfo   *TA_MINUS_DM_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_MINUS_DM_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_14,
  NULL
};

DEF_FUNCTION( MINUS_DM,                     /* name */
              TA_GroupId_MomentumIndicators,   /* groupId */
              "Minus Directional Movement", /* hint */
              "MinusDM",                    /* CamelCase name */
              TA_FUNC_FLG_UNST_PER          /* flags */
             );

/* MINUS_DM END */

/* MOM BEGIN */
static const TA_InputParameterInfo    *TA_MOM_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_MOM_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_MOM_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_10,
  NULL
};

DEF_FUNCTION( MOM,                     /* name */
              TA_GroupId_MomentumIndicators,  /* groupId */
              "Momentum",        /* hint */
              "Mom",             /* CamelCase name */
              0                  /* flags */
             );
/* MOM END */

/* MULT BEGIN */
DEF_MATH_BINARY_OPERATOR( MULT, "Vector Arithmetic Mult", "Mult" )
/* MULT END */

/****************************************************************************
 * Step 2 - Add your TA function to the table.
 *          Keep in alphabetical order. Must be NULL terminated.
 ****************************************************************************/
const TA_FuncDef *TA_DEF_TableM[] =
{
   ADD_TO_TABLE(MA),
   ADD_TO_TABLE(MACD),
   ADD_TO_TABLE(MACDEXT),
   ADD_TO_TABLE(MACDFIX),
   ADD_TO_TABLE(MAMA),
   ADD_TO_TABLE(MAVP),
   ADD_TO_TABLE(MAX),
   ADD_TO_TABLE(MAXINDEX),
   ADD_TO_TABLE(MEDPRICE),
   ADD_TO_TABLE(MFI),
   ADD_TO_TABLE(MIDPOINT),
   ADD_TO_TABLE(MIDPRICE),
   ADD_TO_TABLE(MIN),
   ADD_TO_TABLE(MININDEX),
   ADD_TO_TABLE(MINMAX),
   ADD_TO_TABLE(MINMAXINDEX),
   ADD_TO_TABLE(MINUS_DI),
   ADD_TO_TABLE(MINUS_DM),
   ADD_TO_TABLE(MOM),
   ADD_TO_TABLE(MULT),
   NULL
};


/* Do not modify the following line. */
const unsigned int TA_DEF_TableMSize =
              ((sizeof(TA_DEF_TableM)/sizeof(TA_FuncDef *))-1);


/****************************************************************************
 * Step 3 - Make sure "gen_code" is executed for generating all other
 *          source files derived from this one.
 *          You can then re-compile the library as usual and you are done!
 ****************************************************************************/
