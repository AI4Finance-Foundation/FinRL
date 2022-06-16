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
 * This file contains only TA functions starting with the letter 'S' *
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

/* SAR BEGIN */
static const TA_RealRange TA_DEF_AccelerationFactor =
{
   0.0,          /* min */
   TA_REAL_MAX,  /* max */
   4,      /* precision */
   0.01,  /* suggested start */
   0.20,  /* suggested end   */
   0.01   /* suggested increment */
};

static const TA_RealRange TA_DEF_AccelerationMax =
{
   0.0,         /* min */
   TA_REAL_MAX, /* max */
   4,     /* precision */
   0.20,  /* suggested start */
   0.40,  /* suggested end   */
   0.01   /* suggested increment */
};

static const TA_OptInputParameterInfo TA_DEF_UI_D_AccelerationFactor =
{
   TA_OptInput_RealRange, /* type */
   "optInAcceleration",  /* paramName */
   0,          /* flags */

   "Acceleration Factor", /* displayName */
   (const void *)&TA_DEF_AccelerationFactor, /* dataSet */
   0.02, /* defaultValue */
   "Acceleration Factor used up to the Maximum value", /* hint */
   NULL /* CamelCase name */
};

static const TA_OptInputParameterInfo TA_DEF_UI_D_AccelerationMaximum =
{
   TA_OptInput_RealRange, /* type */
   "optInMaximum",        /* paramName */
   0,                     /* flags */

   "AF Maximum", /* displayName */
   (const void *)&TA_DEF_AccelerationMax, /* dataSet */
   0.20, /* defaultValue */
   "Acceleration Factor Maximum value", /* hint */

   NULL /* CamelCase name */
};

static const TA_InputParameterInfo    *TA_SAR_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HL,
  NULL
};

static const TA_OutputParameterInfo   *TA_SAR_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_SAR_OptInputs[] =
{ &TA_DEF_UI_D_AccelerationFactor,
  &TA_DEF_UI_D_AccelerationMaximum,
  NULL
};

DEF_FUNCTION( SAR,                        /* name */
              TA_GroupId_OverlapStudies,  /* groupId */
              "Parabolic SAR",            /* hint */
              "Sar",                      /* CamelCase name */
              TA_FUNC_FLG_OVERLAP         /* flags */
             );

/* SAR END */

/* SAREXT BEGIN */
static const TA_RealRange TA_DEF_AccelerationInit =
{
   0.0,         /* min */
   TA_REAL_MAX, /* max */
   4,     /* precision */
   0.01,  /* suggested start */
   0.19,  /* suggested end   */
   0.01   /* suggested increment */
};

static const TA_RealRange TA_DEF_SARStartValue =
{
   TA_REAL_MIN, /* min */
   TA_REAL_MAX, /* max */
   4,     /* precision */
   0, /* suggested start */
   0, /* suggested end   */
   0  /* suggested increment */
};

static const TA_RealRange TA_DEF_SAROffsetOnReverse =
{
   0.0,         /* min */
   TA_REAL_MAX, /* max */
   4,     /* precision */
   0.01,  /* suggested start */
   0.15,  /* suggested end   */
   0.01   /* suggested increment */
};

static const TA_OptInputParameterInfo TA_DEF_UI_D_StartValue =
{
   TA_OptInput_RealRange, /* type */
   "optInStartValue",        /* paramName */
   0,                     /* flags */

   "Start Value", /* displayName */
   (const void *)&TA_DEF_SARStartValue, /* dataSet */
   0.0, /* defaultValue */
   "Start value and direction. 0 for Auto, >0 for Long, <0 for Short", /* hint */

   NULL /* CamelCase name */
};

static const TA_OptInputParameterInfo TA_DEF_UI_D_OffsetOnReverse =
{
   TA_OptInput_RealRange,  /* type */
   "optInOffsetOnReverse", /* paramName */
   0,                      /* flags */

   "Offset on Reverse", /* displayName */
   (const void *)&TA_DEF_SAROffsetOnReverse, /* dataSet */
   0.0, /* defaultValue */
   "Percent offset added/removed to initial stop on short/long reversal", /* hint */

   NULL /* CamelCase name */
};

static const TA_OptInputParameterInfo TA_DEF_UI_D_AccelerationInitLong =
{
   TA_OptInput_RealRange, /* type */
   "optInAccelerationInitLong",        /* paramName */
   0,                     /* flags */

   "AF Init Long", /* displayName */
   (const void *)&TA_DEF_AccelerationInit, /* dataSet */
   0.02, /* defaultValue */
   "Acceleration Factor initial value for the Long direction", /* hint */

   NULL /* CamelCase name */
};

static const TA_OptInputParameterInfo TA_DEF_UI_D_AccelerationLong =
{
   TA_OptInput_RealRange, /* type */
   "optInAccelerationLong",  /* paramName */
   0,          /* flags */

   "AF Long", /* displayName */
   (const void *)&TA_DEF_AccelerationFactor, /* dataSet */
   0.02, /* defaultValue */
   "Acceleration Factor for the Long direction", /* hint */
   NULL /* CamelCase name */
};

static const TA_OptInputParameterInfo TA_DEF_UI_D_AccelerationMaxLong =
{
   TA_OptInput_RealRange, /* type */
   "optInAccelerationMaxLong",        /* paramName */
   0,                     /* flags */

   "AF Max Long", /* displayName */
   (const void *)&TA_DEF_AccelerationMax, /* dataSet */
   0.20, /* defaultValue */
   "Acceleration Factor maximum value for the Long direction", /* hint */

   NULL /* CamelCase name */
};

static const TA_OptInputParameterInfo TA_DEF_UI_D_AccelerationInitShort =
{
   TA_OptInput_RealRange, /* type */
   "optInAccelerationInitShort",        /* paramName */
   0,                     /* flags */

   "AF Init Short", /* displayName */
   (const void *)&TA_DEF_AccelerationInit, /* dataSet */
   0.02, /* defaultValue */
   "Acceleration Factor initial value for the Short direction", /* hint */

   NULL /* CamelCase name */
};

static const TA_OptInputParameterInfo TA_DEF_UI_D_AccelerationShort =
{
   TA_OptInput_RealRange, /* type */
   "optInAccelerationShort",  /* paramName */
   0,          /* flags */

   "AF Short", /* displayName */
   (const void *)&TA_DEF_AccelerationFactor, /* dataSet */
   0.02, /* defaultValue */
   "Acceleration Factor for the Short direction", /* hint */
   NULL /* CamelCase name */
};

static const TA_OptInputParameterInfo TA_DEF_UI_D_AccelerationMaxShort =
{
   TA_OptInput_RealRange, /* type */
   "optInAccelerationMaxShort",        /* paramName */
   0,                     /* flags */

   "AF Max Short", /* displayName */
   (const void *)&TA_DEF_AccelerationMax, /* dataSet */
   0.20, /* defaultValue */
   "Acceleration Factor maximum value for the Short direction", /* hint */

   NULL /* CamelCase name */
};

static const TA_InputParameterInfo    *TA_SAREXT_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HL,
  NULL
};

static const TA_OutputParameterInfo   *TA_SAREXT_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_SAREXT_OptInputs[] =
{ &TA_DEF_UI_D_StartValue,
  &TA_DEF_UI_D_OffsetOnReverse,
  &TA_DEF_UI_D_AccelerationInitLong,
  &TA_DEF_UI_D_AccelerationLong,
  &TA_DEF_UI_D_AccelerationMaxLong,
  &TA_DEF_UI_D_AccelerationInitShort,
  &TA_DEF_UI_D_AccelerationShort,
  &TA_DEF_UI_D_AccelerationMaxShort,
  NULL
};

DEF_FUNCTION( SAREXT,                     /* name */
              TA_GroupId_OverlapStudies,  /* groupId */
              "Parabolic SAR - Extended", /* hint */
              "SarExt",                   /* CamelCase name */
              TA_FUNC_FLG_OVERLAP         /* flags */
             );

/* SAREXT END */


/* SIN BEGIN */
DEF_MATH_UNARY_OPERATOR( SIN, "Vector Trigonometric Sin", "Sin" )
/* SIN END */

/* SINH BEGIN */
DEF_MATH_UNARY_OPERATOR( SINH, "Vector Trigonometric Sinh", "Sinh" )
/* SINH END */

/* SMA BEGIN */
static const TA_InputParameterInfo    *TA_SMA_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_SMA_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_SMA_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_30_MINIMUM2,
  NULL
};

DEF_FUNCTION( SMA,                        /* name */
              TA_GroupId_OverlapStudies,  /* groupId */
              "Simple Moving Average",    /* hint */
              "Sma",                      /* CamelCase name */
              TA_FUNC_FLG_OVERLAP         /* flags */
             );

/* SMA END */

/* SQRT BEGIN */
DEF_MATH_UNARY_OPERATOR( SQRT, "Vector Square Root", "Sqrt" )
/* SQRT END */

/* SUB BEGIN */
DEF_MATH_BINARY_OPERATOR( SUB, "Vector Arithmetic Substraction", "Sub" )
/* SUB END */

/* SUM BEGIN */
static const TA_InputParameterInfo    *TA_SUM_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_SUM_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_SUM_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_30_MINIMUM2,
  NULL
};

DEF_FUNCTION( SUM, /* name */
              TA_GroupId_MathOperators, /* groupId */
              "Summation", /* hint */
              "Sum",   /* CamelCase name */
              0        /* flags */
             );
/* SUM END */

/* STDDEV BEGIN */
static const TA_InputParameterInfo    *TA_STDDEV_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_STDDEV_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_STDDEV_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_5_MINIMUM2,
  &TA_DEF_UI_NbDeviation,
  NULL
};

DEF_FUNCTION( STDDEV,                   /* name */
              TA_GroupId_Statistic,     /* groupId */
              "Standard Deviation",     /* hint */
              "StdDev",                 /* CamelCase name */
              0                         /* flags */
             );
/* STDDEV END */

/* STOCH BEGIN */
static const TA_OptInputParameterInfo TA_DEF_UI_FastK_Period =
{
   TA_OptInput_IntegerRange, /* type */
   "optInFastK_Period",           /* paramName */
   0,                        /* flags */

   "Fast-K Period", /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive, /* dataSet */
   5, /* defaultValue */
   "Time period for building the Fast-K line", /* hint */

   NULL /* CamelCase name */
};

static const TA_OptInputParameterInfo TA_DEF_UI_SlowK_Period =
{
   TA_OptInput_IntegerRange, /* type */
   "optInSlowK_Period",       /* paramName */
   0,                        /* flags */

   "Slow-K Period",     /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive, /* dataSet */
   3, /* defaultValue */
   "Smoothing for making the Slow-K line. Usually set to 3", /* hint */

   NULL /* CamelCase name */
};

static const TA_OptInputParameterInfo TA_DEF_UI_SlowD_Period =
{
   TA_OptInput_IntegerRange, /* type */
   "optInSlowD_Period",           /* paramName */
   0,                        /* flags */

   "Slow-D Period",            /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive, /* dataSet */
   3, /* defaultValue */
   "Smoothing for making the Slow-D line", /* hint */

   NULL /* CamelCase name */
};

const TA_OptInputParameterInfo TA_DEF_UI_SlowK_MAType =
{
   TA_OptInput_IntegerList, /* type */
   "optInSlowK_MAType",     /* paramName */
   0,                       /* flags */

   "Slow-K MA",                /* displayName */
   (const void *)&TA_MA_TypeList, /* dataSet */
   0, /* defaultValue = simple average */
   "Type of Moving Average for Slow-K", /* hint */

   NULL /* CamelCase name */
};

const TA_OptInputParameterInfo TA_DEF_UI_SlowD_MAType =
{
   TA_OptInput_IntegerList, /* type */
   "optInSlowD_MAType",     /* paramName */
   0,                       /* flags */

   "Slow-D MA",                /* displayName */
   (const void *)&TA_MA_TypeList, /* dataSet */
   0, /* defaultValue = simple average */
   "Type of Moving Average for Slow-D", /* hint */

   NULL /* CamelCase name */
};

const TA_OutputParameterInfo TA_DEF_UI_Output_SlowK =
                               { TA_Output_Real, "outSlowK", TA_OUT_DASH_LINE };

const TA_OutputParameterInfo TA_DEF_UI_Output_SlowD =
                               { TA_Output_Real, "outSlowD", TA_OUT_DASH_LINE };

static const TA_InputParameterInfo    *TA_STOCH_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_STOCH_Outputs[]   =
{
  &TA_DEF_UI_Output_SlowK,
  &TA_DEF_UI_Output_SlowD,
  NULL
};

static const TA_OptInputParameterInfo *TA_STOCH_OptInputs[] =
{ &TA_DEF_UI_FastK_Period,
  &TA_DEF_UI_SlowK_Period,
  &TA_DEF_UI_SlowK_MAType,
  &TA_DEF_UI_SlowD_Period,
  &TA_DEF_UI_SlowD_MAType,
  NULL
};

DEF_FUNCTION( STOCH,                   /* name */
              TA_GroupId_MomentumIndicators, /* groupId */
              "Stochastic",             /* hint */
              "Stoch",                  /* CamelCase name */
              0                         /* flags */
             );
/* STOCH END */

/* STOCHF BEGIN */
static const TA_OptInputParameterInfo TA_DEF_UI_FastD_Period =
{
   TA_OptInput_IntegerRange, /* type */
   "optInFastD_Period",       /* paramName */
   0,                        /* flags */

   "Fast-D Period",     /* displayName */
   (const void *)&TA_DEF_TimePeriod_Positive, /* dataSet */
   3, /* defaultValue */
   "Smoothing for making the Fast-D line. Usually set to 3", /* hint */

   NULL /* CamelCase name */
};

const TA_OptInputParameterInfo TA_DEF_UI_FastD_MAType =
{
   TA_OptInput_IntegerList, /* type */
   "optInFastD_MAType",     /* paramName */
   0,                       /* flags */

   "Fast-D MA",                /* displayName */
   (const void *)&TA_MA_TypeList, /* dataSet */
   0, /* defaultValue = simple average */
   "Type of Moving Average for Fast-D", /* hint */

   NULL /* CamelCase name */
};

const TA_OutputParameterInfo TA_DEF_UI_Output_FastK =
                               { TA_Output_Real, "outFastK", TA_OUT_LINE };

const TA_OutputParameterInfo TA_DEF_UI_Output_FastD =
                               { TA_Output_Real, "outFastD", TA_OUT_LINE };

static const TA_InputParameterInfo    *TA_STOCHF_Inputs[]    =
{
  &TA_DEF_UI_Input_Price_HLC,
  NULL
};

static const TA_OutputParameterInfo   *TA_STOCHF_Outputs[]   =
{
  &TA_DEF_UI_Output_FastK,
  &TA_DEF_UI_Output_FastD,
  NULL
};

static const TA_OptInputParameterInfo *TA_STOCHF_OptInputs[] =
{ &TA_DEF_UI_FastK_Period,
  &TA_DEF_UI_FastD_Period,
  &TA_DEF_UI_FastD_MAType,
  NULL
};

DEF_FUNCTION( STOCHF,                   /* name */
              TA_GroupId_MomentumIndicators, /* groupId */
              "Stochastic Fast",        /* hint */
              "StochF",                 /* CamelCase name */
              0                         /* flags */
             );
/* STOCHF END */

/* STOCHRSI BEGIN */
static const TA_InputParameterInfo    *TA_STOCHRSI_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_STOCHRSI_Outputs[]   =
{
  &TA_DEF_UI_Output_FastK,
  &TA_DEF_UI_Output_FastD,
  NULL
};

static const TA_OptInputParameterInfo *TA_STOCHRSI_OptInputs[] =
{
  &TA_DEF_UI_TimePeriod_14_MINIMUM2,
  &TA_DEF_UI_FastK_Period,
  &TA_DEF_UI_FastD_Period,
  &TA_DEF_UI_FastD_MAType,
  NULL
};

DEF_FUNCTION( STOCHRSI,                        /* name */
              TA_GroupId_MomentumIndicators,  /* groupId */
              "Stochastic Relative Strength Index",  /* hint */
              "StochRsi",                 /* CamelCase name */
              TA_FUNC_FLG_UNST_PER        /* flags */
             );

/* STOCHRSI END */

/****************************************************************************
 * Step 2 - Add your TA function to the table.
 *          Keep in alphabetical order. Must be NULL terminated.
 ****************************************************************************/
const TA_FuncDef *TA_DEF_TableS[] =
{
   ADD_TO_TABLE(SAR),
   ADD_TO_TABLE(SAREXT),
   ADD_TO_TABLE(SIN),
   ADD_TO_TABLE(SINH),
   ADD_TO_TABLE(SMA),
   ADD_TO_TABLE(SQRT),
   ADD_TO_TABLE(STDDEV),
   ADD_TO_TABLE(STOCH),
   ADD_TO_TABLE(STOCHF),
   ADD_TO_TABLE(STOCHRSI),
   ADD_TO_TABLE(SUB),
   ADD_TO_TABLE(SUM),
   NULL
};


/* Do not modify the following line. */
const unsigned int TA_DEF_TableSSize =
              ((sizeof(TA_DEF_TableS)/sizeof(TA_FuncDef *))-1);


/****************************************************************************
 * Step 3 - Make sure "gen_code" is executed for generating all other
 *          source files derived from this one.
 *          You can then re-compile the library as usual and you are done!
 ****************************************************************************/
