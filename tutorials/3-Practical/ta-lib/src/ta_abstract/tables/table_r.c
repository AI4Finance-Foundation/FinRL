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
 * This file contains only TA functions starting with the letter 'R' *
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

/* ROC BEGIN */
static const TA_InputParameterInfo    *TA_ROC_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_ROC_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_ROC_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_10,
  NULL
};

DEF_FUNCTION( ROC,                     /* name */
              TA_GroupId_MomentumIndicators,  /* groupId */
              "Rate of change : ((price/prevPrice)-1)*100", /* hint */
              "Roc",            /* CamelCase name */
              0                 /* flags */
             );
/* ROC END */

/* ROCP BEGIN */
static const TA_InputParameterInfo    *TA_ROCP_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_ROCP_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_ROCP_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_10,
  NULL
};

DEF_FUNCTION( ROCP,                    /* name */
              TA_GroupId_MomentumIndicators,  /* groupId */
              "Rate of change Percentage: (price-prevPrice)/prevPrice", /* hint */
              "RocP",           /* CamelCase name */
              0                 /* flags */
             );
/* ROCP END */

/* ROCR BEGIN */
static const TA_InputParameterInfo    *TA_ROCR_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_ROCR_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_ROCR_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_10,
  NULL
};

DEF_FUNCTION( ROCR,                    /* name */
              TA_GroupId_MomentumIndicators,  /* groupId */
              "Rate of change ratio: (price/prevPrice)", /* hint */
              "RocR",           /* CamelCase name */
              0                 /* flags */
             );
/* ROCR END */

/* ROCR100 BEGIN */
static const TA_InputParameterInfo    *TA_ROCR100_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_ROCR100_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_ROCR100_OptInputs[] =
{ &TA_DEF_UI_TimePeriod_10,
  NULL
};

DEF_FUNCTION( ROCR100,                    /* name */
              TA_GroupId_MomentumIndicators,  /* groupId */
              "Rate of change ratio 100 scale: (price/prevPrice)*100", /* hint */
              "RocR100",       /* CamelCase name */
              0                /* flags */
             );
/* ROCR100 END */

/* RSI BEGIN */
static const TA_InputParameterInfo    *TA_RSI_Inputs[]    =
{
  &TA_DEF_UI_Input_Real,
  NULL
};

static const TA_OutputParameterInfo   *TA_RSI_Outputs[]   =
{
  &TA_DEF_UI_Output_Real,
  NULL
};

static const TA_OptInputParameterInfo *TA_RSI_OptInputs[] =
{
  &TA_DEF_UI_TimePeriod_14_MINIMUM2,
  NULL
};

DEF_FUNCTION( RSI,                        /* name */
              TA_GroupId_MomentumIndicators,  /* groupId */
              "Relative Strength Index",  /* hint */
              "Rsi",                      /* CamelCase name */
              TA_FUNC_FLG_UNST_PER        /* flags */
             );
/* RSI END */

/****************************************************************************
 * Step 2 - Add your TA function to the table.
 *          Keep in alphabetical order. Must be NULL terminated.
 ****************************************************************************/
const TA_FuncDef *TA_DEF_TableR[] =
{
   ADD_TO_TABLE(ROC),
   ADD_TO_TABLE(ROCP),
   ADD_TO_TABLE(ROCR),
   ADD_TO_TABLE(ROCR100),
   ADD_TO_TABLE(RSI),
   NULL
};


/* Do not modify the following line. */
const unsigned int TA_DEF_TableRSize =
              ((sizeof(TA_DEF_TableR)/sizeof(TA_FuncDef *))-1);


/****************************************************************************
 * Step 3 - Make sure "gen_code" is executed for generating all other
 *          source files derived from this one.
 *          You can then re-compile the library as usual and you are done!
 ****************************************************************************/
