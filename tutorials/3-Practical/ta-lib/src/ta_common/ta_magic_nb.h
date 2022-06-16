#ifndef TA_MAGIC_NB_H
#define TA_MAGIC_NB_H

/* Many allocated structures contains a magic number.
 *
 * These numbers are used solely to make sure that when a pointer is
 * provided, it is really pointing on the expected type of data.
 * It helps also for the detection of memory corruption.
 * This mechanism is simple, but add a non-negligeable level of
 * reliability at a very low cost (speed/memory wise).
 */
#define TA_FUNC_DEF_MAGIC_NB            0xA201B201
#define TA_PARAM_HOLDER_PRIV_MAGIC_NB   0xA202B202
#define TA_LIBC_PRIV_MAGIC_NB           0xA203B203
#define TA_UDBASE_MAGIC_NB              0xA204B204
#define TA_CATEGORY_TABLE_MAGIC_NB      0xA205B205
#define TA_SYMBOL_TABLE_MAGIC_NB        0xA206B206
#define TA_WEBPAGE_MAGIC_NB             0xA207B207
#define TA_STREAM_MAGIC_NB              0xA208B208
#define TA_STREAM_ACCESS_MAGIC_NB       0xA209B209
#define TA_YAHOO_IDX_MAGIC_NB           0xA20AB20A
#define TA_STRING_TABLE_GROUP_MAGIC_NB  0xA20BB20B
#define TA_STRING_TABLE_FUNC_MAGIC_NB   0xA20CB20C
#define TA_MARKET_PAGE_MAGIC_NB         0xA20DB20D
#define TA_TRADELOGPRIV_MAGIC_NB        0xA20EB20E
#define TA_PMPRIV_MAGIC_NB              0xA20FB20F
#define TA_PMREPORT_MAGIC_NB            0xA210B210
#define TA_TRADEREPORT_MAGIC_NB         0xA211B211
#define TA_HISTORY_MAGIC_NB             0xA212B212

#endif
