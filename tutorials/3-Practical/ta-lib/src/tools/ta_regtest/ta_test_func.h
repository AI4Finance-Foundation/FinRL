#ifndef TA_TEST_FUNC_H
#define TA_TEST_FUNC_H

ErrorNumber test_candlestick  ( TA_History *history );
ErrorNumber test_func_adx     ( TA_History *history );
ErrorNumber test_func_ma      ( TA_History *history );
ErrorNumber test_func_trange  ( TA_History *history );
ErrorNumber test_func_rsi     ( TA_History *history );
ErrorNumber test_func_po      ( TA_History *history );
ErrorNumber test_func_macd    ( TA_History *history );
ErrorNumber test_func_stddev  ( TA_History *history );
ErrorNumber test_func_bbands  ( TA_History *history );
ErrorNumber test_func_mom_roc ( TA_History *history );
ErrorNumber test_func_stoch   ( TA_History *history );
ErrorNumber test_func_minmax  ( TA_History *history );
ErrorNumber test_func_per_ema ( TA_History *history );
ErrorNumber test_func_per_hl  ( TA_History *history );
ErrorNumber test_func_per_hlc ( TA_History *history );
ErrorNumber test_func_per_hlcv( TA_History *history );
ErrorNumber test_func_per_ohlc( TA_History *history );
ErrorNumber test_func_per     ( TA_History *history );
ErrorNumber test_func_sar     ( TA_History *history );
ErrorNumber test_func_1in_1out( TA_History *history );
ErrorNumber test_func_1in_2out( TA_History *history );


#endif
