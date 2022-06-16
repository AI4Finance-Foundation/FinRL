
# ta_regrest is used for testing and should
# not be installed.
noinst_PROGRAMS = ta_regtest

ta_regtest_SOURCES = ta_regtest.c \
	test_data.c \
	test_util.c \
	test_abstract.c \
	ta_test_func/test_adx.c \
	ta_test_func/test_mom.c \
	ta_test_func/test_sar.c \
	ta_test_func/test_rsi.c \
	ta_test_func/test_candlestick.c \
	ta_test_func/test_per_ema.c \
	ta_test_func/test_per_hlc.c \
	ta_test_func/test_stoch.c \
	ta_test_func/test_macd.c \
	ta_test_func/test_minmax.c \
	ta_test_func/test_per_hlcv.c \
	ta_test_func/test_1in_1out.c \
	ta_test_func/test_1in_2out.c \
	ta_test_func/test_per_ohlc.c \
	ta_test_func/test_stddev.c \
	ta_test_func/test_bbands.c \
	ta_test_func/test_ma.c \
	ta_test_func/test_po.c \
	ta_test_func/test_per_hl.c \
	ta_test_func/test_trange.c \
	test_internals.c

ta_regtest_CPPFLAGS = -I../../ta_func \
		      -I../../ta_common/trio \
		      -I../../ta_common/mt \
		      -I../../ta_common \
		      -I../../ta_abstract
ta_regtest_LDFLAGS = -L../.. -lta_lib \
		     -lm
