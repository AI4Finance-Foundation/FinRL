
noinst_LTLIBRARIES = libta_abstract.la libta_abstract_gc.la

libta_abstract_la_SOURCES = ta_group_idx.c \
	ta_def_ui.c \
	ta_abstract.c \
	ta_func_api.c \
	frames/ta_frame.c \
	tables/table_a.c \
	tables/table_b.c \
	tables/table_c.c \
	tables/table_d.c \
	tables/table_e.c \
	tables/table_f.c \
	tables/table_g.c \
	tables/table_h.c \
	tables/table_i.c \
	tables/table_j.c \
	tables/table_k.c \
	tables/table_l.c \
	tables/table_m.c \
	tables/table_n.c \
	tables/table_o.c \
	tables/table_p.c \
	tables/table_q.c \
	tables/table_r.c \
	tables/table_s.c \
	tables/table_t.c \
	tables/table_u.c \
	tables/table_v.c \
	tables/table_w.c \
	tables/table_x.c \
	tables/table_y.c \
	tables/table_z.c
	
libta_abstract_gc_la_SOURCES = $(libta_abstract_la_SOURCES)

libta_abstract_la_LDFLAGS = -version-info $(TALIB_LIBRARY_VERSION)
libta_abstract_gc_la_LDFLAGS = $(libta_abstract_la_LDFLAGS)

libta_abstract_la_CPPFLAGS = -I../ta_common/ -Iframes/

# The 'gc' version is a minimal version used to just to compile gen_code
libta_abstract_gc_la_CPPFLAGS = -DTA_GEN_CODE $(libta_abstract_la_CPPFLAGS)

libta_abstractdir=$(includedir)/ta-lib/
libta_abstract_HEADERS = ../../include/ta_defs.h \
	../../include/ta_libc.h \
	../../include/ta_abstract.h
