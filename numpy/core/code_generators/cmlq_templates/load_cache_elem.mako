    CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

%if locality_stats:
<%namespace file="cache_stats_macro.mako" import="*"/>

    CMLQCacheStatsElem *cache_stats = &elem->stats;
    cache_stats->opname = "${opname}";

<%count_stat("op_exec_count")%>
%endif