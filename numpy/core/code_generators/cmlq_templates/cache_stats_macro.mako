<%def name="count_stat(name, condition=None)">
    % if locality_stats:
        %if condition is not None:
        if (${condition}) {
        %endif
        cache_stats->${name}++;
        %if condition is not None:
        }
        %endif
    % endif
</%def>