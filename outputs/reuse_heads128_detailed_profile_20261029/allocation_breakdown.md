# Allocation detail: heads128 original reuse_explore

Instrumented forward wall time 87.142905s; allocation interval 56.639786s. One warmup plus one diagnostic forward, GPU7. Extra inner events add overhead; compare neither individual phase totals nor full-forward times to earlier instrumentation as an algorithm speed change.

| Function | Seconds | Allocation share | Full forward share | Calls | Mean ms/call |
|---|---:|---:|---:|---:|---:|
| top_history | 8.811893 | 15.558% | 10.112% | 11983 | 0.735366 |
| operating_points | 32.881241 | 58.053% | 37.733% | 11983 | 2.743991 |
| budget_states | 7.391378 | 13.050% | 8.482% | 11983 | 0.616822 |
| pack_support | 6.806485 | 12.017% | 7.811% | 11983 | 0.568012 |
| Setup and interval gaps | 0.748789 | 1.322% | 0.859% | N/A | N/A |

11983 calls per function = 23 allocated layers x 521 windows; layer0 is representative seed only. Each full window handles 8 query groups. There are 26009 fine clusters, 4164 query groups, 11 states per fine cluster, a 64D head-mean Value contribution proxy and 16 budget dual iterations.

Per-layer nested parent and child timings (do not sum both):

| Global layer | Total allocation s | History s | State errors s | Budget s | Packing s |
|---|---:|---:|---:|---:|---:|
| 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | 2.387374 | 0.360667 | 1.377101 | 0.321983 | 0.294938 |
| 2 | 2.358306 | 0.361922 | 1.360629 | 0.312323 | 0.291857 |
| 3 | 2.352040 | 0.360295 | 1.356687 | 0.312627 | 0.291122 |
| 4 | 2.428975 | 0.442466 | 1.354750 | 0.309738 | 0.290351 |
| 5 | 2.355811 | 0.362605 | 1.356916 | 0.312014 | 0.292763 |
| 6 | 2.428467 | 0.361136 | 1.406659 | 0.328340 | 0.299860 |
| 7 | 2.349692 | 0.361299 | 1.353434 | 0.310787 | 0.292500 |
| 8 | 2.672279 | 0.489587 | 1.502598 | 0.342046 | 0.305047 |
| 9 | 2.394167 | 0.367950 | 1.373698 | 0.322618 | 0.296280 |
| 10 | 2.353836 | 0.364138 | 1.355053 | 0.311308 | 0.291999 |
| 11 | 2.359102 | 0.361951 | 1.360442 | 0.313629 | 0.291752 |
| 12 | 2.410746 | 0.361170 | 1.396500 | 0.322570 | 0.295990 |
| 13 | 2.379567 | 0.361757 | 1.378529 | 0.315291 | 0.292743 |
| 14 | 2.566726 | 0.489723 | 1.429944 | 0.316631 | 0.295880 |
| 15 | 3.621823 | 0.366820 | 2.477519 | 0.406219 | 0.333301 |
| 16 | 2.380399 | 0.372446 | 1.368591 | 0.314964 | 0.292497 |
| 17 | 2.465494 | 0.363367 | 1.437987 | 0.330423 | 0.301142 |
| 18 | 2.349387 | 0.364450 | 1.351295 | 0.311753 | 0.290640 |
| 19 | 2.471604 | 0.363510 | 1.445543 | 0.329213 | 0.300976 |
| 20 | 2.528384 | 0.494514 | 1.393512 | 0.315481 | 0.292946 |
| 21 | 2.337955 | 0.356362 | 1.346879 | 0.310549 | 0.291398 |
| 22 | 2.348002 | 0.361624 | 1.348540 | 0.312050 | 0.291770 |
| 23 | 2.339648 | 0.362133 | 1.348436 | 0.308821 | 0.288734 |
