include "increasing.mzn";
include "at_least.mzn";

int: SEQUENCES;		
int: SEQUENCE_LENGHT;
int: MAX_STATES;

set of int: seq_len = 1..SEQUENCE_LENGHT;
set of int: sequences = 1..SEQUENCES;

array[sequences, seq_len] of int: app_powers;

array[int] of int: pow_values ;

% weights for cost function
array[1..5] of int: cost_weights;

float: POW_BASE = 2;
int: MAX_POWER =  max(app_powers);
int: POW_MULTIPLIER = MAX_POWER ;


set of int: powers =  0..MAX_POWER;
set of int: states=  0..(MAX_STATES+1);
set of int: real_states =  1..MAX_STATES;
set of int: pow_mul_value = 1..POW_MULTIPLIER;








array[sequences, seq_len] of var states: states_map;
constraint forall(q in sequences) (increasing(row(states_map,q)));
constraint forall(q in sequences, s in real_states)(count(row(states_map, q), s) > 0);

array[sequences,states] of var seq_len: change_point;
constraint forall(q in sequences, s in real_states)( states_map[q,change_point[q,s]] = s /\ states_map[q,change_point[q,s] - 1] = s -1 );
constraint forall(q in sequences) (states_map[q, change_point[q, MAX_STATES + 1]] = MAX_STATES + 1 /\ states_map[q, change_point[q, MAX_STATES + 1] - 1] = MAX_STATES );
constraint forall(q in sequences) (change_point[q, 0] = 1);



array[sequences, real_states] of var powers: pow_base;
array[sequences, real_states] of var pow_mul_value: pow_multiplier;

constraint forall(q in sequences, s in real_states) (if pow_multiplier[q,s] < pow_values[0] then pow_multiplier[q,s] = 0 endif) ;

array[sequences, real_states, seq_len] of var int: pow_states_dev = array3d(sequences, real_states, seq_len, [ if states_map[q, e] = s then abs(app_powers[q, e] -  (pow_multiplier[q, s] div pow_values[e - change_point[q, s] + 1] + pow_base[q, s] )) else 0  endif | q in sequences, s in real_states, e in seq_len  ]);

array[real_states, 1..2] of var powers: pow_base_limits;
constraint forall(s in real_states)(pow_base_limits[s, 1] = min(col(pow_base, s)));
constraint forall(s in real_states)(pow_base_limits[s, 2] = max(col(pow_base, s)));
array[real_states] of var powers: pow_base_diff = [pow_base_limits[s, 2] - pow_base_limits[s, 1] | s in real_states];

array[real_states, 1..2] of var powers: pow_multiplier_limits;
constraint forall(s in real_states)(pow_multiplier_limits[s, 1] = min(col(pow_multiplier, s)));
constraint forall(s in real_states)(pow_multiplier_limits[s, 2] = max(col(pow_multiplier, s)));
array[real_states] of var powers: pow_multiplier_diff = [pow_multiplier_limits[s, 2] - pow_multiplier_limits[s, 1] | s in real_states];

array[sequences,real_states] of var seq_len: counter = array2d(sequences, real_states,[count(row(states_map, q), s) |q in sequences, s in real_states ]);
array[real_states, 1..2] of var powers: counter_limits;
constraint forall(s in real_states)(counter_limits[s, 1] = min(col(counter, s)));
constraint forall(s in real_states)(counter_limits[s, 2] = max(col(counter, s)));
array[real_states] of var powers: counter_diff = [counter_limits[s, 2] - counter_limits[s, 1] | s in real_states];

var powers: baseline;
array[sequences, seq_len] of var int: baseline_outer_dev = array2d(sequences, seq_len, [ if ((states_map[q, e] = 0) \/ (states_map[q, e] = MAX_STATES+1)) then abs(app_powers[q, e] - baseline) else 0  endif | q in sequences, e in seq_len  ]);

solve minimize cost_weights[1]*sum(pow_states_dev) + cost_weights[2]*sum(baseline_outer_dev) + cost_weights[3]*sum(counter_diff) + cost_weights[4]*sum(pow_multiplier_diff) + cost_weights[5]*sum(pow_base_diff);

