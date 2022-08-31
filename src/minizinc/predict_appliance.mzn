include "increasing.mzn";


array[int] of int: pred_powers;
array[int] of int: mains_powers;
array[real_states, 1..2] of int: pow_base_limits;
array[int] of int: pow_multiplier;
array[real_states, 1..2] of int: states_limits;

int: idle_mean ; 
array[1..2] of int: seq_weights;

int: REAL_STATES = length(pow_multiplier);
float: POW_BASE = 2;
array[int] of int: pow_values ;%= [round(min(POW_BASE^i,(MAX_POWER + 1)))| i in seq_len];


int: MAX_POWER =  max(max(pred_powers), max(mains_powers));
int: SEQUENCE_LENGHT = length(pred_powers);
int: POW_MULTIPLIER = MAX_POWER ;


set of int: powers =  0..MAX_POWER;
set of int: seq_len = 1..SEQUENCE_LENGHT;
set of int: states =  0..(REAL_STATES+1);
set of int: real_states =  1..REAL_STATES;
set of int: pow_mul_value = 1..POW_MULTIPLIER;
set of int: powbases =	min(pow_base_limits)..max(pow_base_limits);

var bool: no_event;

array[real_states] of var powbases: pow_base;
constraint forall(s in real_states)( (not no_event) -> ( pow_base[s] >= pow_base_limits[s,1] /\ pow_base[s] <= pow_base_limits[s,2]));



array[seq_len] of var states: states_map;
constraint increasing(states_map);
constraint forall(s in real_states)( (not no_event) -> ((count(states_map, s) >= states_limits[s, 1]) /\ (count(states_map, s) <= states_limits[s, 2])  ));

array[states] of var seq_len: change_point;
constraint forall(s in real_states)( (not no_event) -> (states_map[change_point[s]] = s /\ states_map[change_point[s] - 1] = s -1 ));
constraint (not no_event) -> ( states_map[change_point[REAL_STATES + 1]] = REAL_STATES + 1 /\ states_map[change_point[REAL_STATES + 1] - 1] = REAL_STATES );
constraint change_point[0] = 1;

array[real_states, seq_len] of var int: pow_states_dev = array2d(real_states, seq_len, [ if (not no_event) /\ states_map[e] = s then abs(pred_powers[e] -  (pow_multiplier[s] div pow_values[max(1,e - change_point[s] + 1)] + pow_base[s] )) else 0  endif | s in real_states, e in seq_len  ]);

array[seq_len] of var int: outer_dev = [ if (not no_event) /\ ((states_map[e] = 0) \/ (states_map[e] = REAL_STATES+1)) then abs(pred_powers[e]  - idle_mean) else 0  endif | e in seq_len  ];


int: baseline;

array[real_states, seq_len] of var int: baseline_pow_states_dev = array2d(real_states, seq_len, [ if states_map[e] = s then abs(mains_powers[e] - baseline - (pow_multiplier[s] div pow_values[max(1, e - change_point[s] + 1)] + pow_base[s] )) else 0  endif | s in real_states, e in seq_len  ]);
array[seq_len] of var int: baseline_outer_dev = [ if ((states_map[e] = 0) \/ (states_map[e] = REAL_STATES+1)) then abs(mains_powers[e] - baseline  - idle_mean) else 0  endif | e in seq_len  ];


var int: pred_cost = ((sum(pow_states_dev) + sum(outer_dev)) * seq_weights[1]) div (seq_weights[1] + seq_weights[2]);
var int: mains_cost = ((sum(baseline_pow_states_dev) + sum(baseline_outer_dev)) * seq_weights[2]) div (seq_weights[1] + seq_weights[2]);


var int: obj;

int: no_ev_cost;

constraint obj = (not no_event) * (pred_cost + mains_cost ) + no_event * (sum(pred_powers) * no_ev_cost );% - length(pred_powers)* idle_mean);


solve minimize obj;


/*
output ["states_means = \(states_means)\nstates_dev = \(states_dev)\nstates_map = \(states_map)\npred_powers = \(pred_powers)\nexpected_pow_value = \(expected_pow_value)\nreal_pow_value = \(real_pow_value)"] ;

output ["states_means = \(states_means)\nstates_dev = \(states_dev)\nstates_map = \(states_map)\nreal_dev = \(real_dev)\npow_values = \(real_dev)"] ;
*/

