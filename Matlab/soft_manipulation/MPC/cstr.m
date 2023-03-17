A = [-0.0285 -0.0014; -0.0371 -0.1476];
B = [-0.0850 0.0238; 0.0802 0.4462];
C = [0 1; 1 0];
D = zeros(2,2);
CSTR = ss(A,B,C,D);

CSTR.InputName = {'T_c','C_A_i'};
CSTR.OutputName = {'T','C_A'};
CSTR.StateName = {'C_A','T'};
CSTR.InputGroup.MV = 1;
CSTR.InputGroup.UD = 2;
CSTR.OutputGroup.MO = 1;
CSTR.OutputGroup.UO = 2;

old_status = mpcverbosity('off');

Ts = 1;
MPCobj = mpc(CSTR,Ts)

MPCobj.PredictionHorizon = 15;

MPCobj.MV.Min = -10;
MPCobj.MV.Max = 10;
MPCobj.MV.RateMin = -3;
MPCobj.MV.RateMax = 3;

MPCobj.W.ManipulatedVariablesRate = 0.3;
MPCobj.W.OutputVariables = [1 0];

T=26;
r = [0 0; 2 0];
sim(MPCobj, T, r)