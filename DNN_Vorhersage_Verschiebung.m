%% Einladen der Ansys CSV Tabelle


%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 21, "Encoding", "UTF-8");

% Specify range and delimiter
opts.DataLines = [8, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9", "VarName10", "VarName11", "VarName12", "VarName13", "VarName14", "VarName15", "VarName16", "VarName17", "VarName18", "VarName19", "VarName20", "VarName21"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";

% Specify variable properties
opts = setvaropts(opts, "VarName1", "TrimNonNumeric", true);
opts = setvaropts(opts, "VarName1", "ThousandsSeparator", ",");

% Import the data
Datensatz3400DP = readtable("D:\MATLAB\Neuronales Netz\Datensatz_3_400DP.csv", opts);

%% Convert to output type
Datensatz3400DP = table2array(Datensatz3400DP);

%% Clear temporary variables
clear opts





M_Dehnung_1 = Datensatz3400DP;
x_1 = sortrows(M_Dehnung_1,1,'ascend');

%M für Minus und P für Plus nur für die erste Zeile zur kontrolle
%Kanal 1 Oben X-Richtung
D_OPX = (x_1(1:end,10))*10^6;
D_OMX = (x_1(1:end,11))*10^6;
%Kanal 2 Oben Y-Richtung
D_OPY = (x_1(1:end,12))*10^6;         
D_OMY = (x_1(1:end,13))*10^6;
%Kanal 3 Unten X-Richtung
D_UPX = (x_1(1:end,14))*10^6;
D_UMX = (x_1(1:end,15))*10^6;
%Kanal 4 Unten Y-Richtung
D_UPY = (x_1(1:end,16))*10^6;
D_UMY = (x_1(1:end,17))*10^6;
%Kanal 5 Unten Y-Richtung
D_MPX = (x_1(1:end,18))*10^6;
D_MMX = (x_1(1:end,19))*10^6;
%Kanal 6 Unten Y-Richtung
D_MPY = (x_1(1:end,20))*10^6;
D_MMY = (x_1(1:end,21))*10^6;

%%%%%%Verschiebungen%%%%%%%%%%%%%%%%%%
%Oben X-Richtung
V_OX = x_1(1:end,3);
%Oben Y-Richtung
V_OY = x_1(1:end,5); 
%Unten X-Richtung
V_UX = x_1(1:end,6);
V_UX1 = x_1(1:end,7);
%Unten Y-Richtung
V_UY = x_1(1:end,8);
V_UY1 = x_1(1:end,9);


%%%%%%Biegebelastung jedes Abschnitts
B_OX = (D_OPX-D_OMX)/2;
B_MX = (D_MPX-D_MMX)/2;
B_UX = (D_UPX-D_UMX)/2;

B_OY = (D_OPY-D_OMY)/2;
B_MY = (D_MPY-D_MMY)/2;
B_UY = (D_UPY-D_UMY)/2;

%%%%%Vereinfachung
B_Y_Neu_R = (D_OPY-D_UMY)/2;
B_X_Neu_R = (D_OPX-D_UMX)/2;


%%%%%Maximale Biegebelastung
B_OM = sqrt(B_OX.^(2) + B_OY.^(2));
%B_MM = sqrt(B_MX.^(2) + B_MY.^(2));
B_UM = sqrt(B_UX.^(2) + B_UY.^(2));


%%%%%%Richtung der maximalen Biegebelastung sollte am Ende null werden
R_OM = atan(B_OX/B_OY);
%R_MM = atan(B_MX/B_MY);
R_UM = atan(B_UX/B_UY);

a123 = [B_OX, B_OY, B_MX, B_MY, B_UX, B_UY, V_UX];



%% Anzahl von Parametern 
numDehnung = 6;
numVerschiebung = 1;

%% Trainingsdaten definieren
x = a123(:, 1:numDehnung);         %Eingangsparameter Dehnung für das Training
y = a123(:, numDehnung + 1:end);   %Ausgangsparameter Verschiebung  für das Training

%% Erstellung der Validierungsdaten (Eventuell mehr nutzen)
%XValidation = a123(end-Zeile_T:end, 1:numDehnung);
%YValidation = a123(end-Zeile_T:end, numDehnung + 1:end);

%% Schichtenarchitektur definieren
%https://towardsdatascience.com/training-neural-networks-for-price-prediction-with-tensorflow-8aafe0c55198
%Durch das Anwenden von Dropout wird in jeder Epoche während des Trainings zufällig ein Teil der Neuronen in einer Schicht fallen gelassen, 
%was die verbleibenden Neuronen dazu zwingt, vielseitiger zu sein - dies verringert das Overfitting, da ein Neuron nicht mehr eine bestimmte 
%Instanz abbilden kann, da es während des Trainings nicht immer vorhanden sein wird.
optVars.AnzahlNeuronen1 = 247;
optVars.AnzahlNeuronen2 = 100;
optVars.AnzahlNeuronen3 = 344;

layers = [
    featureInputLayer(numDehnung,"Name","featureinput","Normalization","rescale-symmetric")
                fullyConnectedLayer(optVars.AnzahlNeuronen1, "Name", "myFullyConnectedLayer1")
                tanhLayer("Name","tanh_0")
                scalingLayer("Name","scaling_1","Scale",1.25)
                fullyConnectedLayer(optVars.AnzahlNeuronen2, "Name", "myFullyConnectedLayer2")
                fullyConnectedLayer(optVars.AnzahlNeuronen3, "Name", "myFullyConnectedLayer3")
          fullyConnectedLayer(numVerschiebung, "Name", "myFullyConnectedLayer6")
    regressionLayer("Name","regressionoutput")
    ];
%analyzeNetwork(layers)


%% Optionen des Trainings definieren
% Anzahl der Epochen eingeben
anzahlEpoche = 1781;
anfaenglicheLernrate = 0.010052 ; 
miniBatchSize = 128;

opts = trainingOptions('adam', ...
    'MaxEpochs', anzahlEpoche, ...
    'InitialLearnRate', anfaenglicheLernrate,...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);



%% Neurales Netzwek trainieren
[trainiertesNetzwerk, info] = trainNetwork(x, y, layers, opts);

%% Verschiebung von Dehnung voraussagen
DehnungIST = [-165.070258009183 0.0408137033058864 -161.315124372419 0.0398370454247239 -157.408731226767 0.0389030402486873];
VerschiebungSoll = predict(trainiertesNetzwerk, DehnungIST)




