clear;clc;
tic;
% Specify the folder where the files live.
myFolder = 'D:\wingthinks';
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*.csv'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
 targetFolder='D:\wingthinks\2018';
targetFilename = fullfile(targetFolder, baseFileName);
%datafolder='D:\2017';
cd(targetFolder);
%list=dir([datafolder,'\narcotTest320.csv']);
Data=load(fullFileName); 
 cankao=Data(:,1);cankao=cankao';

 data1=Data(:,2);
data1=data1';
 %data11=data1-cankao;
data1_1_detrend.data=cankao;%data11;
N=length(data1_1_detrend.data);
fs=250;
locutoff=1.;
hicutoff=40.;
data1_1_detrend.srate = fs;
data1_1_detrend.trials = 1; 
data1_1_detrend.event = [ ];
data1_1_detrend.pnts = N; 
[data1_2_FIR, com, b] = pop_eegfiltnew(data1_1_detrend,locutoff,hicutoff);

 data2=Data(:,3);
data2=data2';
 %data2=data2-cankao;
data2_1_detrend.data=data1;%data2;
N=length(data2_1_detrend.data);

data2_1_detrend.srate = fs; 
data2_1_detrend.trials = 1; 
data2_1_detrend.event = [ ]; 
data2_1_detrend.pnts = N;
[data2_2_FIR, com, b] = pop_eegfiltnew(data2_1_detrend,locutoff,hicutoff);

 %data3=Data(:,4);
%data3=data3';
 %data3=data3-cankao;
data3_1_detrend.data=data2;%data3;
N=length(data3_1_detrend.data);

data3_1_detrend.srate = fs; 
data3_1_detrend.trials = 1; 
data3_1_detrend.event = [ ]; 
data3_1_detrend.pnts = N; 
[data3_2_FIR, com, b] = pop_eegfiltnew(data3_1_detrend,locutoff,hicutoff);

fileID = fopen(targetFilename, 'w');
fprintf(fileID, '%f,%f,%f\n', data1_2_FIR.data, data2_2_FIR.data, data3_2_FIR.data);
fclose(fileID);
end