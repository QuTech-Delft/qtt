%% Read qcodes HDF5 dataset
addpath('d:\users\eendebakpt\qtt\tools')


xdir='C:\Users\tud205521\tmp\qdata\2016-08-30\13-41-33' 
ll=dir([xdir,'\*hdf5']);
fprintf('%s\n' , ll(1).name);
filename=fullfile(xdir, ll(1).name);

[dataset, fileinfo]=readQcodes(filename);

fprintf('\n');
disp(dataset);

%%

