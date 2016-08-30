
function [dataset, fileinfo] = readQcodes(filename, verbose)
%% Read codes dataset
%
% Arguments:
%   filename (string): HDF5 dataset to read
%   verbose (int): verbosity level
%
% Output:
%   dataset (struct): data in Matlab format
%   fileinfo (struct): full HDF5 file info
%
% More data can be read from the file using the hdf5read command, e.g.
%
% > data = hdf5read(filename,'/Data Arrays/gates_L_set');

%
% See also: hdf5read
%
if nargin<2
    verbose=1;
end
fileinfo = hdf5info(filename);
dataset=struct();

didx=-1;
for ii =1: length( fileinfo.GroupHierarchy.Groups )
    nm=[fileinfo.GroupHierarchy.Groups(ii).Name];
    if verbose
        fprintf('HDF5: main group %s\n', nm);
    end
    
    if strcmp(fileinfo.GroupHierarchy.Groups(ii).Name, '/Data Arrays')
        didx=ii;
    end
end
%fprintf(v)

dd=fileinfo.GroupHierarchy.Groups(didx).Datasets;
for ii=1:length(dd)
    nm=dd(ii).Name;
    if verbose
        fprintf('HDF5: data array %s\n', nm);
    end
    
    dataset.array(ii).label = dd(ii).Attributes(1).Value.Data;
    dataset.array(ii).name=nm;
    
    dataset.array(ii).shape=dd(ii).Dims;
    
    data = hdf5read(filename,dd(ii).Name);
    dataset.array(ii).data=reshape(data, dataset.array(ii).shape );
    %dataset.array(ii).data=data
end

%% Read metadata

dataset.metadata=struct('dummy', 'not implemented yet');

    if verbose
        fprintf('HDF5: completed reading data %s\n', nm);
    end

return

end
