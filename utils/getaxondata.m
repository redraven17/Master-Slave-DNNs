%
% GETAXONDATA Loads data from Axon file.
%
% [header time data] = getaxondata(filename (optional), pathname(optional))
%
% Author Marina Gobrial
%
% This function is used to read data from an Axon file. Axon files are
% generated from AxoScope, and the files used for this function can either
% be axon text files or axon binary files (.atf or .abf respectively). This
% function throws an exception if it cannot obtain data (for example, if no
% file is selected by the user or if the selected file cannot be opened).
%
% Inputs
%    filename (optional input): String which must end with '.atf' or '.abf'
%           - If not provided, a GUI opens so the user can browse for the
%             file
%    pathname (optional input, but must come after filename if given):
%             String which contains the path to the folder where file is
%             located (must end with the folder name)
%           - If not provided, the default is to assume that the file with 
%             the filename provided is located in the current directory
%
% Outputs
%    header:  Contains header information in a struct with the following
%             fields:
%                   -SignalsExported:   Vector of cells, each with a string
%                   -SignalInformation: Vector of cells, each with a string
%                                       (column headings and/or units)
%                   -NumOfChannels:     Integer with the number of signals
%                   -SamplingFreq:      Sampling frequency of signal (Hz)
%                   -FileName:      	String containing the filename
%                   -Other:             Struct containing extra information 
%                                       This struct may contain different 
%                                       information depending on whether 
%                                       the source file was a text file or  
%                                       a binary file.
%     time:   Times corresponding to each sample (in seconds)
%     data:   Matrix of doubles with each column corresponding to a
%             different signal
%
% Example Usage
%     [header time data] = getaxondata() opens a GUI to allow the user to 
%     browse for the data file (file extension must be .atf or .abf).
%
%     [header time data] = getaxondata('L33R1.atf') returns the data for
%     the text file 'L33R1.atf', located in the current directory.
%
%     [header time data] = getaxondata('L33R1.abf','C:\EMG\Data') returns
%     the data for the binary file 'L33R1.abf' located in 'C:\EMG\Data'.
%
% References
%     http://www.mathworks.com/matlabcentral/fileexchange/22114
%     File Name (to search for on Matlab Central): abf2load
%     Author: Forrest Collman
%     Updated July 28, 2009; Obtained April 27th, 2011
%
% Modifications
% 11/03/05 MG First created.
%

function [header time data] = getaxondata(varargin)

    switch nargin
    %Determine filename and pathname (as provided by user or use default
    %settings)    
    
        case 0 
                %If filename or pathname is given, open a GUI which allows
                %user to browse for the file
                [FileName, PathName, ] = uigetfile({...
                         '*.atf;*.abf', 'All Axon Files (*.atf. *.abf)';...
                         '*.atf','Axon Text Files (*.atf)';...
                         '*.abf','Axon Binary Files (*.abf)'}, ...
                         'Please select EMG data file');
                
                %Check if user pressed cancel
                if isequal(FileName,0) || isequal(PathName,0)
                        exception = MException(...
                                        'OpenFileError:NoSelectedFile',...
                                        'No file has been selected.');
                        throw(exception);
                end

        case 1 
                %If filename is given without a pathname, assume the file
                %is located in the current directory
                PathName = pwd;
                FileName = varargin{1};             
            
        case 2
                %If both filename and pathname are given, use those 
                PathName = varargin{2};
                FileName = varargin{1};  
                
        otherwise
            
                return;
    end
    
    %Determine if the file is a text or binary file from the filename
    %extension
    FileExt     =   FileName(end-2:end);    
    
    %Set filepath to be used when opening file
    FilePath    =   [PathName '\' FileName];

    %Read raw data from file (raw is a variable containing 1 cell with all
    %the data as characters)
    switch  FileExt
        case 'abf' %abf = Axon Binary File
            try
                [header time data]  =   readBinaryFile(FilePath);            
            catch exception
                rethrow(exception)
            end
        case 'atf' %atf = Axon Text File
            try
                [header time data]  =   readTextFile(FilePath);
            catch exception
                rethrow(exception)
            end
        otherwise %Open error dialog and exit program
            exception = MException('OpenFileError:InvalidExtension',...
                ['The file must be an Axon Text File (*.atf) '...
                 'or Axon Binary File (*.abf).']);
            throw(exception);
    end
    
    %Add filename to the header information
    header.FileName = FileName;
    
    clear FileName PathName FileExt FilePath

function [header time data] = readTextFile(FilePath)
%This function is used to read data from an axon text file. The argument
%'FilePath' must contain the file name (which needs to end with '.atf'),
%and the filepath (i.e. directory) if it's not in the current directory.
%The output arguments for this function are as follows:
%   --> header: Contains header information in a struct with the following
%               fields:
%                   -SignalsExported:   Vector of cells, each with a string
%                   -SignalInformation: Vector of cells, each with a string
%                                       (column headings and/or units)
%                   -NumOfChannels:     Integer with the number of signals
%                   -SamplingFreq:      Sampling frequency of signal (Hz)
%                   -Other:             Struct containing extra information
%                                       This struct may contain different 
%                                       information depending on whether 
%                                       the source file was a text file or  
%                                       a binary file.
%   --> time: Times corresponding to each sample (in seconds)
%   --> data: Matrix of doubles with each column corresponding to a
%             different signal

    %Open text file
    [fileID,msg] = fopen(FilePath);

    %Throw an exception if file cannot be opened
    if fileID == -1
        exception = MException('OpenFileError:CannotOpenFile',...
                              ['Cannot open selected file. ' msg]);
        throw(exception);
    end
    
    clear msg
    
    %Scan for header information only
    raw     =   textscan(fileID,'%s',10, 'Delimiter', '\r');
    h       =   raw{1}(1:10); %Obtain the header information
    
    clear raw
    
    %Scan for data (automatically starts scanning right after header
    %information)
    d       =   textscan(fileID,'%f', 'Delimiter', '\t');    
    
    %Close text file
    fclose(fileID);
    
    clear fileID
    
    %Format raw data into a header struct and data matrix
    [header time data] = formatTextData(h,d{1});
    
    clear h d

function [header time data] = formatTextData(h,d)
%This function takes raw data (from an axon file) and formats it, putting
%header information into a struct and signal data into a matrix of doubles.
%The input arguments are as follows:
%   --> h: Must be a cell of size (10,1) which contains the header
%          information in the following manner:
%             -h(1:2): Extra information
%             -h(3): Acquisition Mode
%             -h(4): Comments
%             -h(5): Top values of y axis
%             -h(6): Bottom values of y axis
%             -h(7): Sweep Start Time MS
%             -h(8): Same information as h(9) - This is ignored
%             -h(9): Signals Exported
%             -h(10): Column Headings (there should be one more heading 
%                     than the number of signals exported since one of the
%                     columnts should be time).
%   --> d: Must be a vector containing all the data. The vector length must
%          be a multiple of the number of column headings.
%The output arguments for this function are as follows:
%   --> header: Contains header information in a struct with the following
%               fields:
%                   -SignalsExported:   Vector of cells, each with a string
%                   -SignalInformation: Vector of cells, each with a string
%                                       (column headings and/or units)
%                   -NumOfChannels:     Integer with the number of signals
%                   -SamplingFreq:      Sampling frequency of signal (Hz)
%                   -Other:             Struct containing extra information 
%                                       This struct may contain different 
%                                       information depending on whether 
%                                       the source file was a text file or  
%                                       a binary file.
%   --> time: Times corresponding to each sample (in seconds)
%   --> data: Matrix of doubles with each column corresponding to a
%             different signal
%
    
    %Extract header information
    rawHeader.source            =   'text';
    rawHeader.other             =   h(1:2);
    rawHeader.AcquisitionMode   =   h{3}(18:end-1);
    rawHeader.Comment           =   h{4}(10:end-1);
    rawHeader.SweepStartTimesMS =   str2num(h{7}(20:end-1));  %#ok<ST2NM>

    ytop        =   textscan(h{5}(7:end-1),'%f','Delimiter',',');
    ybottom     =   textscan(h{6}(10:end-1),'%f','Delimiter',',');
    signals     =   textscan(h{9},'%s','Delimiter','"\t"',...
                                       'MultipleDelimsAsOne',true);
    columns     =   textscan(h{10},'%s','Delimiter','"\t"',...
                                        'MultipleDelimsAsOne',true);
    
    rawHeader.YTop              =   ytop{1};
    rawHeader.YBottom           =   ybottom{1};
    
    clear ytop ybottom h
    
    %Assign header information to appropriate struct fields; Keep important
    %information in separate fields and put all other information together
    %in field called "Other"
    header.SignalInformation    =   columns{1}(2:end); %Column one is time
    header.SignalsExported      =   signals{1}(2:end); %Column one is time
    header.NumOfChannels        =   length(header.SignalInformation);
    header.Other                =   rawHeader;
    
    clear rawHeader columns signals
    
    %Separate data into columnts each corresponding to a different signal
    d       =   d(:); %Ensure input is a vertical vector
    col     =   1 + header.NumOfChannels; %Number of signals + time vector
    row     =   length(d)/col; %Number of data points for each signal
    d       =   reshape(d,col,row).';
    time    =   d(:,1);
    data    =   d(:,2:end);
    
    %Add sampling frequency to header struct
    sampleInterval          =   time(2)-time(1);
    header.SamplingFreq     =   1/sampleInterval;
    
    clear sampleInterval d row col

function [header time data] = readBinaryFile(FilePath)
%This function is used to read data from an axon binary file. The argument
%'FilePath' must contain the file name (which needs to end with '.abf'),
%and the filepath (i.e. directory) if it's not in the current directory.
%The output arguments for this function are as follows:
%   --> header: Contains header information in a struct with the following
%               fields:
%                   -SignalsExported:   Vector of cells, each with a string
%                   -SignalInformation: Vector of cells, each with a string
%                                       (column headings and/or units)
%                   -NumOfChannels:     Integer with the number of signals
%                   -SamplingFreq:      Sampling frequency of signal (Hz)
%                   -Other:             Struct containing extra information 
%                                       This struct may contain different 
%                                       information depending on whether 
%                                       the source file was a text file or  
%                                       a binary file.
%   --> time: Times corresponding to each sample (in seconds)
%   --> data: Matrix of doubles with each column corresponding to a
%             different signal

    %This function was obtained from Matlab Central's File Exchange site:
    %http://www.mathworks.com/matlabcentral/fileexchange/22114
    %File Name (to search for on Matlab Central): abf2load
    %Author: Forrest Collman
    %Updated July 28, 2009; Obtained April 27th, 2011
    %Outputs: data (matrix); sample interval (in microseconds); 
    %h = header information (struct)
    try
        [data,sampleInterval,rawHeader]=abfload(FilePath);
    catch exception
        rethrow(exception)
    end
    
    %Convert sample Interval from microseconds to seconds
    sampleInterval = sampleInterval*0.000001; 
    
    %Create a time vector for this data
    time = 0:sampleInterval:sampleInterval*(rawHeader.dataPtsPerChan-1);
    
    %Format header data - Keep important information in separate fields and
    %put all other information together in field called "Other"
    header.SignalsExported      =   rawHeader.recChNames;
    header.SignalInformation    =   rawHeader.recChUnits;
    header.NumOfChannels        =   rawHeader.nADCNumChannels;
    header.SamplingFreq         =   1/sampleInterval;
    %Remove redundant fields in header struct
    rawHeader                   =   rmfield(rawHeader,{'recChUnits';...
                                     'recChNames';'nADCNumChannels';'si'});
    header.Other                =   rawHeader;
    
    clear rawHeader sampleInterval

%These functions were obtained from Matlab Central's File Exchange site:
%http://www.mathworks.com/matlabcentral/fileexchange/22114
%File Name (to search for on Matlab Central): abf2load
%Author: Forrest Collman
%Updated July 28, 2009; Obtained April 27th, 2011
%Outputs: data (matrix); sample interval (in microseconds); 
%h = header information (struct)

%The functions were modified slightly for the purposes of this program

function [d,si,h]=abfload(fn,varargin)
% ** function [d,si,h]=abfload(fn,varargin)
% loads and returns data in ABF (Axon Binary File) format.
% Data may have been acquired in the following modes:
% (1) event-driven variable-length (currently only abf versions < 2.0)
% (2) event-driven fixed-length or waveform-fixed length
% (3) gap-free
% Information about scaling, the time base and the number of channels and 
% episodes is extracted from the header of the abf file.
%
% OPERATION
% If the second input variable is the char array 'info' as in 
%         [d,si,h]=abfload('d:\data01.abf','info') 
% abfload will not load any data but return detailed information (header
% parameters) on the file in output variable h. d and si will be empty.
% In all other cases abfload will load data. Optional input parameters
% listed below (= all except the file name) must be specified as
% parameter/value pairs, e.g. as in 
%         d=abfload('d:\data01.abf','start',100,'stop','e');
%
% >>> INPUT VARIABLES >>>
% NAME        TYPE, DEFAULT      DESCRIPTION
% fn          char array         abf data file name
% start       scalar, 0          only gap-free-data: start of cutout to be 
%                                 read (unit: s)
% stop        scalar or char,    only gap-free-data: end of cutout to be  
%             'e'                 read (unit: sec). May be set to 'e' (end 
%                                 of file).
% sweeps      1d-array or char,  only episodic data: sweep numbers to be 
%             'a'                 read. By default, all sweeps will be read
%                                 ('a').
% channels    cell array         names of channels to be read, like 
%              or char, 'a'       {'IN 0','IN 8'} (make sure spelling is
%                                 100% correct, including blanks). If set 
%                                 to 'a', all channels will be read.
% chunk       scalar, 0.05       only gap-free-data: the elementary chunk  
%                                 size (megabytes) to be used for the 
%                                 'discontinuous' mode of reading data 
%                                 (fewer channels to be read than exist)
% machineF    char array,        the 'machineformat' input parameter of the
%              'ieee-le'          matlab fopen function. 'ieee-le' is the 
%                                 correct option for windows; depending on 
%                                 the platform the data were recorded/shall
%                                 be read by abfload 'ieee-be' is the 
%                                 alternative.
% << OUTPUT VARIABLES <<<
% NAME  TYPE            DESCRIPTION
% d                     the data read, the format depending on the record-
%                        ing mode
%   1. GAP-FREE:
%   2d array        2d array of size 
%                    <data pts> by <number of chans>
%                    Examples of access:
%                    d(:,2)       data from channel 2 at full length
%                    d(1:100,:)   first 100 data points from all channels
%   2. EPISODIC FIXED-LENGTH/WAVEFORM FIXED-LENGTH:
%   3d array        3d array of size 
%                    <data pts per sweep> by <number of chans> by <number 
%                    of sweeps>.
%                    Examples of access:
%                    d(:,2,:)            a matrix containing all episodes 
%                                        (at full length) of the second 
%                                        channel in its columns
%                    d(1:200,:,[1 11])   contains first 200 data points of 
%                                        episodes 1 and 11 of all channels
%   3. EPISODIC VARIABLE-LENGTH:
%   cell array      cell array whose elements correspond to single sweeps. 
%                    Each element is a (regular) array of size
%                    <data pts per sweep> by <number of chans>
%                    Examples of access:
%                    d{1}            a 2d-array which contains episode 1 
%                                    (all of it, all channels)
%                    d{2}(1:100,2)   a 1d-array containing the first 100
%                                    data points of channel 2 in episode 1
% si    scalar           the sampling interval in us
% h     struct           information on file (selected header parameters)
% 
% 
% CONTRIBUTORS
%   Original version by Harald Hentschke(harald.hentschke@uni-tuebingen.de)
%   Extended to abf version 2.0 by Forrest Collman (fcollman@Princeton.edu)
%   pvpmod.m by Ulrich Egert (egert@bccn.uni-freiburg.de)
%   Date of this version: May 20, 2009

% -------------------------------------------------------------------------
%                       PART 1: check of input vars
% -------------------------------------------------------------------------
%disp(['** ' mfilename])
% --- defaults   
% gap-free
start=0.0;
stop='e';
% episodic
sweeps='a';
% general
channels='a';
% the size of data chunks (see above) in Mb. 0.05 Mb is an empirical value
% which works well for abf with 6-16 channels and recording durations of 
% 5-30 min
chunk=0.05;
machineF='ieee-le';
verbose=1;
% if first and only optional input argument is string 'info' the user's
% request is to obtain information on the file (header parameters), so set
% flag accordingly
if nargin==2 && ischar(varargin{1}) && strcmp('info',varargin{1})
 doLoadData=false;
else
 doLoadData=true;
 % assign values of optional input parameters if any were given
 pvpmod(varargin);
end

% some constants
BLOCKSIZE=512;
% output variables
d=[]; 
si=[]; %#ok<NASGU>
h=[];
if ischar(stop)
 if ~strcmpi(stop,'e')
   error('OpenFileError:ErrorInABFLOAD',...
        ['input parameter ''stop'' must be specified as ''e'' '...
         '(=end of recording) or as a scalar']);
 end
end
% check existence of file
if ~exist(fn,'file'), 
 error('OpenFileError:ErrorInABFLOAD',['could not find file ' fn]); 
end

% -------------------------------------------------------------------------
%                       PART 2a: determine abf version
% -------------------------------------------------------------------------
%disp(['opening ' fn '..']); 
[fid,messg]=fopen(fn,'r',machineF); 
if fid == -1,
 error('OpenFileError:ErrorInABFLOAD',messg);
end
% on the occasion, determine absolute file size
fseek(fid,0,'eof');
fileSz=ftell(fid);
fseek(fid,0,'bof');

% *** read value of 'fFileSignature' (i.e. abf version) from header ***
sz=4;
[fFileSignature,n]=fread(fid,sz,'uchar=>char');
if n~=sz,
 fclose(fid);
 error('OpenFileError:ErrorInABFLOAD',...
     'something went wrong reading value(s) for fFileSignature');
end
% rewind
fseek(fid,0,'bof');
% transpose
fFileSignature=fFileSignature';

% -------------------------------------------------------------------------
%    PART 2b: define file information ('header') parameters of interest
% -------------------------------------------------------------------------
% The list of header parameters created below (variable 'headPar') is
% derived from the abf version 1.8 header section. It is by no means 
% exhaustive (i.e. there are many more parameters in abf files) but
% sufficient for proper upload, scaling and arrangement of data acquired
% under many conditons. Further below, these parameters will be made fields
% of struct h. h, which is also an output variable, is then used in PART 3,
% which does the actual job of uploading, scaling and rearranging the data.
% That part of the code relies on h having a certain set of fields
% irrespective of ABF version. 
% Unfortunately, in the transition to ABF version 2.0 many of the header
% parameters were moved to different places within the abf file and/or
% given other names or completely restructured. In order for the code to
% work with pre- and post-2.0 data files, all parameters missing in the
% header must be gotten into h. This is accomplished in lines ~262 and
% following:
%     if h.fFileVersionNumber>=2
%       ...
% Furthermore,
% - h as an output from an ABF version < 2.0 file will not contain new 
%   parameters introduced into the header like 'nCRCEnable'
% - h will in any case contain a few 'home-made' fields that have
%   proven to be useful. Some of them depend on the recording mode. Among
%   the more or less self-explanatory ones are
% -- si                   sampling interval
% -- recChNames           the names of all channels, e.g. 'IN 8',...
% -- dataPtsPerChan       sample points per channel
% -- dataPts              sample points in file
% -- recTime              recording start and stop time in seconds from
%                         midnight (millisecond resolution)
% -- sweepLengthInPts     sample points per sweep (one channel)
% -- sweepStartInPts      the start times of sweeps in sample points 
%                         (from beginning of recording)


% call local function for header proper
headPar=define_header(fFileSignature);
  TagInfo=define_TagInfo;
switch fFileSignature
 case 'ABF ' % ** note the blank
   % ************************
   %     abf version < 2.0
   % ************************
   % do nothing, header already defined above

 case 'ABF2'
   % ************************
   %     abf version >= 2.0
   % ************************
   Sections=define_Sections;
   ProtocolInfo=define_ProtocolInfo;
   ADCInfo=define_ADCInfo;
   
 otherwise
   error('OpenFileError:ErrorInABFLOAD',...
       'unknown or incompatible file signature');
end

% convert headPar to struct
s=cell2struct(headPar,{'name','offs','numType','value'},2);
numOfParams=size(s,1);
clear tmp headPar;

% -------------------------------------------------------------------------
%    PART 2c: read parameters of interest
% -------------------------------------------------------------------------
% convert names in structure to variables and read value from header
for g=1:numOfParams
 if fseek(fid, s(g).offs,'bof')~=0, 
   fclose(fid);
   error('OpenFileError:ErrorInABFLOAD',...
        ['something went wrong locating ' s(g).name]); 
 end
 sz=length(s(g).value);
 % use dynamic field names 
 [h.(s(g).name),n]=fread(fid,sz,s(g).numType);
 if n~=sz, 
   fclose(fid);    
   error('OpenFileError:ErrorInABFLOAD',...
        ['something went wrong reading value(s) for ' s(g).name]); 
 end
end
% transpose
h.fFileSignature=h.fFileSignature';
% several header parameters need a fix or version-specific refinement:
if strcmp(h.fFileSignature,'ABF2')
 % h.fFileVersionNumber needs to be converted from an array of integers to
 % a float
 h.fFileVersionNumber=h.fFileVersionNumber(4)+h.fFileVersionNumber(3)*.1...
   +h.fFileVersionNumber(2)*.001+h.fFileVersionNumber(1)*.0001;
 % convert ms to s
 h.lFileStartTime=h.uFileStartTimeMS*.001;
else
 % h.fFileVersionNumber is a float32 the value of which is sometimes a
 % little less than what it should be (e.g. 1.6499999 instead of 1.65)
 h.fFileVersionNumber=.001*round(h.fFileVersionNumber*1000);
 % in abf < 2.0 two parameters are needed to obtain the file start time
 % with millisecond precision - let's integrate both into parameter
 % lFileStartTime (unit: s) so that nFileStartMillisecs will not be needed
 h.lFileStartTime=h.lFileStartTime+h.nFileStartMillisecs*.001;  
end

% *** read file information that has gone elsewhere in ABF version >= 2.0
% and assign values ***
if h.fFileVersionNumber>=2
 % --- read in the Sections
 Sects=cell2struct(Sections,{'name'},2);
 numOfSections=length(Sections);
 offset=76;
 for i=1:numOfSections
   eval([Sects(i).name '=ReadSectionInfo(fid,offset);']);
   offset=offset+4+4+8;
 end
 % --- read in the Strings
 fseek(fid,StringsSection.uBlockIndex*BLOCKSIZE,'bof');
 BigString=fread(fid,StringsSection.uBytes,'char');
 % this is a hack
 goodstart=strfind(lower(char(BigString)'),'clampex');
 %this extends the hack to deal with axoscope files 
 if isempty(goodstart)
      goodstart=strfind(lower(char(BigString)'),'axoscope');
  end
  
 BigString=BigString(goodstart(1):end)';
 stringends=find(BigString==0);
 stringends=[0 stringends];
 for i=1:length(stringends)-1
   Strings{i}=char(BigString(stringends(i)+1:stringends(i+1)-1));
 end
 h.recChNames=[];
 h.recChUnits=[];

 % --- read in the ADCSection
 for i=1:ADCSection.llNumEntries
   ADCsec(i)                    = ReadSection(fid,...
                                  ADCSection.uBlockIndex...
                                  *BLOCKSIZE+ADCSection.uBytes...
                                  *(i-1),ADCInfo);
   ii                           = ADCsec(i).nADCNum+1;
   h.nADCSamplingSeq(i)         = ADCsec(i).nADCNum;
   h.recChNames                 = strvcat(h.recChNames, ...
                                  Strings{ADCsec(i).lADCChannelNameIndex});
   h.recChUnits                 = strvcat(h.recChUnits, ...
                                  Strings{ADCsec(i).lADCUnitsIndex});
   h.nTelegraphEnable(ii)       = ADCsec(i).nTelegraphEnable;
   h.fTelegraphAdditGain(ii)    = ADCsec(i).fTelegraphAdditGain;
   h.fInstrumentScaleFactor(ii) = ADCsec(i).fInstrumentScaleFactor;
   h.fSignalGain(ii)            = ADCsec(i).fSignalGain;
   h.fADCProgrammableGain(ii)   = ADCsec(i).fADCProgrammableGain;
   h.fInstrumentOffset(ii)      = ADCsec(i).fInstrumentOffset;
   h.fSignalOffset(ii)          = ADCsec(i).fSignalOffset;
 end
 % --- read in the protocol section
 ProtocolSec            =   ReadSection(fid,...
                            ProtocolSection.uBlockIndex*BLOCKSIZE,...
                            ProtocolInfo);
 % --- 
 h.nOperationMode       =   ProtocolSec.nOperationMode;
 h.fSynchTimeUnit       =   ProtocolSec.fSynchTimeUnit;
 h.nADCNumChannels      =   ADCSection.llNumEntries;
 h.lActualAcqLength     =   DataSection.llNumEntries;
 h.lDataSectionPtr      =   DataSection.uBlockIndex;
 h.nNumPointsIgnored    =   0;
 % in ABF version < 2.0 h.fADCSampleInterval is the sampling interval
 % defined as 
 %     1/(sampling freq*number_of_channels)
 % so divide ProtocolSec.fADCSequenceInterval by the number of
 % channels
 h.fADCSampleInterval = ProtocolSec.fADCSequenceInterval/h.nADCNumChannels;
 h.fADCRange          = ProtocolSec.fADCRange;
 h.lADCResolution     = ProtocolSec.lADCResolution;
 h.lSynchArrayPtr     = SynchArraySection.uBlockIndex;
 h.lSynchArraySize    = SynchArraySection.llNumEntries;
else
   TagSection.llNumEntries  =  h.lNumTagEntries;
    TagSection.uBlockIndex  =  h.lTagSectionPtr;
    TagSection.uBytes       =  64;
end
Tagsec=[];
for i=1:TagSection.llNumEntries
        Tagsec(i)           =  ReadSection(fid,TagSection.uBlockIndex...
                               *BLOCKSIZE+TagSection.uBytes*(i-1),TagInfo);
        Tagsec(i).sComment  =  char(Tagsec(i).sComment)';
end
h.Tags=Tagsec;


% -------------------------------------------------------------------------
%    PART 2d: groom parameters & perform some plausibility checks
% -------------------------------------------------------------------------
if h.lActualAcqLength<h.nADCNumChannels, 
 fclose(fid);
 error('OpenFileError:ErrorInABFLOAD',...
       'less data points than sampled channels in file'); 
end
% the numerical value of all recorded channels (numbers 0..15)
recChIdx=h.nADCSamplingSeq(1:h.nADCNumChannels);
% the corresponding indices into loaded data d
recChInd=1:length(recChIdx);
if h.fFileVersionNumber<2
 % the channel names, e.g. 'IN 8' (for ABF version 2.0 these have been
 % extracted above at this point)
 h.recChNames=(reshape(char(h.sADCChannelName),10,16))';
 h.recChNames=h.recChNames(recChIdx+1,:);
 % same with signal units
 h.recChUnits=(reshape(char(h.sADCUnits),8,16))';
 h.recChUnits=h.recChUnits(recChIdx+1,:);
end
% convert to cell arrays 
h.recChNames=deblank(cellstr(h.recChNames));
h.recChUnits=deblank(cellstr(h.recChUnits));

% check whether requested channels exist
chInd=[];
eflag=0;
if ischar(channels)
 if strcmp(channels,'a')
   chInd=recChInd;
 else
   fclose(fid);
   error('OpenFileError:ErrorInABFLOAD',...
        ['input parameter ''channels'' must either be a cell array '...
        'holding channel names or the single character ''a'' '...
        '(=all channels)']);
 end
else
 for i=1:length(channels)
   tmpChInd=strmatch(channels{i},h.recChNames,'exact');
   if ~isempty(tmpChInd)
     chInd=[chInd tmpChInd];
   else
     % set error flag to 1
     eflag=1;
   end
 end
end
if eflag
 fclose(fid);
 %disp('**** available channels:');
 %disp(h.recChNames);
 %disp(' ');
 %disp('**** requested channels:');
 %disp(channels);
 error('OpenFileError:ErrorInABFLOAD',...
      ['at least one of the requested channels does not exist '...
       'in data file (see above)']);
end
% display available channels if in info mode
if ~doLoadData
 %disp('**** available channels:');
 %disp(h.recChNames);
end  

% gain of telegraphed instruments, if any
if h.fFileVersionNumber>=1.65
 addGain=h.nTelegraphEnable.*h.fTelegraphAdditGain;
 addGain(addGain==0)=1;
else
 addGain=ones(size(h.fTelegraphAdditGain));
end

% determine offset at which data start
switch h.nDataFormat
 case 0
   dataSz=2;  % bytes/point
   precision='int16';
 case 1
   dataSz=4;  % bytes/point
   precision='float32';
 otherwise
   fclose(fid);
   error('OpenFileError:ErrorInABFLOAD','invalid number format');
end
headOffset=h.lDataSectionPtr*BLOCKSIZE+h.nNumPointsIgnored*dataSz;
% h.fADCSampleInterval is the TOTAL sampling interval
h.si=h.fADCSampleInterval*h.nADCNumChannels;
% assign same value to si, which is an output variable
si=h.si;
if ischar(sweeps) && sweeps=='a'
 nSweeps=h.lActualEpisodes;
 sweeps=1:h.lActualEpisodes;
else
 nSweeps=length(sweeps);
end  

% -------------------------------------------------------------------------
%    PART 3: read data (note: from here on code is generic and abf version
%    should not matter)
% -------------------------------------------------------------------------
switch h.nOperationMode
 case 1
   %disp('data were acquired in event-driven variable-length mode');
   if h.fFileVersionNumber>=2.0
     errordlg(['abfload currently does not work with data acquired in'...
              ' event-driven variable-length mode and ABF version 2.0'],...
              'ABF version issue');
   else
     if (h.lSynchArrayPtr<=0 || h.lSynchArraySize<=0),
       fclose(fid);
       error('OpenFileError:ErrorInABFLOAD',...
             'internal variables ''lSynchArraynnn'' are zero or negative');
     end
     switch h.fSynchTimeUnit
       case 0 
         %time information in synch array section is in terms of ticks
         h.synchArrTimeBase=1;
       otherwise
         % time information in synch array section is in terms of usec
         h.synchArrTimeBase=h.fSynchTimeUnit;
     end
     % the byte offset at which the SynchArraySection starts
     h.lSynchArrayPtrByte=BLOCKSIZE*h.lSynchArrayPtr;
     % before reading Synch Arr parameters check if file is big enough to 
     % hold them
     % 4 bytes/long, 2 values per episode (start and length)
     if h.lSynchArrayPtrByte+2*4*h.lSynchArraySize<fileSz,
       fclose(fid);
       error('OpenFileError:ErrorInABFLOAD',...
           'file seems not to contain complete Synch Array Section');
     end
     if fseek(fid,h.lSynchArrayPtrByte,'bof')~=0,
       fclose(fid);
       error('OpenFileError:ErrorInABFLOAD',...
            ['something went wrong positioning file pointer to Synch'...
             ' Array Section']);
     end
     [synchArr,n]=fread(fid,h.lSynchArraySize*2,'int32');
     if n~=h.lSynchArraySize*2,
       fclose(fid);
       error('OpenFileError:ErrorInABFLOAD',...
             'something went wrong reading synch array section');
     end
     % make synchArr a h.lSynchArraySize x 2 matrix
     synchArr       =   permute(reshape(synchArr',2,h.lSynchArraySize),...
                            [2 1]);
     % the length of episodes in sample points
     segLengthInPts =   synchArr(:,2)/h.synchArrTimeBase;
     % the starting ticks of episodes in sample points WITHIN THE DATA FILE
     segStartInPts  =   cumsum([0 (segLengthInPts(1:end-1))']*dataSz)...
                            + headOffset;
     % start time (synchArr(:,1)) has to be divided by h.nADCNumChannels 
     % to get true value
     % go to data portion
     if fseek(fid,headOffset,'bof')~=0,
       fclose(fid);
       error('OpenFileError:ErrorInABFLOAD',...
            ['something went wrong positioning file pointer '... 
            '(too few data points ?)']);
     end
     % ** load data if requested
     if doLoadData
       for i=1:nSweeps,
         % if selected sweeps are to be read, seek correct position
         if ~isequal(nSweeps,h.lActualEpisodes),
           fseek(fid,segStartInPts(sweeps(i)),'bof');
         end
         [tmpd,n]=fread(fid,segLengthInPts(sweeps(i)),precision);
         if n~=segLengthInPts(sweeps(i)),
           warning('OpenFileWarning:WarningInABFLOAD',...
                  ['something went wrong reading episode '...
                  int2str(sweeps(i)) ': ' segLengthInPts(sweeps(i)) ...
                  ' points should have been read, ' int2str(n)...
                  ' points actually read']);
         end
         h.dataPtsPerChan=n/h.nADCNumChannels;
         if rem(n,h.nADCNumChannels)>0,
           fclose(fid);
           error('OpenFileError:ErrorInABFLOAD',...
                 'number of data points in episode not OK');
         end
         % separate channels..
         tmpd=reshape(tmpd,h.nADCNumChannels,h.dataPtsPerChan);
         % retain only requested channels
         tmpd=tmpd(chInd,:);
         tmpd=tmpd';
         % if data format is integer, scale appropriately; 
         % if it's float, tmpd is fine
         if ~h.nDataFormat
           for j=1:length(chInd),
             ch         =  recChIdx(chInd(j))+1;
             tmpd(:,j)  =  tmpd(:,j)/(h.fInstrumentScaleFactor(ch)...
                           *h.fSignalGain(ch)*h.fADCProgrammableGain(ch)...
                           *addGain(ch))*h.fADCRange/h.lADCResolution...
                           + h.fInstrumentOffset(ch)-h.fSignalOffset(ch);
           end
         end
         % now place in cell array, an element consisting of one sweep 
         % with channels in columns
         d{i}=tmpd;
       end
     end
   end

 case {2,5}
   if h.nOperationMode==2
     %disp('data were acquired in event-driven fixed-length mode');
   else
     %disp(['data were acquired in waveform fixed-length mode'...
     %     ' (clampex only)']);
   end
   % extract timing information on sweeps
   if (h.lSynchArrayPtr<=0 || h.lSynchArraySize<=0),
     fclose(fid);
     error('OpenFileError:ErrorInABFLOAD',...
           'internal variables ''lSynchArraynnn'' are zero or negative');
   end
   switch h.fSynchTimeUnit
     case 0  % time information in synch array section is in terms of ticks
       h.synchArrTimeBase=1;
     otherwise %time information in synch array section is in terms of usec
       h.synchArrTimeBase=h.fSynchTimeUnit;
   end
   % the byte offset at which the SynchArraySection starts
   h.lSynchArrayPtrByte=BLOCKSIZE*h.lSynchArrayPtr;
   % before reading Synch Arr parameters check if file is big 
   % enough to hold them
   % 4 bytes/long, 2 values per episode (start and length)
   if h.lSynchArrayPtrByte+2*4*h.lSynchArraySize>fileSz,
     fclose(fid);
     error('OpenFileError:ErrorInABFLOAD',...
           'file seems not to contain complete Synch Array Section');
   end
   if fseek(fid,h.lSynchArrayPtrByte,'bof')~=0,
     fclose(fid);
     error('OpenFileError:ErrorInABFLOAD',...
          ['something went wrong positioning file pointer to Synch'...
           ' Array Section']);
   end
   [synchArr,n]=fread(fid,h.lSynchArraySize*2,'int32');
   if n~=h.lSynchArraySize*2,
     fclose(fid);
     error('OpenFileError:ErrorInABFLOAD',...
           'something went wrong reading synch array section');
   end
   % make synchArr a h.lSynchArraySize x 2 matrix
   synchArr=permute(reshape(synchArr',2,h.lSynchArraySize),[2 1]);
   if numel(unique(synchArr(:,2)))>1
     fclose(fid);
     error('OpenFileError:ErrorInABFLOAD',...
         'sweeps of unequal length in file recorded in fixed-length mode');
   end
   % the length of sweeps in sample points (**note: parameter lLength of
   % the ABF synch section is expressed in samples (ticks) whereas
   % parameter lStart is given in synchArrTimeBase units)
   h.sweepLengthInPts = synchArr(1,2)/h.nADCNumChannels;
   % the starting ticks of episodes in sample points (t0=1=beginning of
   % recording)
   h.sweepStartInPts  = synchArr(:,1)*(h.synchArrTimeBase/...
                                h.fADCSampleInterval/h.nADCNumChannels);
   % recording start and stop times in seconds from midnight
   h.recTime	= h.lFileStartTime;
   h.recTime  	= h.recTime + [0  ...
                  (1e-6*(h.sweepStartInPts(end)+h.sweepLengthInPts))...
                  *h.fADCSampleInterval*h.nADCNumChannels];
   % determine first point and number of points to be read 
   startPt=0;
   h.dataPts=h.lActualAcqLength;
   h.dataPtsPerChan=h.dataPts/h.nADCNumChannels;
   if rem(h.dataPts,h.nADCNumChannels)>0 ...
                               || rem(h.dataPtsPerChan,h.lActualEpisodes)>0
     fclose(fid);
     error('OpenFileError:ErrorInABFLOAD','number of data points not OK'); 
   end
   % temporary helper var
   dataPtsPerSweep=h.sweepLengthInPts*h.nADCNumChannels;
   if fseek(fid,startPt*dataSz+headOffset,'bof')~=0
     fclose(fid);
     error('OpenFileError:ErrorInABFLOAD',...
          ['something went wrong positioning file pointer'...
          ' (too few data points ?)']); 
   end
   d=zeros(h.sweepLengthInPts,length(chInd),nSweeps);
   % the starting ticks of episodes in sample points WITHIN THE DATA FILE
   selectedSegStartInPts=((sweeps-1)*dataPtsPerSweep)*dataSz+headOffset;
   % ** load data if requested
   if doLoadData
     for i=1:nSweeps,
       fseek(fid,selectedSegStartInPts(i),'bof');
       [tmpd,n]=fread(fid,dataPtsPerSweep,precision);
       if n~=dataPtsPerSweep,
         fclose(fid);
         error('OpenFileError:ErrorInABFLOAD',...
             ['something went wrong reading episode ' int2str(sweeps(i))...
             ': ' dataPtsPerSweep ' points should have been read, ' ...
             int2str(n) ' points actually read']);
       end
       h.dataPtsPerChan=n/h.nADCNumChannels;
       if rem(n,h.nADCNumChannels)>0
         fclose(fid);
         error('OpenFileError:ErrorInABFLOAD',...
               'number of data points in episode not OK');
       end
       % separate channels..
       tmpd=reshape(tmpd,h.nADCNumChannels,h.dataPtsPerChan);
       % retain only requested channels
       tmpd=tmpd(chInd,:);
       tmpd=tmpd';
       % if data format is integer, scale appropriately; 
       % if it's float, d is fine
       if ~h.nDataFormat
         for j=1:length(chInd),
           ch        = recChIdx(chInd(j))+1;
           tmpd(:,j) = tmpd(:,j)/(h.fInstrumentScaleFactor(ch)...
                       *h.fSignalGain(ch)*h.fADCProgrammableGain(ch)...
                       *addGain(ch))*h.fADCRange/h.lADCResolution...
                       +h.fInstrumentOffset(ch)-h.fSignalOffset(ch);
         end
       end
       % now fill 3d array
       d(:,:,i)     =  tmpd;
     end
   end

 case 3
   %disp('data were acquired in gap-free mode');
   % from start, stop, headOffset and h.fADCSampleInterval calculate   
   % first point to be read 
   %  and - unless stop is given as 'e' - number of points
   startPt=floor(1e6*start*(1/h.fADCSampleInterval));
   % this corrects undesired shifts in the reading frame due to rounding 
   % errors in the previous calculation
   startPt=floor(startPt/h.nADCNumChannels)*h.nADCNumChannels;
   % if stop is a char array, it can only be 'e' at this point
   %  (other values would have been caught above)
   if ischar(stop),
     h.dataPtsPerChan   =   h.lActualAcqLength/h.nADCNumChannels...
                                -floor(1e6*start/h.si);
     h.dataPts          =   h.dataPtsPerChan*h.nADCNumChannels;
   else
     h.dataPtsPerChan   =   floor(1e6*(stop-start)*(1/h.si));
     h.dataPts          =   h.dataPtsPerChan*h.nADCNumChannels;
     if h.dataPts<=0 
       fclose(fid);
       error('OpenFileError:ErrorInABFLOAD',...
             'start is larger than or equal to stop'); 
     end
   end
   if rem(h.dataPts,h.nADCNumChannels)>0
     fclose(fid);
     error('OpenFileError:ErrorInABFLOAD','number of data points not OK'); 
   end
   tmp=1e-6*h.lActualAcqLength*h.fADCSampleInterval;
   if verbose
     %disp(['total length of recording: ' num2str(tmp,'%5.1f')...
     %       ' s ~ ' num2str(tmp/60,'%3.0f') ' min']);
     % 8 bytes per data point expressed in Mb
     %disp(['memory requirement for complete upload in matlab: '...
     %  num2str(round(8*h.lActualAcqLength/2^20)) ' MB']);
   end
   % recording start and stop times in seconds from midnight
   h.recTime=h.lFileStartTime;
   h.recTime=[h.recTime h.recTime+tmp];
   if fseek(fid,startPt*dataSz+headOffset,'bof')~=0, 
     fclose(fid);
     error('OpenFileError:ErrorInABFLOAD',...
          ['something went wrong positioning file pointer '...
           '(too few data points ?)']);
   end
   if doLoadData
     %*** decide on the most efficient way to read data:
     %(i) all (of one or several) channels requested: read, done
     %(ii) one (of several) channels requested: use the 'skip' feature of
     %fread
     %(iii) more than one but not all (of several) channels requested:
     %'discontinuous' mode of reading data. Read a reasonable chunk of data
     %(all channels), separate channels, discard non-requested ones (if
     %any), place data in preallocated array, repeat until done. This is
     %faster than reading the data in one big lump, separating channels and
     %discarding the ones not requested
     if length(chInd)==1 && h.nADCNumChannels>1
       % --- situation (ii)
       % jump to proper reading frame position in file
       if fseek(fid,(chInd-1)*dataSz,'cof')~=0
         fclose(fid);
         error('OpenFileError:ErrorInABFLOAD',...
             ['something went wrong positioning file pointer '...
              '(too few data points ?)']);
       end
       % read, skipping h.nADCNumChannels-1 data points after each read
       [d,n]    =   fread(fid,h.dataPtsPerChan,precision,...
                            dataSz*(h.nADCNumChannels-1));
       if n~=h.dataPtsPerChan,
         fclose(fid);
         error('OpenFileError:ErrorInABFLOAD',...
             ['something went wrong reading file (' ...
             int2str(h.dataPtsPerChan) ' points should have been read, '...
             int2str(n) ' points actually read']);
       end
     elseif length(chInd)/h.nADCNumChannels<1
       % --- situation (iii)
       % prepare chunkwise upload:
       % preallocate d
       d=repmat(nan,h.dataPtsPerChan,length(chInd));
       % the number of data points corresponding to the maximal chunk size,
       % rounded off such that from each channel the same number of points
       % is read (do not forget that each data point will by default be
       % made a double of 8 bytes, no matter what the original data format)
       chunkPtsPerChan=floor(chunk*2^20/8/h.nADCNumChannels);
       chunkPts=chunkPtsPerChan*h.nADCNumChannels;
       % the number of those chunks..
       nChunk=floor(h.dataPts/chunkPts);
       % ..and the remainder
       restPts=h.dataPts-nChunk*chunkPts;
       restPtsPerChan=restPts/h.nADCNumChannels;
       % chunkwise row indices into d
       dix=(1:chunkPtsPerChan:h.dataPtsPerChan)';
       dix(:,2)=dix(:,1)+chunkPtsPerChan-1;
       dix(end,2)=h.dataPtsPerChan;
       if verbose && nChunk
         %disp(['reading file in ' int2str(nChunk) ' chunks of ~' ...
         %      num2str(chunk) ' Mb']);
       end
       % do it
       for ci=1:size(dix,1)-1
         [tmpd,n]=fread(fid,chunkPts,precision);
         if n~=chunkPts
           fclose(fid);
           error('OpenFileError:ErrorInABFLOAD',...
                ['something went wrong reading chunk #' int2str(ci) ...
                ' (' int2str(chunkPts) ' points should have been read, '...
                int2str(n) ' points actually read']);
         end
         % separate channels..
         tmpd   =   reshape(tmpd,h.nADCNumChannels,chunkPtsPerChan);
         d(dix(ci,1):dix(ci,2),:)=tmpd(chInd,:)';
       end
       % collect the rest, if any
       if restPts
         [tmpd,n]   =   fread(fid,restPts,precision);
         if n~=restPts
           fclose(fid);
           error('OpenFileError:ErrorInABFLOAD',...
                ['something went wrong reading last chunk (' ...
                int2str(restPts) ' points should have been read, ' ...
                int2str(n) ' points actually read']);
         end
         % separate channels..
         tmpd=reshape(tmpd,h.nADCNumChannels,restPtsPerChan);
         d(dix(end,1):dix(end,2),:)=tmpd(chInd,:)';
       end
     else
       % --- situation (i)
       [d,n]=fread(fid,h.dataPts,precision);
       if n~=h.dataPts,
         fclose(fid);
         error('OpenFileError:ErrorInABFLOAD',...
              ['something went wrong reading file (' int2str(h.dataPts) ...
               ' points should have been read, ' int2str(n) ...
               ' points actually read']);
       end
       % separate channels..
       d=reshape(d,h.nADCNumChannels,h.dataPtsPerChan);
       d=d';
     end
     % if data format is integer, scale appropriately; 
     % if it's float, d is fine
     if ~h.nDataFormat
       for j=1:length(chInd),
         ch=recChIdx(chInd(j))+1;
         d(:,j)=d(:,j)/(h.fInstrumentScaleFactor(ch)*h.fSignalGain(ch)...
                   *h.fADCProgrammableGain(ch)*addGain(ch))...
                   *h.fADCRange/h.lADCResolution+h.fInstrumentOffset(ch)...
                   -h.fSignalOffset(ch);
       end
     end
   end
 otherwise
   %disp(['recording mode is ''high-speed oscilloscope'' which is not '... 
   %      'implemented -- returning empty matrix']);
   d=[];
   h.si=[];
end
% h=orderfields(h);
fclose(fid);

function headPar=define_header(fileSig)
switch fileSig
 case 'ABF ' % ** note the blank
   % ************************
   %     abf version < 2.0
   % ************************
   %
   % temporary initializing var
   tmp=repmat(-1,1,16);
   % define vital header parameters and initialize them with -1: 
   % set up a cell array (and convert it to a struct below, which is more 
   % convenient)
   % column order is
   %        name, position in header in bytes, type, value)
   headPar={
     'fFileSignature',0,'*char',[-1 -1 -1 -1];
     'fFileVersionNumber',4,'float32',-1;
     'nOperationMode',8,'int16',-1;
     'lActualAcqLength',10,'int32',-1;
     'nNumPointsIgnored',14,'int16',-1;
     'lActualEpisodes',16,'int32',-1;
     'lFileStartTime',24,'int32',-1;
     'lDataSectionPtr',40,'int32',-1;
     'lTagSectionPtr',44,'int32',-1;
     'lNumTagEntries',48,'int32',-1;
     'lSynchArrayPtr',92,'int32',-1;
     'lSynchArraySize',96,'int32',-1;
     'nDataFormat',100,'int16',-1;
     'nADCNumChannels', 120, 'int16', -1;
     'fADCSampleInterval',122,'float', -1;
     'fSynchTimeUnit',130,'float',-1;
     'lNumSamplesPerEpisode',138,'int32',-1;
     'lPreTriggerSamples',142,'int32',-1;
     'lEpisodesPerRun',146,'int32',-1;
     'fADCRange', 244, 'float', -1;
     'lADCResolution', 252, 'int32', -1;
     'nFileStartMillisecs', 366, 'int16', -1;
     'nADCPtoLChannelMap', 378, 'int16', tmp;
     'nADCSamplingSeq', 410, 'int16',  tmp;
     'sADCChannelName',442, 'uchar', repmat(tmp,1,10);
     'sADCUnits',602, 'uchar', repmat(tmp,1,8);
     'fADCProgrammableGain', 730, 'float', tmp;
     'fInstrumentScaleFactor', 922, 'float', tmp;
     'fInstrumentOffset', 986, 'float', tmp;
     'fSignalGain', 1050, 'float', tmp;
     'fSignalOffset', 1114, 'float', tmp;
     'nTelegraphEnable',4512,'int16',tmp;
     'fTelegraphAdditGain',4576,'float',tmp
     };
 case 'ABF2'
   % ************************
   %     abf version >= 2.0
   % ************************
   headPar={
     'fFileSignature',0,'*char',[-1 -1 -1 -1];
     'fFileVersionNumber',4,'bit8=>int',[-1 -1 -1 -1];
     'uFileInfoSize',8,'uint32',-1;
     'lActualEpisodes',12,'uint32',-1;
     'uFileStartDate',16','uint32',-1;
     'uFileStartTimeMS',20,'uint32',-1;
     'uStopwatchTime',24,'uint32',-1;
     'nFileType',28,'int16',-1;
     'nDataFormat',30,'int16',-1;
     'nSimultaneousScan',32,'int16',-1;
     'nCRCEnable',34,'int16',-1;
     'uFileCRC',36,'uint32',-1;
     'FileGUID',40,'uint32',-1;
     'uCreatorVersion',56,'uint32',-1;
     'uCreatorNameIndex',60,'uint32',-1;
     'uModifierVersion',64,'uint32',-1;
     'uModifierNameIndex',68,'uint32',-1;
     'uProtocolPathIndex',72,'uint32',-1;
     };
end

function Sections=define_Sections
Sections={'ProtocolSection';
 'ADCSection';
 'DACSection';
 'EpochSection';
 'ADCPerDACSection';
 'EpochPerDACSection';
 'UserListSection';
 'StatsRegionSection';
 'MathSection';
 'StringsSection';
 'DataSection';
 'TagSection';
 'ScopeSection';
 'DeltaSection';
 'VoiceTagSection';
 'SynchArraySection';
 'AnnotationSection';
 'StatsSection';
 };

function ProtocolInfo=define_ProtocolInfo
ProtocolInfo={
 'nOperationMode','int16',1;
 'fADCSequenceInterval','float',1;
 'bEnableFileCompression','bit1',1;
 'sUnused1','char',3;
 'uFileCompressionRatio','uint32',1;
 'fSynchTimeUnit','float',1;
 'fSecondsPerRun','float',1;
 'lNumSamplesPerEpisode','int32',1;
 'lPreTriggerSamples','int32',1;
 'lEpisodesPerRun','int32',1;
 'lRunsPerTrial','int32',1;
 'lNumberOfTrials','int32',1;
 'nAveragingMode','int16',1;
 'nUndoRunCount','int16',1;
 'nFirstEpisodeInRun','int16',1;
 'fTriggerThreshold','float',1;
 'nTriggerSource','int16',1;
 'nTriggerAction','int16',1;
 'nTriggerPolarity','int16',1;
 'fScopeOutputInterval','float',1;
 'fEpisodeStartToStart','float',1;
 'fRunStartToStart','float',1;
 'lAverageCount','int32',1;
 'fTrialStartToStart','float',1;
 'nAutoTriggerStrategy','int16',1;
 'fFirstRunDelayS','float',1;
 'nChannelStatsStrategy','int16',1;
 'lSamplesPerTrace','int32',1;
 'lStartDisplayNum','int32',1;
 'lFinishDisplayNum','int32',1;
 'nShowPNRawData','int16',1;
 'fStatisticsPeriod','float',1;
 'lStatisticsMeasurements','int32',1;
 'nStatisticsSaveStrategy','int16',1;
 'fADCRange','float',1;
 'fDACRange','float',1;
 'lADCResolution','int32',1;
 'lDACResolution','int32',1;
 'nExperimentType','int16',1;
 'nManualInfoStrategy','int16',1;
 'nCommentsEnable','int16',1;
 'lFileCommentIndex','int32',1;
 'nAutoAnalyseEnable','int16',1;
 'nSignalType','int16',1;
 'nDigitalEnable','int16',1;
 'nActiveDACChannel','int16',1;
 'nDigitalHolding','int16',1;
 'nDigitalInterEpisode','int16',1;
 'nDigitalDACChannel','int16',1;
 'nDigitalTrainActiveLogic','int16',1;
 'nStatsEnable','int16',1;
 'nStatisticsClearStrategy','int16',1;
 'nLevelHysteresis','int16',1;
 'lTimeHysteresis','int32',1;
 'nAllowExternalTags','int16',1;
 'nAverageAlgorithm','int16',1;
 'fAverageWeighting','float',1;
 'nUndoPromptStrategy','int16',1;
 'nTrialTriggerSource','int16',1;
 'nStatisticsDisplayStrategy','int16',1;
 'nExternalTagType','int16',1;
 'nScopeTriggerOut','int16',1;
 'nLTPType','int16',1;
 'nAlternateDACOutputState','int16',1;
 'nAlternateDigitalOutputState','int16',1;
 'fCellID','float',3;
 'nDigitizerADCs','int16',1;
 'nDigitizerDACs','int16',1;
 'nDigitizerTotalDigitalOuts','int16',1;
 'nDigitizerSynchDigitalOuts','int16',1;
 'nDigitizerType','int16',1;
 };

function ADCInfo=define_ADCInfo
ADCInfo={
 'nADCNum','int16',1;
 'nTelegraphEnable','int16',1;
 'nTelegraphInstrument','int16',1;
 'fTelegraphAdditGain','float',1;
 'fTelegraphFilter','float',1;
 'fTelegraphMembraneCap','float',1;
 'nTelegraphMode','int16',1;
 'fTelegraphAccessResistance','float',1;
 'nADCPtoLChannelMap','int16',1;
 'nADCSamplingSeq','int16',1;
 'fADCProgrammableGain','float',1;
 'fADCDisplayAmplification','float',1;
 'fADCDisplayOffset','float',1;
 'fInstrumentScaleFactor','float',1;
 'fInstrumentOffset','float',1;
 'fSignalGain','float',1;
 'fSignalOffset','float',1;
 'fSignalLowpassFilter','float',1;
 'fSignalHighpassFilter','float',1;
 'nLowpassFilterType','char',1;
 'nHighpassFilterType','char',1;
 'fPostProcessLowpassFilter','float',1;
 'nPostProcessLowpassFilterType','char',1;
 'bEnabledDuringPN','bit1',1;
 'nStatsChannelPolarity','int16',1;
 'lADCChannelNameIndex','int32',1;
 'lADCUnitsIndex','int32',1;
 };

function TagInfo=define_TagInfo
TagInfo={
   'lTagTime','int32',1;
   'sComment','char',56;
   'nTagType','int16',1;
   'nVoiceTagNumber_or_AnnotationIndex','int16',1;
};

function Section=ReadSection(fid,offset,Format) %#ok<STOUT>
s=cell2struct(Format,{'name','numType','number'},2);
fseek(fid,offset,'bof');
for i=1:length(s)
 eval(['[Section.' s(i).name ',n]=fread(fid,' num2str(s(i).number) ...
       ',''' s(i).numType ''');']);
end

function SectionInfo=ReadSectionInfo(fid,offset) %#ok<DEFNU>
fseek(fid,offset,'bof');
SectionInfo.uBlockIndex=fread(fid,1,'uint32');
fseek(fid,offset+4,'bof');
SectionInfo.uBytes=fread(fid,1,'uint32');
fseek(fid,offset+8,'bof');
SectionInfo.llNumEntries=fread(fid,1,'int64');

function pvpmod(x)
% PVPMOD             - evaluate parameter/value pairs
% pvpmod(x) assigns the value x(i+1) to the parameter defined by the
% string x(i) in the calling workspace. This is useful to evaluate 
% <varargin> contents in an mfile, e.g. to change default settings 
% of any variable initialized before pvpmod(x) is called.
%
% (c) U. Egert 1998

% this loop is assigns the parameter/value pairs in x to the calling
% workspace.
if ~isempty(x)
  for i = 1:2:size(x,2)
     assignin('caller', x{i}, x{i+1});
  end;
end;
