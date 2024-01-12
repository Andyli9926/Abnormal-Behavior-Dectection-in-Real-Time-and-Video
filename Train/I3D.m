doTraining = true;

% Download Training and Validation Data
% This example trains an Inflated-3D (I3D) Video Classifier using the HMDB51 data set. Use the downloadHMDB51 supporting function, listed at the end of this example, to download the HMDB51 data set to a folder named hmdb51.
classes = ["fall_floor","fight","hit","kick","normal","run","stand","walk"];
dataFolder = fullfile("G:\FYP\CNNdata");
The data set contains about 2 GB of video data for 7000 clips over 51 classes, such as drink, run, and shake hands. Each video frame has a height of 240 pixels and a minimum width of 176 pixels. The number of frames ranges from 18 to approximately 1000.
[labels,files] = folders2labels(fullfile(dataFolder,string(classes)),...
    "IncludeSubfolders",true,...
    "FileExtensions",'.avi');

indices = splitlabels(labels,0.8,'randomized');

trainFilenames = files(indices{1});
testFilenames  = files(indices{2});
% To normalize the input data for the network, the minimum and maximum values for the data set are provided in the MAT file inputStatistics.mat, attached to this example. To find the minimum and maximum values for a different data set, use the inputStatistics supporting function, listed at the end of this example.
inputStatsFilename = 'input.mat';
if ~exist(inputStatsFilename, 'file')
    disp("Reading all the training data for input statistics...")
    inputStats = inputStatistics(dataFolder);
else
    d = load(inputStatsFilename);
    inputStats = d.inputStats;    
end

% Load Dataset
% Specify the number of video frames the datastore should be configured to output for each time data is read from the datastore. 
numFrames = 32;
frameSize = [112,112];
rgbChannels = 3;
flowChannels = 2;

isDataForTraining = true;
dsTrain = createFileDatastore(trainFilenames,numFrames,rgbChannels,classes,isDataForTraining);

isDataForTraining = false;
dsVal = createFileDatastore(testFilenames,numFrames,rgbChannels,classes,isDataForTraining);

% Define Network Architecture
baseNetwork = "googlenet-video-flow";
% Specify the input size for the Inflated-3D Video Classifier.
inputSize = [frameSize, rgbChannels, numFrames];

oflowMin = squeeze(inputStats.oflowMin)';
oflowMax = squeeze(inputStats.oflowMax)';
rgbMin   = squeeze(inputStats.rgbMin)';
rgbMax   = squeeze(inputStats.rgbMax)';

stats.Video.Min               = rgbMin;
stats.Video.Max               = rgbMax;
stats.Video.Mean              = [];
stats.Video.StandardDeviation = [];

stats.OpticalFlow.Min               = oflowMin(1:flowChannels);
stats.OpticalFlow.Max               = oflowMax(1:flowChannels);
stats.OpticalFlow.Mean              = [];
stats.OpticalFlow.StandardDeviation = [];

% Create the I3D Video Classifier by using the inflated3dVideoClassifier function.
i3d = inflated3dVideoClassifier(baseNetwork,string(classes),...
    "InputSize",inputSize,...
    "InputNormalizationStatistics",stats);
    
% Specify a model name for the video classifier.
i3d.ModelName = "Inflated-3D Activity Recognizer Using Video and Optical Flow";

% Augment and Preprocess Training Data
dsTrain = transform(dsTrain, @augmentVideo);

preprocessInfo.Statistics = i3d.InputNormalizationStatistics;
preprocessInfo.InputSize = inputSize;
preprocessInfo.SizingOption = "resize";
dsTrain = transform(dsTrain, @(data)preprocessVideoClips(data, preprocessInfo));
dsVal = transform(dsVal, @(data)preprocessVideoClips(data, preprocessInfo));

% Specify Training Options
% Train with a mini-batch size of 20 for 600 iterations. Specify the iteration after which to save the video classifier with the best validation accuracy by using the SaveBestAfterIteration parameter.
% Specify the cosine-annealing learning rate schedule [3] parameters:
% A minimum learning rate of 1e-4.
% A maximum learning rate of 1e-3.
% Cosine number of iterations of 100, 200, and 300, after which the learning rate schedule cycle restarts. The option CosineNumIterations defines the width of each cosine cycle.
% Specify the parameters for SGDM optimization. Initialize the SGDM optimization parameters at the beginning of the training:
% A momentum of 0.9.
% An initial velocity parameter initialized as [].
% An L2 regularization factor of 0.0005.
params.Classes = classes;
params.MiniBatchSize = 10;
params.NumIterations = 1600;
params.SaveBestAfterIteration = 1200;
params.CosineNumIterations = [100, 200, 300];
params.MinLearningRate = 1e-4;
params.MaxLearningRate = 1e-3;
params.Momentum = 0.9;
params.VelocityRGB = [];
params.VelocityFlow = [];
params.L2Regularization = 0.0005;
params.ProgressPlot = true;
params.Verbose = true;
params.ValidationData = dsVal;
params.DispatchInBackground = false;
params.NumWorkers = 4;

% Train I3D Video Classifier
params.ModelFilename = "G:\FYP\inflated3d-V4.mat";
if doTraining
    epoch     = 1;
    bestLoss  = realmax;

    accTrain     = [];
    accTrainRGB  = [];
    accTrainFlow = [];
    lossTrain    = [];

    iteration = 1;
    start     = tic;
    trainTime = start;
    shuffled  = shuffleTrainDs(dsTrain);

    % Number of outputs is three: One for RGB frames, one for optical flow
    % data, and one for ground truth labels.
    numOutputs = 3;
    mbq        = createMiniBatchQueue(shuffled, numOutputs, params);
    
    % Use the initializeTrainingProgressPlot and initializeVerboseOutput
    % supporting functions, listed at the end of the example, to initialize
    % the training progress plot and verbose output to display the training
    % loss, training accuracy, and validation accuracy.
    plotters = initializeTrainingProgressPlot(params);
    initializeVerboseOutput(params);

    while iteration <= params.NumIterations

        % Iterate through the data set.
        [dlVideo,dlFlow,dlY] = next(mbq);

        % Evaluate the model gradients and loss using dlfeval.
        [gradRGB,gradFlow,loss,acc,accRGB,accFlow,stateRGB,stateFlow] = ...
            dlfeval(@modelGradients,i3d,dlVideo,dlFlow,dlY);

        % Accumulate the loss and accuracies.
        lossTrain    = [lossTrain, loss];
        accTrain     = [accTrain, acc];
        accTrainRGB  = [accTrainRGB, accRGB];
        accTrainFlow = [accTrainFlow, accFlow];

        % Update the network state.
        i3d.VideoState       = stateRGB;
        i3d.OpticalFlowState = stateFlow;
        
        % Update the gradients and parameters for the RGB and optical flow
        % subnetworks using the SGDM optimizer.
        [i3d.VideoLearnables,params.VelocityRGB] = ...
            updateLearnables(i3d.VideoLearnables,gradRGB,params,params.VelocityRGB,iteration);
        [i3d.OpticalFlowLearnables,params.VelocityFlow,learnRate] = ...
            updateLearnables(i3d.OpticalFlowLearnables,gradFlow,params,params.VelocityFlow,iteration);
        
        if ~hasdata(mbq) || iteration == params.NumIterations
            % Current epoch is complete. Do validation and update progress.
            trainTime = toc(trainTime);

            [validationTime,cmat,lossValidation,accValidation,accValidationRGB,accValidationFlow] = ...
                doValidation(params, i3d);

            accTrain     = mean(accTrain);
            accTrainRGB  = mean(accTrainRGB);
            accTrainFlow = mean(accTrainFlow);
            lossTrain    = mean(lossTrain);

            % Update the training progress.
            displayVerboseOutputEveryEpoch(params,start,learnRate,epoch,iteration,...
                accTrain,accTrainRGB,accTrainFlow,...
                accValidation,accValidationRGB,accValidationFlow,...
                lossTrain,lossValidation,trainTime,validationTime);
            updateProgressPlot(params,plotters,epoch,iteration,start,lossTrain,accTrain,accValidation);
            
            % Save the trained video classifier and the parameters, that gave 
            % the best validation loss so far. Use the saveData supporting function,
            % listed at the end of this example.
            bestLoss = saveData(i3d,bestLoss,iteration,cmat,lossTrain,lossValidation,...
                accTrain,accValidation,params);
        end
        
        if ~hasdata(mbq) && iteration < params.NumIterations
            % Current epoch is complete. Initialize the training loss, accuracy
            % values, and minibatchqueue for the next epoch.
            accTrain     = [];
            accTrainRGB  = [];
            accTrainFlow = [];
            lossTrain    = [];
        
            trainTime  = tic;
            epoch      = epoch + 1;
            shuffled   = shuffleTrainDs(dsTrain);
            numOutputs = 3;
            mbq        = createMiniBatchQueue(shuffled, numOutputs, params);
            
        end 
        
        iteration = iteration + 1
    end
    
    % Display a message when training is complete.
    endVerboseOutput(params);
    
    disp("Model saved to: " + params.ModelFilename);
end

% Evaluate Trained Network
if doTraining
    transferLearned = load(params.ModelFilename);
    inflated3dPretrained = transferLearned.data.inflated3d;
end

numOutputs = 3;
mbq = createMiniBatchQueue(params.ValidationData, numOutputs, params);

numClasses = numel(classes);
cmat = sparse(numClasses,numClasses);
while hasdata(mbq)
    [dlRGB, dlFlow, dlY] = next(mbq);
    
    % Pass the video input as RGB and optical flow data through the
    % two-stream I3D Video Classifier to get the separate predictions.
    [dlYPredRGB,dlYPredFlow] = predict(inflated3dPretrained,dlRGB,dlFlow);

    % Fuse the predictions by calculating the average of the predictions.
    dlYPred = (dlYPredRGB + dlYPredFlow)/2;
    
    % Calculate the accuracy of the predictions.
    [~,YTest] = max(dlY,[],1);
    [~,YPred] = max(dlYPred,[],1);

    cmat = aggregateConfusionMetric(cmat,YTest,YPred);
end

accuracyEval = sum(diag(cmat))./sum(cmat,"all")
% Display the matrix.
figure
chart = confusionchart(cmat,classes);

% Supporting Functions
function inputStats = inputStatistics(dataFolder)
    ds = createDatastore(dataFolder);
    ds.ReadFcn = @getMinMax;

    tic;
    tt = tall(ds);
    varnames = {'rgbMax','rgbMin','oflowMax','oflowMin'};
    stats = gather(groupsummary(tt,[],{'max','min'}, varnames));
    inputStats.Filename = gather(tt.Filename);
    inputStats.NumFrames = gather(tt.NumFrames);
    inputStats.rgbMax = stats.max_rgbMax;
    inputStats.rgbMin = stats.min_rgbMin;
    inputStats.oflowMax = stats.max_oflowMax;
    inputStats.oflowMin = stats.min_oflowMin;
    save('inputStatistics.mat','inputStats');
    toc;
end

function data = getMinMax(filename)
    reader = VideoReader(filename);
    opticFlow = opticalFlowFarneback;
    data = [];
    while hasFrame(reader)
        frame = readFrame(reader);
        [rgb,oflow] = findMinMax(frame,opticFlow);
        data = assignMinMax(data, rgb, oflow);
    end

    totalFrames = floor(reader.Duration * reader.FrameRate);
    totalFrames = min(totalFrames, reader.NumFrames);
    
    [labelName, filename] = getLabelFilename(filename);
    data.Filename = fullfile(labelName, filename);
    data.NumFrames = totalFrames;

    data = struct2table(data,'AsArray',true);
end

function [labelName, filename] = getLabelFilename(filename) 
    fileNameSplit = split(filename,'/'); 
    labelName = fileNameSplit{end-1}; 
    filename = fileNameSplit{end};
end

function data = assignMinMax(data, rgb, oflow)
    if isempty(data)
        data.rgbMax = rgb.Max;
        data.rgbMin = rgb.Min;
        data.oflowMax = oflow.Max;
        data.oflowMin = oflow.Min;
        return;
    end
    data.rgbMax = max(data.rgbMax, rgb.Max);
    data.rgbMin = min(data.rgbMin, rgb.Min);

    data.oflowMax = max(data.oflowMax, oflow.Max);
    data.oflowMin = min(data.oflowMin, oflow.Min);
end

function [rgbMinMax,oflowMinMax] = findMinMax(rgb, opticFlow)
    rgbMinMax.Max = max(rgb,[],[1,2]);
    rgbMinMax.Min = min(rgb,[],[1,2]);

    gray = rgb2gray(rgb);
    flow = estimateFlow(opticFlow,gray);
    oflow = cat(3,flow.Vx,flow.Vy,flow.Magnitude);

    oflowMinMax.Max = max(oflow,[],[1,2]);
    oflowMinMax.Min = min(oflow,[],[1,2]);
end

function ds = createDatastore(folder)    
    ds = fileDatastore(folder,...
        'IncludeSubfolders', true,...
        'FileExtensions', '.avi',...
        'UniformRead', true,...
        'ReadFcn', @getMinMax);
    disp("NumFiles: " + numel(ds.Files));
end
% createFileDatastore
function datastore = createFileDatastore(trainingFolder,numFrames,numChannels,classes,isDataForTraining)
    readFcn = @(f,u)readVideo(f,u,numFrames,numChannels,classes,isDataForTraining);
    datastore = fileDatastore(trainingFolder,...
        'IncludeSubfolders',true,...
        'FileExtensions','.avi',...
        'ReadFcn',readFcn,...
        'ReadMode','partialfile');
end

% shuffleTrainDs
% The shuffleTrainDs function shuffles the files present in the training datastore dsTrain.
function shuffled = shuffleTrainDs(dsTrain)
shuffled = copy(dsTrain);
transformed = isa(shuffled, 'matlab.io.datastore.TransformedDatastore');
if transformed
    files = shuffled.UnderlyingDatastores{1}.Files;
else 
    files = shuffled.Files;
end
n = numel(files);
shuffledIndices = randperm(n);  
if transformed
    shuffled.UnderlyingDatastores{1}.Files = files(shuffledIndices);
else
    shuffled.Files = files(shuffledIndices);
end

reset(shuffled);
end
% readVideo
function [data,userdata,done] = readVideo(filename,userdata,numFrames,numChannels,classes,isDataForTraining)
    if isempty(userdata)
        userdata.reader      = VideoReader(filename);
        userdata.batchesRead = 0;
        
        userdata.label = getLabel(filename,classes);

        totalFrames = floor(userdata.reader.Duration * userdata.reader.FrameRate);
        totalFrames = min(totalFrames, userdata.reader.NumFrames);
        userdata.totalFrames = totalFrames;
        userdata.datatype = class(read(userdata.reader,1));
    end
    reader      = userdata.reader;
    totalFrames = userdata.totalFrames;
    label       = userdata.label;
    batchesRead = userdata.batchesRead;

    if isDataForTraining
        video = readForTraining(reader, numFrames, totalFrames);
    else
        video = readForValidation(reader, userdata.datatype, numChannels, numFrames, totalFrames);
    end   

    data = {video, label};

    batchesRead = batchesRead + 1;

    userdata.batchesRead = batchesRead;

    if numFrames > totalFrames
        numBatches = 1;
    else
        numBatches = floor(totalFrames/numFrames);
    end
    % Set the done flag to true, if the reader has read all the frames or
    % if it is training.
    done = batchesRead == numBatches || isDataForTraining;
end
% readForTraining
function video = readForTraining(reader, numFrames, totalFrames)
    if numFrames >= totalFrames
        startIdx = 1;
        endIdx = totalFrames;
    else
        startIdx = randperm(totalFrames - numFrames + 1);
        startIdx = startIdx(1);
        endIdx = startIdx + numFrames - 1;
    end
    video = read(reader,[startIdx,endIdx]);
    if numFrames > totalFrames
        % Add more frames to fill in the network input size.
        additional = ceil(numFrames/totalFrames);
        video = repmat(video,1,1,1,additional);
        video = video(:,:,:,1:numFrames);
    end
end

% readForValidation
function video = readForValidation(reader, datatype, numChannels, numFrames, totalFrames)
    H = reader.Height;
    W = reader.Width;
    toRead = min([numFrames,totalFrames]);
    video = zeros([H,W,numChannels,toRead], datatype);
    frameIndex = 0;
    while hasFrame(reader) && frameIndex < numFrames
        frame = readFrame(reader);
        frameIndex = frameIndex + 1;
        video(:,:,:,frameIndex) = frame;
    end
    
    if frameIndex < numFrames
        video = video(:,:,:,1:frameIndex);
        additional = ceil(numFrames/frameIndex);
        video = repmat(video,1,1,1,additional);
        video = video(:,:,:,1:numFrames);       
    end
end
% getLabel
% The getLabel function obtains the label name from the full path of a filename. The label for a file is the folder in which it exists. For example, for a file path such as "/path/to/dataset/clapping/video_0001.avi", the label name is "clapping".
function label = getLabel(filename,classes)
    folder = fileparts(string(filename));
    [~,label] = fileparts(folder);
    label = categorical(string(label), string(classes));
end
% augmentVideo
% The augmentVideo function uses the augment transform function provided by the augmentTransform supporting function to apply the same augmentation across a video sequence.
function data = augmentVideo(data)
    numSequences = size(data,1);
    for ii = 1:numSequences
        video = data{ii,1};
        % HxWxC
        sz = size(video,[1,2,3]);
        % One augmentation per sequence
        augmentFcn = augmentTransform(sz);
        data{ii,1} = augmentFcn(video);
    end
end
% augmentTransform
% The augmentTransform function creates an augmentation method with random left-right flipping and scaling factors.
function augmentFcn = augmentTransform(sz)
% Randomly flip and scale the image.
tform = randomAffine2d('XReflection',true,'Scale',[1 1.1]);
rout = affineOutputView(sz,tform,'BoundsStyle','CenterOutput');

augmentFcn = @(data)augmentData(data,tform,rout);

    function data = augmentData(data,tform,rout)
        data = imwarp(data,tform,'OutputView',rout);
    end
end

% preprocessVideoClips
function preprocessed = preprocessVideoClips(data, info)
inputSize = info.InputSize(1:2);
sizingOption = info.SizingOption;
switch sizingOption
    case "resize"
        sizingFcn = @(x)imresize(x,inputSize);
    case "randomcrop"
        sizingFcn = @(x)cropVideo(x,@randomCropWindow2d,inputSize);
    case "centercrop"
        sizingFcn = @(x)cropVideo(x,@centerCropWindow2d,inputSize);
end
numClips = size(data,1);

rgbMin   = info.Statistics.Video.Min;
rgbMax   = info.Statistics.Video.Max;
oflowMin = info.Statistics.OpticalFlow.Min;
oflowMax = info.Statistics.OpticalFlow.Max;

numChannels = length(rgbMin);
rgbMin   = reshape(rgbMin, 1, 1, numChannels);
rgbMax   = reshape(rgbMax, 1, 1, numChannels);

numChannels = length(oflowMin);
oflowMin = reshape(oflowMin, 1, 1, numChannels);
oflowMax = reshape(oflowMax, 1, 1, numChannels);

preprocessed = cell(numClips, 3);
for ii = 1:numClips
    video   = data{ii,1};
    resized = sizingFcn(video);
    oflow   = computeFlow(resized,inputSize);

    % Cast the input to single.
    resized = single(resized);
    oflow   = single(oflow);

    % Rescale the input between -1 and 1.
    resized = rescale(resized,-1,1,"InputMin",rgbMin,"InputMax",rgbMax);
    oflow   = rescale(oflow,-1,1,"InputMin",oflowMin,"InputMax",oflowMax);

    preprocessed{ii,1} = resized;
    preprocessed{ii,2} = oflow;
    preprocessed{ii,3} = data{ii,2};
end    
end

function outData = cropVideo(data, cropFcn, inputSize)
imsz = size(data,[1,2]);
cropWindow = cropFcn(imsz, inputSize);
numFrames = size(data,4);
sz = [inputSize, size(data,3), numFrames];
outData = zeros(sz, 'like', data);
for f = 1:numFrames
    outData(:,:,:,f) = imcrop(data(:,:,:,f), cropWindow);
end
end
% computeFlow
function opticalFlowData = computeFlow(videoFrames, inputSize)
opticalFlow = opticalFlowFarneback;
numFrames = size(videoFrames,4);
sz = [inputSize, 2, numFrames];
opticalFlowData = zeros(sz, 'like', videoFrames);
for f = 1:numFrames
    gray = rgb2gray(videoFrames(:,:,:,f));
    flow = estimateFlow(opticalFlow,gray);

    opticalFlowData(:,:,:,f) = cat(3,flow.Vx,flow.Vy);
end
end
% createMiniBatchQueue
function mbq = createMiniBatchQueue(datastore, numOutputs, params)
if params.DispatchInBackground && isempty(gcp('nocreate'))
    % Start a parallel pool, if DispatchInBackground is true, to dispatch
    % data in the background using the parallel pool.
    c = parcluster('local');
    c.NumWorkers = params.NumWorkers;
    parpool('local',params.NumWorkers);
end
p = gcp('nocreate');
if ~isempty(p)
    datastore = DispatchInBackgroundDatastore(datastore, p.NumWorkers);
end
inputFormat(1:numOutputs-1) = "SSCTB";
outputFormat = "CB";
mbq = minibatchqueue(datastore, numOutputs, ...
    "MiniBatchSize", params.MiniBatchSize, ...
    "MiniBatchFcn", @batchVideoAndFlow, ...
    "MiniBatchFormat", [inputFormat,outputFormat]);
end
% batchVideoAndFlow
function [video,flow,labels] = batchVideoAndFlow(video, flow, labels)
% Batch dimension: 5
video = cat(5,video{:});
flow = cat(5,flow{:});

% Batch dimension: 2
labels = cat(2,labels{:});

% Feature dimension: 1
labels = onehotencode(labels,1);
end
% modelGradients
function [gradientsRGB,gradientsFlow,loss,acc,accRGB,accFlow,stateRGB,stateFlow] = modelGradients(i3d,dlRGB,dlFlow,Y)

% Pass video input as RGB and optical flow data through the two-stream
% network.
[dlYPredRGB,dlYPredFlow,stateRGB,stateFlow] = forward(i3d,dlRGB,dlFlow);

% Calculate fused loss, gradients, and accuracy for the two-stream
% predictions.
rgbLoss = crossentropy(dlYPredRGB,Y);
flowLoss = crossentropy(dlYPredFlow,Y);
% Fuse the losses.
loss = mean([rgbLoss,flowLoss]);

gradientsRGB = dlgradient(rgbLoss,i3d.VideoLearnables);
gradientsFlow = dlgradient(flowLoss,i3d.OpticalFlowLearnables);

% Fuse the predictions by calculating the average of the predictions.
dlYPred = (dlYPredRGB + dlYPredFlow)/2;

% Calculate the accuracy of the predictions.
[~,YTest] = max(Y,[],1);
[~,YPred] = max(dlYPred,[],1);

acc = gather(extractdata(sum(YTest == YPred)./numel(YTest)));

% Calculate the accuracy of the RGB and flow predictions.
[~,YTest] = max(Y,[],1);
[~,YPredRGB] = max(dlYPredRGB,[],1);
[~,YPredFlow] = max(dlYPredFlow,[],1);

accRGB = gather(extractdata(sum(YTest == YPredRGB)./numel(YTest)));
accFlow = gather(extractdata(sum(YTest == YPredFlow)./numel(YTest)));
end
% updateLearnables
% The updateLearnables function updates the provided learnables with gradients and other parameters using SGDM optimization function sgdmupdate.
function [learnables,velocity,learnRate] = updateLearnables(learnables,gradients,params,velocity,iteration)
    % Determine the learning rate using the cosine-annealing learning rate schedule.
    learnRate = cosineAnnealingLearnRate(iteration, params);

    % Apply L2 regularization to the weights.
    idx = learnables.Parameter == "Weights";
    gradients(idx,:) = dlupdate(@(g,w) g + params.L2Regularization*w, gradients(idx,:), learnables(idx,:));

    % Update the network parameters using the SGDM optimizer.
    [learnables, velocity] = sgdmupdate(learnables, gradients, velocity, learnRate, params.Momentum);
end
% cosineAnnealingLearnRate
function lr = cosineAnnealingLearnRate(iteration, params)
    if iteration == params.NumIterations
        lr = params.MinLearningRate;
        return;
    end
    cosineNumIter = [0, params.CosineNumIterations];
    csum = cumsum(cosineNumIter);
    block = find(csum >= iteration, 1,'first');
    cosineIter = iteration - csum(block - 1);
    annealingIteration = mod(cosineIter, cosineNumIter(block));
    cosineIteration = cosineNumIter(block);
    minR = params.MinLearningRate;
    maxR = params.MaxLearningRate;
    cosMult = 1 + cos(pi * annealingIteration / cosineIteration);
    lr = minR + ((maxR - minR) *  cosMult / 2);
end
% aggregateConfusionMetric
function cmat = aggregateConfusionMetric(cmat,YTest,YPred)
YTest = gather(extractdata(YTest));
YPred = gather(extractdata(YPred));
[m,n] = size(cmat);
cmat = cmat + full(sparse(YTest,YPred,1,m,n));
end
% doValidation
function [validationTime, cmat, lossValidation, accValidation, accValidationRGB, accValidationFlow] = doValidation(params, i3d)

validationTime = tic;

numOutputs = 3;
mbq = createMiniBatchQueue(params.ValidationData, numOutputs, params);

lossValidation = [];
numClasses = numel(params.Classes);
cmat = sparse(numClasses,numClasses);
cmatRGB = sparse(numClasses,numClasses);
cmatFlow = sparse(numClasses,numClasses);
while hasdata(mbq)

    [dlX1,dlX2,dlY] = next(mbq);

    [loss,YTest,YPred,YPredRGB,YPredFlow] = predictValidation(i3d,dlX1,dlX2,dlY);

    lossValidation = [lossValidation,loss];
    cmat = aggregateConfusionMetric(cmat,YTest,YPred);
    cmatRGB = aggregateConfusionMetric(cmatRGB,YTest,YPredRGB);
    cmatFlow = aggregateConfusionMetric(cmatFlow,YTest,YPredFlow);
end
lossValidation = mean(lossValidation);
accValidation = sum(diag(cmat))./sum(cmat,"all");
accValidationRGB = sum(diag(cmatRGB))./sum(cmatRGB,"all");
accValidationFlow = sum(diag(cmatFlow))./sum(cmatFlow,"all");

validationTime = toc(validationTime);
end
% predictValidation
% The predictValidation function calculates the loss and prediction values using the provided video classifier for RGB and optical flow data.
function [loss,YTest,YPred,YPredRGB,YPredFlow] = predictValidation(i3d,dlRGB,dlFlow,Y)

% Pass the video input through the two-stream Inflated-3D video classifier.
[dlYPredRGB,dlYPredFlow] = predict(i3d,dlRGB,dlFlow);

% Calculate the cross-entropy separately for the two-stream outputs.
rgbLoss = crossentropy(dlYPredRGB,Y);
flowLoss = crossentropy(dlYPredFlow,Y);

% Fuse the losses.
loss = mean([rgbLoss,flowLoss]);

% Fuse the predictions by calculating the average of the predictions.
dlYPred = (dlYPredRGB + dlYPredFlow)/2;

% Calculate the accuracy of the predictions.
[~,YTest] = max(Y,[],1);
[~,YPred] = max(dlYPred,[],1);

[~,YPredRGB] = max(dlYPredRGB,[],1);
[~,YPredFlow] = max(dlYPredFlow,[],1);

end
% saveData
% The saveData function saves the given Inflated-3d Video Classifier, accuracy, loss, and other training parameters to a MAT-file.
function bestLoss = saveData(inflated3d,bestLoss,iteration,cmat,lossTrain,lossValidation,...
                accTrain,accValidation,params)
if iteration >= params.SaveBestAfterIteration
    lossValidtion = extractdata(gather(lossValidation));
    if lossValidtion < bestLoss
        params = rmfield(params, 'VelocityRGB');
        params = rmfield(params, 'VelocityFlow');
        bestLoss = lossValidtion;
        inflated3d = gatherFromGPUToSave(inflated3d);
        data.BestLoss = bestLoss;
        data.TrainingLoss = extractdata(gather(lossTrain));
        data.TrainingAccuracy = accTrain;
        data.ValidationAccuracy = accValidation;
        data.ValidationConfmat= cmat;
        data.inflated3d = inflated3d;
        data.Params = params;
        save(params.ModelFilename, 'data');
    end
end
end
% gatherFromGPUToSave
% The gatherFromGPUToSave function gathers data from the GPU in order to save the video classifier to disk.
function classifier = gatherFromGPUToSave(classifier)
if ~canUseGPU
    return;
end
p = string(properties(classifier));
p = p(endsWith(p, ["Learnables","State"]));
for jj = 1:numel(p)
    prop = p(jj);
    classifier.(prop) = gatherValues(classifier.(prop));
end
    function tbl = gatherValues(tbl)
        for ii = 1:height(tbl)
            tbl.Value{ii} = gather(tbl.Value{ii});
        end
    end
end
% checkForHMDB51Folder
function classes = checkForHMDB51Folder(dataLoc)
hmdbFolder = fullfile(dataLoc, "hmdb51_org");
if ~isfolder(hmdbFolder)
    error("Download 'hmdb51_org.rar' file using the supporting function 'downloadHMDB51' before running the example and extract the RAR file.");    
end

classes = ["brush_hair","cartwheel","catch","chew","clap","climb","climb_stairs",...
    "dive","draw_sword","dribble","drink","eat","fall_floor","fencing",...
    "flic_flac","golf","handstand","hit","hug","jump","kick","kick_ball",...
    "kiss","laugh","pick","pour","pullup","punch","push","pushup","ride_bike",...
    "ride_horse","run","shake_hands","shoot_ball","shoot_bow","shoot_gun",...
    "sit","situp","smile","smoke","somersault","stand","swing_baseball","sword",...
    "sword_exercise","talk","throw","turn","walk","wave"];
expectFolders = fullfile(hmdbFolder, classes);
if ~all(arrayfun(@(x)exist(x,'dir'),expectFolders))
    error("Download hmdb51_org.rar using the supporting function 'downloadHMDB51' before running the example and extract the RAR file.");
end
end
% downloadHMDB51
% The downloadHMDB51 function downloads the data set and saves it to a directory.
function downloadHMDB51(dataLoc)

if nargin == 0
    dataLoc = pwd;
end
dataLoc = string(dataLoc);

if ~isfolder(dataLoc)
    mkdir(dataLoc);
end

dataUrl     = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar";
options     = weboptions('Timeout', Inf);
rarFileName = fullfile(dataLoc, 'hmdb51_org.rar');

% Download the RAR file and save it to the download folder.
if ~isfile(rarFileName)
    disp("Downloading hmdb51_org.rar (2 GB) to the folder:")
    disp(dataLoc)
    disp("This download can take a few minutes...") 
    websave(rarFileName, dataUrl, options); 
    disp("Download complete.")
    disp("Extract the hmdb51_org.rar file contents to the folder: ") 
    disp(dataLoc)
end
end
% initializeTrainingProgressPlot
function plotters = initializeTrainingProgressPlot(params)
if params.ProgressPlot
    % Plot the loss, training accuracy, and validation accuracy.
    figure
    
    % Loss plot
    subplot(2,1,1)
    plotters.LossPlotter = animatedline;
    xlabel("Iteration")
    ylabel("Loss")
    
    % Accuracy plot
    subplot(2,1,2)
    plotters.TrainAccPlotter = animatedline('Color','b');
    plotters.ValAccPlotter = animatedline('Color','g');
    legend('Training Accuracy','Validation Accuracy','Location','northwest');
    xlabel("Iteration")
    ylabel("Accuracy")
else
    plotters = [];
end
end
% updateProgressPlot
% The updateProgressPlot function updates the progress plot with loss and accuracy information during training.
function updateProgressPlot(params,plotters,epoch,iteration,start,lossTrain,accuracyTrain,accuracyValidation)
if params.ProgressPlot
    
    % Update the training progress.
    D = duration(0,0,toc(start),"Format","hh:mm:ss");
    title(plotters.LossPlotter.Parent,"Epoch: " + epoch + ", Elapsed: " + string(D));
    addpoints(plotters.LossPlotter,iteration,double(gather(extractdata(lossTrain))));
    addpoints(plotters.TrainAccPlotter,iteration,accuracyTrain);
    addpoints(plotters.ValAccPlotter,iteration,accuracyValidation);
    drawnow
end
end
% initializeVerboseOutput
function initializeVerboseOutput(params)
if params.Verbose
    disp(" ")
    if canUseGPU
        disp("Training on GPU.")
    else
        disp("Training on CPU.")
    end
    p = gcp('nocreate');
    if ~isempty(p)
        disp("Training on parallel cluster '" + p.Cluster.Profile + "'. ")
    end
    disp("NumIterations:" + string(params.NumIterations));
    disp("MiniBatchSize:" + string(params.MiniBatchSize));
    disp("Classes:" + join(string(params.Classes), ","));    
    disp("|=======================================================================================================================================================================|")
    disp("| Epoch | Iteration | Time Elapsed |     Mini-Batch Accuracy    |    Validation Accuracy     | Mini-Batch | Validation |  Base Learning  | Train Time | Validation Time |")
    disp("|       |           |  (hh:mm:ss)  |       (Avg:RGB:Flow)       |       (Avg:RGB:Flow)       |    Loss    |    Loss    |      Rate       | (hh:mm:ss) |   (hh:mm:ss)    |")
    disp("|=======================================================================================================================================================================|")
end
end
% displayVerboseOutputEveryEpoch
% The displayVerboseOutputEveryEpoch function displays the verbose output of the training values, such as the epoch, mini-batch accuracy, validation accuracy, and mini-batch loss.
function displayVerboseOutputEveryEpoch(params,start,learnRate,epoch,iteration,...
    accTrain,accTrainRGB,accTrainFlow,accValidation,accValidationRGB,accValidationFlow,lossTrain,lossValidation,trainTime,validationTime)
if params.Verbose
    D = duration(0,0,toc(start),'Format','hh:mm:ss');
    trainTime = duration(0,0,trainTime,'Format','hh:mm:ss');
    validationTime = duration(0,0,validationTime,'Format','hh:mm:ss');

    lossValidation = gather(extractdata(lossValidation));
    lossValidation = compose('%.4f',lossValidation);

    accValidation = composePadAccuracy(accValidation);
    accValidationRGB = composePadAccuracy(accValidationRGB);
    accValidationFlow = composePadAccuracy(accValidationFlow);

    accVal = join([accValidation,accValidationRGB,accValidationFlow], " : ");

    lossTrain = gather(extractdata(lossTrain));
    lossTrain = compose('%.4f',lossTrain);

    accTrain = composePadAccuracy(accTrain);
    accTrainRGB = composePadAccuracy(accTrainRGB);
    accTrainFlow = composePadAccuracy(accTrainFlow);

    accTrain = join([accTrain,accTrainRGB,accTrainFlow], " : ");
    learnRate = compose('%.13f',learnRate);

    disp("| " + ...
        pad(string(epoch),5,'both') + " | " + ...
        pad(string(iteration),9,'both') + " | " + ...
        pad(string(D),12,'both') + " | " + ...
        pad(string(accTrain),26,'both') + " | " + ...
        pad(string(accVal),26,'both') + " | " + ...
        pad(string(lossTrain),10,'both') + " | " + ...
        pad(string(lossValidation),10,'both') + " | " + ...
        pad(string(learnRate),13,'both') + " | " + ...
        pad(string(trainTime),10,'both') + " | " + ...
        pad(string(validationTime),15,'both') + " |")
end

    function acc = composePadAccuracy(acc)
        acc = compose('%.2f',acc*100) + "%";
        acc = pad(string(acc),6,'left');
    end

end
% endVerboseOutput
% The endVerboseOutput function displays the end of verbose output during training.
function endVerboseOutput(params)
if params.Verbose
    disp("|=======================================================================================================================================================================|")        
end
end
