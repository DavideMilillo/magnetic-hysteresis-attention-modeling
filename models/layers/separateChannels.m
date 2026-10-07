function varargout = separateChannels(X, numOutputs)
% Separate channels of CB or CBT input so each output is a 1B or 1BT array
arguments
    X
    numOutputs = []
end

numChannels = size(X, 1);

if isempty(numOutputs)
    numOutputs = numChannels;
end

channelDist = repmat(numChannels / numOutputs, [1 numOutputs]);
varargout = mat2cell(X, channelDist);
end
