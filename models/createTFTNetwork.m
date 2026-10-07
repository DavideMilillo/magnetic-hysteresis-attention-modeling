function net = createTFTNetwork(inputNames, ...
    unknownTimeVaryingInputIdx, ...
    knownTimeVaryingInputIdx, ...
    staticInputIdx, ...
    categoricalInputIdx, ...
    numCategories, ...
    numHiddenUnits, ...
    numAttentionHeads, ...
    numPastTimeSteps, ...
    numFutureTimeSteps, ...
    numQuantiles, ...
    opts)

% createTFTNetwork Create a Temporal Transfusion Transformer network.
arguments
    inputNames
    unknownTimeVaryingInputIdx
    knownTimeVaryingInputIdx
    staticInputIdx
    categoricalInputIdx
    numCategories
    numHiddenUnits
    numAttentionHeads
    numPastTimeSteps
    numFutureTimeSteps
    numQuantiles
    opts.DropoutProbability = 0
end

net = dlnetwork();

% Create input layers
numInputs = numel(inputNames);
numChannelsPerInput = 1;

for ii = unknownTimeVaryingInputIdx
    inputName = inputNames(ii);
    inLayer = sequenceInputLayer(numChannelsPerInput, ...
        MinLength=numPastTimeSteps, ...
        Name=inputName);

    net = addLayers(net,inLayer);
end

numTotalTimeSteps = numPastTimeSteps + numFutureTimeSteps;
for ii = knownTimeVaryingInputIdx
    inputName = inputNames(ii);
    inLayer = sequenceInputLayer(numChannelsPerInput, ...
        MinLength=numTotalTimeSteps, ...
        Name=inputName);

    net = addLayers(net,inLayer);
end

for ii = staticInputIdx
    inputName = inputNames(ii);
    inLayer = featureInputLayer(numChannelsPerInput,Name=inputName);
    net = addLayers(net,inLayer);
end

% Embed continuous inputs using linear transformations
continuousInputIdx = setdiff(1:numInputs,categoricalInputIdx);
for ii = continuousInputIdx
    inputName = inputNames(ii);
    embedLayerName = inputName+"_embed";
    embedLayer = fullyConnectedLayer(numHiddenUnits,Name=embedLayerName);
    
    net = addLayers(net,embedLayer);
    net = connectLayers(net,inputName,embedLayerName);
end

embeddingDimension = numHiddenUnits;
for ii = numel(categoricalInputIdx)
    inputIdx = categoricalInputIdx(ii);
    numCats = numCategories(ii);
    inputName = inputNames(inputIdx);
    embedLayerName = inputName+"_embed";
    embedLayer = wordEmbeddingLayer(embeddingDimension,numCats,OOVMode="error",Name=embedLayerName);
    
    net = addLayers(net,embedLayer);
    net = connectLayers(net,inputName,embedLayerName);
end

% Split past and future for known inputs
for ii = knownTimeVaryingInputIdx
    inputName = inputNames(ii);
    embedLayerName = inputName+"_embed";
    splitLayerName = inputName+"_split";
    splitLayer = functionLayer(@(x) deal(x(:,:,1:numPastTimeSteps),x(:,:,numPastTimeSteps+1:end)),NumOutputs=2,Name=splitLayerName);
    
    net = addLayers(net,splitLayer);
    net = connectLayers(net,embedLayerName,splitLayerName);
end

unknownInputNames = inputNames(unknownTimeVaryingInputIdx);
knownInputNames = inputNames(knownTimeVaryingInputIdx);
pastLayerNames = [unknownInputNames+"_embed",knownInputNames+"_split/out1"];
futureLayerNames = knownInputNames+"_split/out2";

% Static variable selection
numStaticInputs = numel(staticInputIdx);
staticVarSelectLayer = variableSelectionNetworkLayer(numStaticInputs, ...
    numHiddenUnits, ...
    DropoutProbability=opts.DropoutProbability, ...
    HasScoresOutput=true, ...
    Name="static_varselect");
net = addLayers(net,staticVarSelectLayer);

for ii = 1:numStaticInputs
    inputIdx = staticInputIdx(ii);
    embedLayerName = inputNames(inputIdx)+"_embed";
    varselectInputName = "static_varselect/in"+ii;
    net = connectLayers(net,embedLayerName,varselectInputName);
end

% Process outputs of static variable selection network with four separate
% GRNs
staticContextForVariableSelectionLayer = grnNetworkLayer(numHiddenUnits, ...
    DropoutProbability=opts.DropoutProbability, ...
    Name="static_context_varselect");
net = addLayers(net, staticContextForVariableSelectionLayer);
net = connectLayers(net, "static_varselect/out", "static_context_varselect");

staticContextForHiddenStateLayer = grnNetworkLayer(numHiddenUnits, ...
    DropoutProbability=opts.DropoutProbability, ...
    Name="static_context_hidden_state");
net = addLayers(net, staticContextForHiddenStateLayer);
net = connectLayers(net, "static_varselect/out", "static_context_hidden_state");

staticContextForCellStateLayer = grnNetworkLayer(numHiddenUnits, ...
    DropoutProbability=opts.DropoutProbability, ...
    Name="static_context_cell_state");
net = addLayers(net, staticContextForCellStateLayer);
net = connectLayers(net, "static_varselect/out", "static_context_cell_state");

staticContextForEnrichmentLayer = grnNetworkLayer(numHiddenUnits, ...
    DropoutProbability=opts.DropoutProbability, ...
    Name="static_context_enrich");
net = addLayers(net, staticContextForEnrichmentLayer);
net = connectLayers(net, "static_varselect/out", "static_context_enrich");

% Past variable selection
numPastInputs = numel(pastLayerNames);
pastVarSelectLayer = variableSelectionNetworkLayer(numPastInputs, ...
    numHiddenUnits, ...
    DropoutProbability=opts.DropoutProbability, ...
    HasScoresOutput=true, ...
    HasContextInput=true, ...
    Name="past_varselect");
net = addLayers(net,pastVarSelectLayer);
for ii = 1:numPastInputs
    varselectInputName = "past_varselect/in"+ii;
    net = connectLayers(net,pastLayerNames(ii),varselectInputName);
end
net = connectLayers(net, "static_context_varselect", "past_varselect/grn_varselect/context_in");

% Future variable selection
numFutureInputs = numel(futureLayerNames);
futureVarSelectLayer = variableSelectionNetworkLayer(numFutureInputs, ...
    numHiddenUnits, ...
    DropoutProbability=opts.DropoutProbability, ...
    HasScoresOutput=true, ...
    HasContextInput=true, ...
    Name="future_varselect");
net = addLayers(net,futureVarSelectLayer);
for ii = 1:numFutureInputs
    varselectInputName = "future_varselect/in"+ii;
    net = connectLayers(net,futureLayerNames(ii),varselectInputName);
end
net = connectLayers(net, "static_context_varselect", "future_varselect/grn_varselect/context_in");

% Concatenate outputs of past and future variable selection networks along 
% the time dimension
varselectConcatLayer = functionLayer(@(x,y) cat(3,x,y),Name="concat_varselect");
net = addLayers(net,varselectConcatLayer);
net = connectLayers(net,"past_varselect/out","concat_varselect/in1");
net = connectLayers(net,"future_varselect/out","concat_varselect/in2");

% LSTM
lstmEncoderLayer = lstmLayer(numHiddenUnits,HasStateInputs=true,HasStateOutputs=true,Name="lstm_encoder");
net = addLayers(net,lstmEncoderLayer);
net = connectLayers(net,"past_varselect/out","lstm_encoder/in");
net = connectLayers(net,"static_context_hidden_state","lstm_encoder/hidden");
net = connectLayers(net,"static_context_cell_state","lstm_encoder/cell");

lstmDecoderLayer = lstmLayer(numHiddenUnits,HasStateInputs=true,Name="lstm_decoder");
net = addLayers(net,lstmDecoderLayer);
net = connectLayers(net,"future_varselect/out","lstm_decoder/in");
net = connectLayers(net,"lstm_encoder/hidden","lstm_decoder/hidden");
net = connectLayers(net,"lstm_encoder/cell","lstm_decoder/cell");

lstmConcatLayer = functionLayer(@(x,y) cat(3,x,y),Name="concat_lstm");
net = addLayers(net,lstmConcatLayer);
net = connectLayers(net,"lstm_encoder/out","concat_lstm/in1");
net = connectLayers(net,"lstm_decoder","concat_lstm/in2");

% Gated skip connection
skipLayers = [gluNetworkLayer(numHiddenUnits,DropoutProbability=opts.DropoutProbability,Name="lstm_gate")
    additionLayer(2,Name="lstm_skip_add")
    layerNormalizationLayer(Name="lstm_skip_norm")];
net = addLayers(net,skipLayers);
net = connectLayers(net,"concat_lstm","lstm_gate");
net = connectLayers(net,"concat_varselect","lstm_skip_add/in2");

% Static enrichment
enrichmentLayer = grnNetworkLayer(numHiddenUnits, ...
    DropoutProbability=opts.DropoutProbability, ...
    HasContextInput=true, ...
    Name="grn_enrich");
net = addLayers(net,enrichmentLayer);
net = connectLayers(net,"lstm_skip_norm","grn_enrich/seq_in");
net = connectLayers(net,"static_context_enrich","grn_enrich/context_in");

% Interpretable multihead attention
attentionLayer = interpretableSelfAttentionNetworkLayer(numAttentionHeads, ...
    numHiddenUnits, ...
    AttentionMask="causal", ...
    HasScoresOutput=true, ...
    DropoutProbability=opts.DropoutProbability, ...
    Name="attn");
net = addLayers(net,attentionLayer);
net = connectLayers(net,"grn_enrich","attn");

% Gated skip connection
skipLayers = [gluNetworkLayer(numHiddenUnits,DropoutProbability=opts.DropoutProbability,Name="attn_gate")
    additionLayer(2,Name="attn_skip_add")
    layerNormalizationLayer(Name="attn_skip_norm")];
net = addLayers(net,skipLayers);
net = connectLayers(net,"attn/out","attn_gate");
net = connectLayers(net,"grn_enrich","attn_skip_add/in2");

% Output processing
finalGRNLayer = grnNetworkLayer(numHiddenUnits, ...
    DropoutProbability=opts.DropoutProbability, ...
    Name="grn_out");
net = addLayers(net,finalGRNLayer);
net = connectLayers(net,"attn_skip_norm","grn_out");

% Final skip connection
skipLayers = [gluNetworkLayer(numHiddenUnits,DropoutProbability=opts.DropoutProbability,Name="out_gate")
    additionLayer(2,Name="out_skip_add")
    layerNormalizationLayer(Name="out_skip_norm")];
net = addLayers(net,skipLayers);
net = connectLayers(net,"grn_out","out_gate");
net = connectLayers(net,"lstm_skip_norm","out_skip_add/in2");

% Pick out future time steps
futureLayer = functionLayer(@(x) x(:,:,numPastTimeSteps+1:end), ...
    Name="splitFuture");
net = addLayers(net,futureLayer);
net = connectLayers(net,"out_skip_norm","splitFuture");

% Separate into quantiles
quantileLayer = fullyConnectedLayer(numQuantiles,Name="quantile_out");
net = addLayers(net,quantileLayer);
net = connectLayers(net,"splitFuture","quantile_out");

% Arrange the inputs to be in the same order as the inputNames argument
net.InputNames = inputNames;

net.OutputNames = "quantile_out";
end
