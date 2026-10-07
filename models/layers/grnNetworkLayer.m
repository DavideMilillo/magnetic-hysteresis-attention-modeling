function layer = grnNetworkLayer(numHiddenUnits, opts)
% grnNetworkLayer  Network layer implementing a Gated Residual Network
arguments
    numHiddenUnits
    opts.NumOutputChannels
    opts.DropoutProbability = 0
    opts.HasContextInput = false
    opts.Name = "grn"
end

includeFCSkip = isfield(opts,"NumOutputChannels");

if ~isfield(opts, "NumOutputChannels")
    opts.NumOutputChannels = numHiddenUnits;
end

layers = [identityLayer(Name="seq_in")
    fullyConnectedLayer(numHiddenUnits)
    eluLayer];

if opts.HasContextInput
    layers = [layers
        additionLayer(2,Name="add_context")];
end

layers = [layers
    fullyConnectedLayer(numHiddenUnits)
    gluNetworkLayer(opts.NumOutputChannels,DropoutProbability=opts.DropoutProbability)
    additionLayer(2,Name="add_skip")];

subnet = dlnetwork(layers,Initialize=false);

if opts.HasContextInput
    subnet = addLayers(subnet,fullyConnectedLayer(numHiddenUnits,BiasLearnRateFactor=0,Name="context_in"));
    subnet = connectLayers(subnet,"context_in","add_context/in2");
end

if includeFCSkip
    % Add a fully connected layer to make sure the sizes match either side
    % of the skip connection
    subnet = addLayers(subnet,fullyConnectedLayer(opts.NumOutputChannels,Name="fc_skip"));
    subnet = connectLayers(subnet,"seq_in","fc_skip");
    subnet = connectLayers(subnet,"fc_skip","add_skip/in2");
else
    subnet = connectLayers(subnet,"seq_in","add_skip/in2");
end
subnet = addLayers(subnet,layerNormalizationLayer(Name="layernorm"));
subnet = connectLayers(subnet,"add_skip","layernorm");
layer = networkLayer(subnet, Name=opts.Name);
end
