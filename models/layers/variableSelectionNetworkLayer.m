function layer = variableSelectionNetworkLayer(numInputs, numHiddenUnits, opts)
% variableSelectionNetworkLayer    Network layer implementing
% variable selection for inputs passed as CB or CBT arrays.
% Layer has one input for each variable. Variables must have the same
% embedding dimension.

arguments
    numInputs
    numHiddenUnits
    opts.DropoutProbability = 0
    opts.HasContextInput = false
    opts.HasScoresOutput = false
    opts.Name = "varselect"
end

if numInputs == 1
    % Special case for single input because there is no need for
    % combination of inputs
    scoresBranch = [identityLayer(Name="in1")
        grnNetworkLayer(numHiddenUnits, ...
        NumOutputChannels=numInputs, ...
        DropoutProbability=opts.DropoutProbability, ...
        HasContextInput=opts.HasContextInput, ...
        Name="grn_varselect")
        softmaxLayer(Name="scores")
        multiplicationLayer(2,Name="out")];
    subnet = dlnetwork(scoresBranch,Initialize=false);

    subnet = addLayers(subnet,grnNetworkLayer(numHiddenUnits, ...
        DropoutProbability=opts.DropoutProbability, ...
        Name="grn"));
    subnet = connectLayers(subnet,"in1","grn");
    subnet = connectLayers(subnet,"grn","out/in2");
else
    scoresBranch = [depthConcatenationLayer(numInputs, Name="input_cat")
        grnNetworkLayer(numHiddenUnits, ...
        NumOutputChannels=numInputs, ...
        DropoutProbability=opts.DropoutProbability, ...
        HasContextInput=opts.HasContextInput, ...
        Name="grn_varselect")
        softmaxLayer(Name="scores")
        functionLayer(@(x) separateChannels(x),NumOutputs=numInputs,Formattable=true,Name="separate")];
    subnet = dlnetwork(scoresBranch,Initialize=false);

    subnet = addLayers(subnet,additionLayer(numInputs,Name="out"));

    for ii = 1:numInputs
        inLayerName = "in"+ii;
        subnet = addLayers(subnet,identityLayer(Name=inLayerName));
        subnet = connectLayers(subnet,inLayerName,"input_cat/in"+ii);

        grnLayerName = "grn"+ii;
        subnet = addLayers(subnet,grnNetworkLayer(numHiddenUnits, ...
            DropoutProbability=opts.DropoutProbability, ...
            Name=grnLayerName));
        subnet = connectLayers(subnet,inLayerName,grnLayerName);

        multLayerName = "mult"+ii;
        subnet = addLayers(subnet,multiplicationLayer(2,Name=multLayerName));
        subnet = connectLayers(subnet,"separate/out"+ii,multLayerName+"/in1");
        subnet = connectLayers(subnet,grnLayerName,multLayerName+"/in2");
        subnet = connectLayers(subnet,multLayerName,"out/in"+ii);
    end
end

if opts.HasContextInput
    % Reorder the input names so context is last
    subnet.InputNames = ["in"+(1:numInputs) "grn_varselect/context_in"];
end

if opts.HasScoresOutput
    subnet.OutputNames = ["out" "scores"];
end

layer = networkLayer(subnet,Name=opts.Name);
end
