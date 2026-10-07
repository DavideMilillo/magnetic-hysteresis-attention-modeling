function layer = interpretableSelfAttentionNetworkLayer(numHeads,numHiddenUnits,opts)
% interpretableSelfAttentionNetworkLayer  NetworkLayer implementing 
% interpretable multi-head self-attention

arguments
    numHeads
    numHiddenUnits
    opts.Name = "interpretableselfattention"
    opts.HasScoresOutput = false
    opts.AttentionMask {mustBeMember(opts.AttentionMask, ["none", "causal"])} = "none"
    opts.DropoutProbability = 0
end

if numHeads == 1
    layer = selfAttentionLayer(numHeads, numHiddenUnits, ...
        OutputSize=numHiddenUnits, ...
        HasScoresOutput=opts.HasScoresOutput, ...
        AttentionMask=opts.AttentionMask, ...
        BiasLearnRateFactor=0, ...
        DropoutProbability=opts.DropoutProbability, ...
        Name=opts.Name);
    return
end

if mod(numHiddenUnits, numHeads) ~= 0
    error("Number of attention heads must divide number of hidden units exactly.");
end

queryBranch = [identityLayer(Name="in")
    fullyConnectedLayer(numHiddenUnits,BiasLearnRateFactor=0,Name="fc_q")
    functionLayer(@(x) separateChannels(x,numHeads),NumOutputs=numHeads,Name="splitheads_q")];

keyBranch = [fullyConnectedLayer(numHiddenUnits,BiasLearnRateFactor=0,Name="fc_k")
    functionLayer(@(x) separateChannels(x,numHeads),NumOutputs=numHeads,Name="splitheads_k")];

numValueChannels = numHiddenUnits / numHeads;

% Value weights are shared between attention heads
valueBranch = fullyConnectedLayer(numValueChannels,BiasLearnRateFactor=0,Name="fc_v");

subnet = dlnetwork(queryBranch,Initialize=false);
subnet = addLayers(subnet,keyBranch);
subnet = connectLayers(subnet,"in","fc_k");

subnet = addLayers(subnet,valueBranch);
subnet = connectLayers(subnet,"in","fc_v");

averagingLayer = networkLayer([additionLayer(numHeads,Name="add"),scalingLayer(Scale=1/numHeads,Name="out")],Name="mean");
subnet = addLayers(subnet,averagingLayer);

if opts.HasScoresOutput
    subnet = addLayers(subnet,functionLayer(@(varargin) dlarray(cat(3,varargin{:}),"UUUB"),NumInputs=numHeads,Formattable=true,Name="scores"));
end

for ii = 1:numHeads
    attnLayerName = "attn_"+ii;
    subnet = addLayers(subnet,attentionLayer(1,AttentionMask=opts.AttentionMask,HasScoresOutput=opts.HasScoresOutput,DropoutProbability=opts.DropoutProbability,Name=attnLayerName));

    if opts.HasScoresOutput
        subnet = connectLayers(subnet,attnLayerName+"/scores","scores/in"+ii);
        attnOutputName = attnLayerName+"/out";
    else
        attnOutputName = attnLayerName;
    end

    subnet = connectLayers(subnet,"splitheads_q/out"+ii,attnLayerName+"/query");
    subnet = connectLayers(subnet,"splitheads_k/out"+ii,attnLayerName+"/key");
    subnet = connectLayers(subnet,"fc_v",attnLayerName+"/value");
    subnet = connectLayers(subnet,attnOutputName,"mean/add/in"+ii);
end

outputBranch = fullyConnectedLayer(numHiddenUnits,BiasLearnRateFactor=0,Name="out");

subnet = addLayers(subnet,outputBranch);
subnet = connectLayers(subnet,"mean","out");

if opts.HasScoresOutput
    % Make sure the outputs are ordered correctly
    subnet.OutputNames = ["out" "scores"];
end

layer = networkLayer(subnet,Name=opts.Name);
end
