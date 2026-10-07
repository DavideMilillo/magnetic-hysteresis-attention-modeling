function layer = gluNetworkLayer(numHiddenUnits, opts)
% gluNetworkLayer  Network layer implementing the Gated Linear Unit

arguments
    numHiddenUnits
    opts.DropoutProbability = 0
    opts.Name = "glu"
end

backbone = [dropoutLayer(opts.DropoutProbability,Name="dropout")
    fullyConnectedLayer(numHiddenUnits,Name="fc_act")
    multiplicationLayer(2,Name="mult")];
subnet = dlnetwork(backbone, Initialize=false);

gatingBranch = [fullyConnectedLayer(numHiddenUnits, Name="fc_gate")
    sigmoidLayer(Name="sigmoid")];

subnet = addLayers(subnet,gatingBranch);
subnet = connectLayers(subnet,"dropout","fc_gate");
subnet = connectLayers(subnet,"sigmoid","mult/in2");

layer = networkLayer(subnet,Name=opts.Name);
end
