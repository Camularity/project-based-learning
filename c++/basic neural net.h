//Neural Net Test

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;

struct Connection
{
    
    double weight;
    double deltaWeight;
    
};

class Neuron;

typedef vector<Neuron> Layer;

// ********** class Neuron **********

class Neuron
{
    
    public:
        Neuron(unsigned numOutputs, unsigned myIndex);
        void setOutputVal(double val) { m_outputVal = val; }
        double getOutputVal(void) const { return m_outputVal; }
        void feedForward(const Layer &prevLayer);
        void calcOutputGradients(double tagetVals);
        void calcHiddenGradients(const Layer &nextLayer);
        void updateInputWeights(Layer &prevLayer);
    
    private:
        static double eta; // [0.0 .. 1.0] overall net training private
        static double alpha; // [0.0 .. n] multiplier of last weight change (momentum)
        static double transferFunction(double x);
        static double transferFunctionDerivative(double x);
        static double randomWeight(void) { return rand() / double(RAND_MAX); }
        double sumDOW(const Layer &nextLayer) const;
        double m_outputVal;
        vector<Connection> m_outputWeights;
        unsigned m_myIndex;
        double m_gradient;
};

double Neuron::eta = 0.15; // overall training rate
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight

void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // of the neurons in the preceding layer
    
    for (unsigned n = 0; n < prevLayer.size(); ++n)
    {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        
        double newDeltaWeight = 
                eta
                * neuron.getOutputVal()
                * m_gradient
                + alpha * oldDeltaWeight;
        
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;
    
    //sum our contributions of the errors at the nodes we feedForward
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
    {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVals)
{
    double delta = targetVals - m_outputVal;
    m_gradient = delta * transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
    // tanh - output range (-1.0 .. 1.0)
    // Make sure to set input data to the range of the transferFunction (sigmoid, tanh etc)
    return tanh(x);
    
}

double Neuron::transferFunctionDerivative(double x) 
{
    // tanh derivative (could be any math formula or c++ function that gets the derivative of the transferfunction set)
    // simple approximate version of d/dx tanh is below
    return 1 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;
    
    // Sum the previous layer's outputs (which are this local 'this' inputs)
    // Include the bias neuron from the previous layer
    
    for (unsigned n = 0; n < prevLayer.size(); ++n )
    {
        sum += prevLayer[n].getOutputVal() * 
                prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    
    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) 
{
    
    for (unsigned c = 0; c< numOutputs; ++c) 
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
        
    }
    
    m_myIndex = myIndex;
}

// ********** class Net **********

class Net
{
    
    public:
        Net(const vector<unsigned> &topology);
        void feedForward(const vector<double> &inputVals);
        void backProp(const vector<double> &tagetVals);
        void getResults(vector<double> &resultsVals) const;
        
        
    private:
        vector<Layer> m_layers; // m_layers[layernum][NeuronNum]
        double m_error;
        double m_recentAverageError;
        double m_recentAverageSmoothingFactor;
    
    
};

void Net::getResults(vector<double> &resultsVals) const
{
    resultsVals.clear();
    
    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
    {
        resultsVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void Net::backProp(const vector<double> &tagetVals)
{
    // Calc overall net error (Root Mean Square of output neuron erros)
    
    Layer &outputLayer = m_layers.back();
    //Use a middle value for initial error (no error)
    m_error = 0.0;
    
    // Loop outputs not including bias neuron
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        double delta = tagetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    
    m_error /= outputLayer.size() - 1; // average error across neurons
    m_error = sqrt(m_error); //resultant RMS
    
    //implement a recent average measurment
    m_recentAverageError = 
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
        / (m_recentAverageSmoothingFactor + 1.0);
    
    // Calc output layer gradients (derivative)
    
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        outputLayer[n].calcOutputGradients(tagetVals[n]);
    }
    
    // Calc gradiends on hidden layers starting with the right most and work back
    // -1 would give output layer, -2 gives next back from output layer (due 0 being first array size 6 last index 5(output) )
    
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];
        
        for (unsigned n = 0; n < hiddenLayer.size(); ++n)
        {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }
    
    // for all layers from outputs to first hidden layer,
    // update connection weights
    
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
    {
        Layer &layer = m_layers[layerNum];
        Layer &prevlayer = m_layers[layerNum - 1];
        
        // - 1 for not including bias
        for (unsigned n = 0; n < layer.size() - 1; ++n)
        {
            layer[n].updateInputWeights(prevlayer);
        }
    }
    
}

void Net::feedForward(const vector<double> &inputVals)
{
    //error check that there is a feed forward to do
    assert(inputVals.size() == m_layers[0].size() - 1);
    
    //Assign i(latch) the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); ++i)
    {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
    
    //Start at layer[1] to skip the initial layer[0] as this is the input layer, we need to feed the hidden layers
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
    {
        //setup prevlayer reference to enable the neuron math to calculate the previous layers values
        Layer &prevLayer = m_layers[layerNum - 1];
        
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n)
        {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
    
    
}

Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        m_layers.push_back(Layer());
        //outputs = if last layer then no outputs else the size of the next layer
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        //we have made a new layer, now fill it with neurons, and
        // add a bias neuron layer : include <= to add one extra above size for the bias neuron per layer
        
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
        {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "Made a Neuron!" << endl;
            
        }
        
        //force bias neuron value to 1.0
        m_layers.back().back().setOutputVal(1.0);
    }
}

int main()
{
    
    //e.g., {3,2,1}
    vector<unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);
    
    Net myNet(topology);
    
    vector<double> inputVals;
    myNet.feedForward(inputVals);
    
    vector<double> targetVals;
    myNet.backProp(targetVals);
    
    vector<double> resultsVals;
    myNet.getResults(resultsVals);
    
    
    
}
