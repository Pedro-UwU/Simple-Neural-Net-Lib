public interface NeuralNetworkInterface {
    double train(double[] input, double[] target);
    double[] query(double[] inputArray);
}
