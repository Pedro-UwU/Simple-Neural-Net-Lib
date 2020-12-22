import org.json.JSONArray;
import org.json.JSONObject;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDRandom;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Stream;

public class NeuralNetwork implements NeuralNetworkInterface {

    //TODO create an interface to choose the activation function
    //TODO add bias

    private int iNodes, hNodes, oNodes;
    private double learningRate;
    private INDArray Wih, Who;
    private Function<INDArray, INDArray> activation = Transforms::sigmoid;
    private Function<INDArray, INDArray> activationDerivative = Transforms::sigmoidDerivative;

    public NeuralNetwork(int iNodes, int hNodes, int oNodes, double learningRate) {
        this.iNodes = iNodes + 1;
        this.hNodes = hNodes + 1;
        this.oNodes = oNodes;
        this.learningRate = learningRate;
        NDRandom generator = new NDRandom();
        Wih = generator.normal(0, 1.0/Math.sqrt(this.iNodes), DataType.DOUBLE, this.hNodes, this.iNodes);//.rand(DataType.DOUBLE, this.hNodes, this.iNodes).mul(2).sub(1); //between -1 and 1
        Who = generator.normal(0, 1.0/Math.sqrt(this.hNodes), DataType.DOUBLE, this.oNodes, this.hNodes);
    }


    @Override
    public double train(double[] inputArray, double[] targetArray) {
        INDArray target = Nd4j.create(targetArray, new int[]{oNodes, 1});
        INDArray input = Nd4j.create(addBias(inputArray), new int[]{iNodes,1});

        INDArray hiddenDerivative = activationDerivative.apply(Wih.mmul(input));
        INDArray hidden = activation.apply(Wih.mmul(input));
        hidden.put(hNodes-1, 0, 1); //Added bias

        INDArray outputDerivative = activationDerivative.apply(Who.mmul(hidden));
        INDArray output = activation.apply(Who.mmul(hidden));

        INDArray outputErrors = target.sub(output);
        INDArray hiddenErrors = Who.transpose().mmul(outputErrors);

        //Weight Tweaking
        //INDArray deltaHO = outputErrors.mul((output)).mul((output.mul(-1).add(1))).mmul(hidden.transpose()).mul(learningRate);//.mul(Transforms.sigmoid(output.sub(1).mul(-1))).mmul(hidden.transpose()).mul(learningRate);
        INDArray deltaHO = outputErrors.mul(outputDerivative).mmul(hidden.transpose()).mul(learningRate);
        Who.addi(deltaHO);

        //INDArray deltaIH = hiddenErrors.mul((hidden)).mul((hidden.mul(-1).add(1))).mmul(input.transpose()).mul(learningRate);
        INDArray deltaIH = hiddenErrors.mul(hiddenDerivative).mmul(input.transpose()).mul(learningRate);
        Wih.addi(deltaIH);
        return (double) outputErrors.meanNumber();

    }

    @Override
    public double[] query(double[] inputArray) {
        INDArray input =  Nd4j.create(addBias(inputArray), new int[]{iNodes, 1}); //Column vector
        input.put(iNodes-1, 0, 1);
        INDArray hidden = Transforms.sigmoid(Wih.mmul(input));
        hidden.put(hNodes-1, 0, 1);
        INDArray output = Transforms.sigmoid(Who.mmul(hidden));
        return  output.toDoubleVector();
    }

    private static double[] addBias(double[] array) {
        double[] newArray = new double[array.length + 1];
        System.arraycopy(array, 0, newArray, 0, array.length);
        newArray[array.length] = 1;
        return newArray;
    }

    public static void saveModel(NeuralNetwork nn, String modelName, String filePath) throws IOException {
        JSONObject master = new JSONObject();
        JSONObject shape = new JSONObject();
        shape.put("inputNodes", nn.iNodes);
        shape.put("hiddenNodes", nn.hNodes);
        shape.put("outputNodes", nn.oNodes);
        shape.put("learningRate", nn.learningRate);
        master.put("shape", shape);
        master.put("Wih", nn.Wih.toDoubleMatrix());
        master.put("Who", nn.Who.toDoubleMatrix());
        FileWriter fW = new FileWriter(filePath +"/" + modelName + ".json");
        fW.write(master.toString());
        fW.close();
    }

    public static NeuralNetwork loadModel(String fileName) {
        String nnStr = readLineByLineJava8(fileName);
        if (nnStr.isEmpty()) return null;
        JSONObject json = new JSONObject(nnStr);
        JSONObject shape = json.getJSONObject("shape");
        int iNodes = shape.getInt("inputNodes") - 1;
        int hNodes = shape.getInt("hiddenNodes") - 1;
        int oNodes = shape.getInt("outputNodes");
        double lr = shape.getDouble("learningRate");
        NeuralNetwork nn = new NeuralNetwork(iNodes, hNodes, oNodes, lr);

        JSONArray WihRows = json.getJSONArray("Wih");
        double[][] newWih = new double[hNodes + 1][iNodes + 1];
        for (int i = 0; i < WihRows.length(); i++) {
            JSONArray WihCols = WihRows.getJSONArray(i);
            for (int j = 0; j < WihCols.length(); j++) {
                newWih[i][j] = WihCols.getDouble(j);
            }
        }
        nn.loadWIH(newWih);

        JSONArray WhoRows = json.getJSONArray("Who");
        double[][] newWho = new double[oNodes][hNodes + 1];
        for (int i = 0; i < WhoRows.length(); i++) {
            JSONArray WhoCols = WhoRows.getJSONArray(i);
            for (int j = 0; j < WhoCols.length(); j++) {
                newWho[i][j] = WhoCols.getDouble(j);
            }
        }
        nn.loadWHO(newWho);
        return nn;
    }

    public void setActivationFunction(String name) {
        name = name.toLowerCase();
        switch (name) {
            case "sigmoid": {
                activation = Transforms::sigmoid;
                activationDerivative = Transforms::sigmoidDerivative;
                break;
            }
            default: throw new RuntimeException("Invalid Activation Function");
        }
    }

    public void mutate(double mutProbability, double standardDeviation) {
        //Change the weights from the net with a probability of mutProbability
        //Uses a random number generated from a Gaussian Distribution of mean 0 and std 1
        //Multiplies that number by the standard deviation given
        //Recommendation: use a small value for the standardDeviation
        if (mutProbability > 1 || mutProbability < 0) throw new RuntimeException("Mutation probability must be between 1 and 0");
        for (int i = 0; i < Wih.rows(); i++) {
            for (int j = 0; j < Wih.columns(); j++) {
                if (Math.random() < mutProbability) {
                    double value = Wih.getDouble(i,j);
                    double added = new Random().nextGaussian() * standardDeviation;
                    Wih.put(i, j, value + added);
                }
            }
        }
        for (int i = 0; i < Who.rows(); i++) {
            for (int j = 0; j < Who.columns(); j++) {
                if (Math.random() < mutProbability) {
                    double value = Who.getDouble(i,j);
                    double added = new Random().nextGaussian() * standardDeviation;
                    Who.put(i, j, value + added);
                }
            }
        }
    }

    private static String readLineByLineJava8(String filePath)
    {
        StringBuilder contentBuilder = new StringBuilder();

        try (Stream<String> stream = Files.lines( Paths.get(filePath), StandardCharsets.UTF_8))
        {
            stream.forEach(s -> contentBuilder.append(s).append("\n"));
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }

        return contentBuilder.toString();
    }

    public double[][] getWih() {
        return Wih.toDoubleMatrix();
    }

    public double[][] getWho() {
        return Who.toDoubleMatrix();
    }

    @Override
    public String toString() {
        return "NeuralNetwork: " +
                "iNodes:" + (iNodes-1) +
                ", hNodes: " + (hNodes-1) +
                ", oNodes: " + oNodes +
                ", learningRate: " + learningRate;
    }

    public void loadWIH(double[][] array) throws RuntimeException {
        if (array.length != hNodes) throw new RuntimeException("Cannot copy a " + array.length + " rows matrix into a " + hNodes + " rows matrix");
        if (array[0].length != iNodes) throw new RuntimeException("Cannot copy a " + array[0].length + " columns matrix into a " + iNodes + " columns matrix");
        Wih = Nd4j.create(array);
    }
    public void loadWHO(double[][] array) {
        if (array.length != oNodes) throw new RuntimeException("Cannot copy a " + array.length + " rows matrix into a " + oNodes + " rows matrix");
        if (array[0].length != hNodes) throw new RuntimeException("Cannot copy a " + array[0].length + " columns matrix into a " + hNodes + " columns matrix");
        Who = Nd4j.create(array);
    }
    private void loadWIH(INDArray array) {
        Wih = array.dup();
    }
    private void loadWHO(INDArray array) {
        Who = array.dup();
    }

    public NeuralNetwork copy() {
        NeuralNetwork nn = new NeuralNetwork(this.iNodes-1, this.hNodes-1, this.oNodes, this.learningRate);
        nn.loadWIH(Wih);
        nn.loadWHO(Who);
        return nn;
    }

//    public static void main(String[] args) {
//    }



}
