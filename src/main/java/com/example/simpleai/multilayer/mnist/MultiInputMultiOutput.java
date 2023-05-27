package com.example.simpleai.multilayer.mnist;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Paths;
import java.security.InvalidParameterException;
import java.util.stream.Stream;

/**
 * 3 layers model for MNIST analysis
 */
public class MultiInputMultiOutput {
    private static final double ALPHA = 0.00002;
    private static final double CONSTANT_START_WEIGHT = 0.02;
    private static final int INPUT_LAYER_SIZE = 784;
    private static final int HIDDEN_LAYER_SIZE = 800;
    private static final int OUTPUT_LAYER_SIZE = 10;
    private static final int NUMBER_OF_INPUT_SAMPLES = 10000;
    private static final int NUMBER_TRAINING_REPEATS = 10;
    private static final double[][] INPUT = new double[NUMBER_OF_INPUT_SAMPLES][INPUT_LAYER_SIZE];
    private static final double[][] OUTPUT = new double[NUMBER_OF_INPUT_SAMPLES][OUTPUT_LAYER_SIZE];

    /**
     * WEIGHTS_0_1.length = HIDDEN_SIZE
     * WEIGHTS_0_1[i].length = INPUT[i].length
     */
    private static final double[][] WEIGHTS_0_1 = new double[HIDDEN_LAYER_SIZE][INPUT_LAYER_SIZE];

    /**
     * WEIGHTS_1_2.length = OUTPUT SIZE
     * WEIGHTS_1_2[i].length = HIDDEN SIZE
     */
    private static final double[][] WEIGHTS_1_2 = new double[OUTPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];

    public static void main(String[] args) throws IOException {
        fillWeights(true);
        printStartRandomWeightsToFile("src/main/java/com/example/simpleai/multilayer/mnist/start_random_weights/weights.txt");
        fillData();
        for (int k = 0; k < NUMBER_TRAINING_REPEATS; k++) {

            for (int i = 0; i < INPUT.length; i++) {
                double[] predictionInFirstLayer = calcPredMultiToMulti(INPUT[i], WEIGHTS_0_1);

                relu(predictionInFirstLayer);

                double[] predResult = calcPredMultiToMulti(predictionInFirstLayer, WEIGHTS_1_2);

                // Calculate delta for result -> hidden layer
                double[] delta = calculateDeltaForOutputLayer(predResult, OUTPUT[i]);

                correctWeightsFromHiddenToOutput(delta, predictionInFirstLayer, WEIGHTS_1_2);

                double[] deltaInputToHidden = new double[HIDDEN_LAYER_SIZE];
                for (int j = 0; j < deltaInputToHidden.length; j++) {
                    for (int t = 0; t < delta.length; t++) {
                        deltaInputToHidden[j] = delta[t] * WEIGHTS_1_2[t][j] * (predictionInFirstLayer[j] < 0 ? 0 : 1);
                    }
                }

                correctWeightsFromHiddenToOutput(deltaInputToHidden, INPUT[i], WEIGHTS_0_1);
            }
        }

        printResults();
    }

    private static double[] calculateDeltaForOutputLayer(double[] predResult, double[] realResult) {
        if (predResult.length != realResult.length) {
            throw new InvalidParameterException("Output prediction size not equals to output real size");
        }

        double[] delta = new double[predResult.length];
        for (int j = 0; j < predResult.length; j++) {
            delta[j] = predResult[j] - realResult[j];
        }
        return delta;
    }

    /**
     * Fill input and output arrays with data from file
     */
    private static void fillData() throws IOException {
        File fileWithTrainingData = Paths.get("src/main/java/com/example/simpleai/multilayer/mnist/input_data/trainingMnistOne.json").toFile();
        MnistMatrix[] mnistMatrix = new ObjectMapper().readValue(fileWithTrainingData, MnistMatrix[].class);
        File secondFileWithTrainingData = Paths.get("src/main/java/com/example/simpleai/multilayer/mnist/input_data/trainingMnistTwo.json").toFile();
        MnistMatrix[] secondMnistMatrix = new ObjectMapper().readValue(secondFileWithTrainingData, MnistMatrix[].class);

        MnistMatrix[] mergedMnist = Stream.of(mnistMatrix, secondMnistMatrix)
                .flatMap(Stream::of)
                .toArray(MnistMatrix[]::new);
        if (mergedMnist.length < NUMBER_OF_INPUT_SAMPLES) {
            throw new InvalidParameterException("INVALID INPUT DATA SIZE: number of data in file not equals to CONSTANT");
        }

        for (int i = 0; i < NUMBER_OF_INPUT_SAMPLES; i++) {
            fillOutputValue(OUTPUT[i], mergedMnist[i].getLabel());
            fillInputValue(INPUT[i], mergedMnist[i].getData());
        }
    }

    private static void fillOutputValue(double[] outputs, int value) {
        for (int i = 0; i < outputs.length; i++) {
            if (i == value) {
                outputs[i] = 1;
            }
        }
    }

    private static void fillInputValue(double[] input, int[][] data) {
        if (data.length * data[0].length != input.length) {
            throw new InvalidParameterException("INVALID INPUT DATA SIZE: number of data array in file not equals to INPUT array size");
        }

        int row = 0;
        int column = 0;

        for (int i = 0; i < input.length; i++) {
            input[i] = data[row][column];
            column++;
            if (column % 28 == 0) {
                column = 0;
                row++;
            }
        }
    }

    /**
     * Correct weight from hidden to output layer.
     */
    private static void correctWeightsFromHiddenToOutput(double[] delta, double[] predictionInFirstLayer, double[][] weights) {
        double[][] weightDeltas = calcDeltas(delta, predictionInFirstLayer);
        for (int k = 0; k < weightDeltas.length; k++) {
            for (int j = 0; j < weightDeltas[k].length; j++) {
                weights[k][j] -= weightDeltas[k][j];
            }
        }
    }

    private static double[][] calcDeltas(double[] delta, double[] inp) {
        double[][] deltas = new double[delta.length][inp.length];
        for (int i = 0; i < inp.length; i++) {
            for (int j = 0; j < delta.length; j++) {
                deltas[j][i] = ALPHA * inp[i] * delta[j];
            }
        }
        return deltas;
    }

    /**
     * ReLU. Make part dependence. We depend on input, only if it is more than 0;
     */
    private static void relu(double[] layer) {
        for (int i = 0; i < layer.length; i++) {
            if (layer[i] < 0) {
                layer[i] = 0;
            }
        }
    }

    private static void printResults() {
        for (int i = 0; i < NUMBER_OF_INPUT_SAMPLES; i += 100) {
            double[] real = OUTPUT[i];
            double[] pred = calculateOutput(i);

            printPair(real, pred);
        }
    }

    private static void printPair(double[] real, double[] pred) {
        System.out.println("REAL IS:");
        for (int i = 0; i < real.length; i++) {
            System.out.print(String.format("%.2f", real[i]) + " ");
        }
        System.out.println();
        System.out.println("PREDICTION IS:");
        for (int i = 0; i < pred.length; i++) {
            System.out.print(String.format("%.2f", pred[i]) + " ");
        }
        System.out.println();
    }

    /**
     * Calculate result, when input has several nodes, and output has several nodes
     */
    private static double[] calcPredMultiToMulti(double[] input, double[][] weights) {
        if (weights[0].length != input.length) {
            return new double[weights.length];
        }

        double[] res = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            res[i] = arrayMultiply(input, weights[i]);
        }
        return res;
    }

    /**
     * Multiply one array to another
     */
    private static double arrayMultiply(double[] input, double[] weight) {
        double res = 0;
        for (int i = 0; i < weight.length; i++) {
            res += input[i] * weight[i];
        }
        return res;
    }

    /**
     * Fill all weight with random value -1.0 : 1.0 or with CONSTANT
     */
    private static void fillWeights(boolean random) {
        for (int i = 0; i < WEIGHTS_0_1.length; i++) {
            for (int j = 0; j < WEIGHTS_0_1[i].length; j++) {
                WEIGHTS_0_1[i][j] = random ? getRandomFromMinusOneToOne() : CONSTANT_START_WEIGHT;
            }
        }

        for (int i = 0; i < WEIGHTS_1_2.length; i++) {
            for (int j = 0; j < WEIGHTS_1_2[i].length; j++) {
                WEIGHTS_1_2[i][j] = random ? getRandomFromMinusOneToOne() : CONSTANT_START_WEIGHT;
            }
        }
    }

    private static double getRandomFromMinusOneToOne() {
        return Math.random() * (Math.random() < 0.5 ? -0.001 : 0.001);
    }

    /**
     * Calculate output after training
     */
    private static double[] calculateOutput(int i) {
        double[] predictionInFirstLayer = calcPredMultiToMulti(INPUT[i], WEIGHTS_0_1);

        relu(predictionInFirstLayer);

        return calcPredMultiToMulti(predictionInFirstLayer, WEIGHTS_1_2);
    }

    private static void printStartRandomWeightsToFile(String path) {
        try (PrintWriter printWriter = new PrintWriter(new FileWriter(path))) {
            printWriter.print("ALPHA: " + ALPHA + "\n");
            printWriter.print("WEIGHTS FROM INPUT TO HIDDEN:\n");
            for (int i = 0; i < WEIGHTS_0_1.length; i++) {
                for (int j = 0; j < WEIGHTS_0_1[i].length; j++) {
                    printWriter.print(String.format("%.6f", WEIGHTS_0_1[i][j]) + " ");
                }
                printWriter.print("\n");
            }
            printWriter.print("\n");
            printWriter.print("WEIGHTS FROM HIDDEN TO OUTPUT:\n");
            for (int i = 0; i < WEIGHTS_1_2.length; i++) {
                for (int j = 0; j < WEIGHTS_1_2[i].length; j++) {
                    printWriter.print(String.format("%.6f", WEIGHTS_1_2[i][j]) + " ");
                }
                printWriter.print("\n");
            }
        } catch (IOException e) {
            System.out.println("Can't write weights to file: " + path + "; " + e.getMessage());
        }
    }
}
