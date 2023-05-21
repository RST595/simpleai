package com.example.simpleai.multilayer;

/**
 * We add hidden layer to make correlation in data set,
 * which originally don't have correlation between input and output layers
 */
public class MultiInputSingleOutput {
    private static final double ALPHA = 0.002;
    private static final int HIDDEN_SIZE = 4;
    private static final double[][] INPUT = {
            {1, 0, 1},
            {0, 1, 1},
            {0, 0, 1},
            {1, 1, 1}
    };

    private static final double[] OUTPUT = {1, 1, 0, 0};

    /**
     * WEIGHTS_0_1.length = HIDDEN_SIZE
     * WEIGHTS_0_1[i].length = INPUT[i].length
     */
    private static final double[][] WEIGHTS_0_1 = {
            {-0.95, -0.43, -0.14},
            {-0.66, 0.97, 0.19},
            {0.95, 0.22, -0.43},
            {-0.55, 0.6, 0.99}
    };

    /**
     * WEIGHTS_1_2.length = HIDDEN_SIZE
     */
    private static final double[] WEIGHTS_1_2 = {0.6, 0.55, 0.47, 0.22};

    private static final double[][] ORIGINAL_RANDOM_WEIGHTS_0_1 = new double[WEIGHTS_0_1.length][WEIGHTS_0_1[0].length];
    private static final double[] ORIGINAL_RANDOM_WEIGHTS_1_2 = new double[WEIGHTS_1_2.length];

    public static void main(String[] args) {
        //fillRandomWeights();
        for (int k = 0; k < 60_000; k++) {
            for (int i = 0; i < INPUT.length; i++) {
                double[] input = {INPUT[i][0], INPUT[i][1], INPUT[i][2]};

                double[] predictionInFirstLayer = calcPredMultiToMulti(input, WEIGHTS_0_1);

                relu(predictionInFirstLayer);

                double predResult = calcPredMultiToOne(predictionInFirstLayer, WEIGHTS_1_2);

                // Calculate delta for result -> hidden layer
                double delta = predResult - OUTPUT[i];

                correctWeightsFromHiddenToOutput(delta, predictionInFirstLayer);

                double[] deltaInputToHidden = new double[WEIGHTS_0_1.length];
                for (int j = 0; j < deltaInputToHidden.length; j++) {
                    deltaInputToHidden[j] = delta * WEIGHTS_1_2[j] * (predictionInFirstLayer[j] < 0 ? 0 : 1);
                }
                correctWeightsFromInputToHidden(deltaInputToHidden, INPUT[i]);
            }

            printResults();
        }
        printOriginalRandomWeights();
    }

    /**
     * Correct weight from hidden to output layer.
     */
    private static void correctWeightsFromHiddenToOutput(double delta, double[] predictionInFirstLayer) {
        for (int i = 0; i < predictionInFirstLayer.length; i++) {
            double weightDelta = ALPHA * delta * predictionInFirstLayer[i];
            WEIGHTS_1_2[i] -= weightDelta;
        }
    }

    /**
     * Correct weight from input to hidden layer.
     */
    private static void correctWeightsFromInputToHidden(double[] deltaInputToHidden, double[] input) {
        for (int i = 0; i < deltaInputToHidden.length; i++) {
            for (int j = 0; j < WEIGHTS_0_1[i].length; j++) {
                double weightDelta = ALPHA * deltaInputToHidden[i] * input[j];
                WEIGHTS_0_1[i][j] -= weightDelta;
            }
        }
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
        System.out.println("WEIGHTS_0_1:");
        for (int i = 0; i < WEIGHTS_0_1.length; i++) {
            for (int j = 0; j < WEIGHTS_0_1[i].length; j++) {
                System.out.print(String.format("%.2f", WEIGHTS_0_1[i][j]) + "      ");
            }
            System.out.println();
        }
        System.out.println();
        System.out.println("WEIGHTS_1_2:");
        for (int i = 0; i < WEIGHTS_1_2.length; i++) {
            System.out.print(String.format("%.2f", WEIGHTS_1_2[i]) + " ");
        }
        System.out.println();
        System.out.println();
        System.out.println("RESULT:");
        for (int i = 0; i < OUTPUT.length; i++) {
            System.out.println("ACTUAL: " + OUTPUT[i] + "; PREDICTION: " + calcResult(i) + ";");
        }
        System.out.println("--------------------------------------------------------------------");
    }

    private static String calcResult(int num) {
        double[] inp = INPUT[num];
        double res = 0;
        double[] hid = calcPredMultiToMulti(inp, WEIGHTS_0_1);
        for (int i = 0; i < hid.length; i++) {
            if (hid[i] < 0) {
                continue;
            }
            res += hid[i] * WEIGHTS_1_2[i];
        }

        return String.format("%.2f", res);
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
     * Calculate result, when input has several nodes, and output has only one node
     */
    private static double calcPredMultiToOne(double[] input, double[] weights) {
        double res = 0;
        if (input.length != weights.length) {
            return res;
        }

        for (int i = 0; i < input.length; i++) {
            res += (input[i] * weights[i]);
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
     * Fill all weight with random value -1.0 : 1.0
     */
    private static void fillRandomWeights() {
        for (int i = 0; i < WEIGHTS_0_1.length; i++) {
            for (int j = 0; j < WEIGHTS_0_1[i].length; j++) {
                WEIGHTS_0_1[i][j] = Math.random() * (Math.random() < 0.5 ? -1 : 1);
                ORIGINAL_RANDOM_WEIGHTS_0_1[i][j] = WEIGHTS_0_1[i][j];
            }
        }

        for (int i = 0; i < WEIGHTS_1_2.length; i++) {
            WEIGHTS_1_2[i] = Math.random() * (Math.random() < 0.5 ? -1 : 1);
            ORIGINAL_RANDOM_WEIGHTS_1_2[i] = WEIGHTS_1_2[i];
        }
    }

    private static void printOriginalRandomWeights() {
        System.out.println("ORIGINAL RANDOM WEIGHTS 0_1:");
        for (int i = 0; i < WEIGHTS_0_1.length; i++) {
            for (int j = 0; j < WEIGHTS_0_1[i].length; j++) {
                System.out.print(String.format("%.2f", ORIGINAL_RANDOM_WEIGHTS_0_1[i][j]) + "   ");
            }
            System.out.println();
        }

        System.out.println("ORIGINAL RANDOM WEIGHTS 1_2:");
        for (int i = 0; i < WEIGHTS_1_2.length; i++) {
            System.out.print(String.format("%.2f", ORIGINAL_RANDOM_WEIGHTS_1_2[i]) + "   ");
        }
        System.out.println();
    }

    private static void printOriginalWeights() {
        System.out.println("ORIGINAL WEIGHTS 0_1:");
        for (int i = 0; i < WEIGHTS_0_1.length; i++) {
            for (int j = 0; j < WEIGHTS_0_1[i].length; j++) {
                System.out.print(String.format("%.2f", WEIGHTS_0_1[i][j]) + "   ");
            }
            System.out.println();
        }

        System.out.println("ORIGINAL WEIGHTS 1_2:");
        for (int i = 0; i < WEIGHTS_1_2.length; i++) {
            System.out.print(String.format("%.2f", WEIGHTS_1_2[i]) + "   ");
        }
        System.out.println();
    }
}
