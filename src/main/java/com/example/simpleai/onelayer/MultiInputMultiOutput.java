package com.example.simpleai.onelayer;

public class MultiInputMultiOutput {
    private static final double ALPHA = 0.001;

    private static final double[][] INPUT = {
            {8.5, 9.5, 9.0, 9.9},
            {0.65, 0.8, 0.8, 0.9},
            {1.2, 1.3, 0.5, 1.0}};

    // {inj, wins, sad}
    private static final double[][] OUTPUT = {
            {0.1, 0.0, 0.0, 0.1},
            {1, 1, 0, 1},
            {0.1, 0.0, 0.1, 0.2}};
    private static final double[][] ACTUAL = new double[3][4];

    // {{score, win/loose ratio, number of fans} inj, wins, sad}
    private static final double[][] WEIGHTS = {
            {0.1, 0.1, -0.3},
            {0.1, 0.2, 0.0},
            {0.0, 1.3, 0.1}};

    public static void main(String[] args) {

        for (int e = 0; e < 10_000; e++) {
            for (int i = 0; i < INPUT[0].length; i++) {
                double[] inp = {INPUT[0][i], INPUT[1][i], INPUT[2][i]};

                double[] pred = calcPred(inp, WEIGHTS);

                double[] delta = new double[inp.length];
                for (int j = 0; j < delta.length; j++) {
                    delta[j] = pred[j] - OUTPUT[j][i];
                }

                double[][] weightDeltas = calcDeltas(inp, delta);
                for (int k = 0; k < weightDeltas.length; k++) {
                    for (int j = 0; j < weightDeltas[k].length; j++) {
                        WEIGHTS[k][j] -= weightDeltas[k][j];
                    }
                }
                System.out.println();
            }

            printWeigths();
            printExpected();
            printActual();
        }
    }

    private static double[] calcPred(double[] input, double[][] weights) {
        if (weights.length != input.length) {
            return new double[weights.length];
        }

        double[] res = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            res[i] = matrixMultiply(input, weights[i]);
        }
        return res;
    }

    private static double matrixMultiply(double[] input, double[] weight) {
        double res = 0;
        for (int i = 0; i < weight.length; i++) {
            res += input[i] * weight[i];
        }
        return res;
    }

    private static double[][] calcDeltas(double[] inp, double[] delta) {
        double[][] deltas = new double[inp.length][delta.length];
        for (int i = 0; i < inp.length; i++) {
            for (int j = 0; j < delta.length; j++) {
                deltas[j][i] = ALPHA * inp[i] * delta[j]; // deltas[i][j] ???
            }
        }
        return deltas;
    }



    private static void printActual() {
        System.out.println("Actual:");
        for (int k = 0; k < OUTPUT.length; k++) {
            for (int j = 0; j < OUTPUT[k].length; j++) {
                System.out.print(String.format("%.3f",
                                 (INPUT[0][j] * WEIGHTS[k][0]
                                + INPUT[1][j] * WEIGHTS[k][1]
                                + INPUT[2][j] * WEIGHTS[k][2])) + " ");
            }
            System.out.println();
        }
    }

    private static void printExpected() {
        System.out.println("True:");
        for (int k = 0; k < OUTPUT.length; k++) {
            for (int j = 0; j < OUTPUT[k].length; j++) {
                System.out.print(OUTPUT[k][j] + " ");
            }
            System.out.println();
        }
    }

    private static void printWeigths() {
        System.out.println("Calculated weights:");
        for (int k = 0; k < WEIGHTS.length; k++) {
            for (int j = 0; j < WEIGHTS[k].length; j++) {
                System.out.print(String.format("%.3f", WEIGHTS[k][j]) + " ");
            }
            System.out.println();
        }
    }
}
