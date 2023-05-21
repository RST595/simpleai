package com.example.simpleai.onelayer;

public class SingleInputMultiOutput {
    private static final int INPUT_SIZE = 4;
    private static final int NUMBER_OF_OUTS = 3;
    private static final double ALPHA = 0.00000001;
    private static final double[] INPUT = {0.65, 1.0, 1.0, 0.9};
    private static final double[][] OUTPUT = {{0.1, 0.0, 0.0, 0.1}, {1, 1, 0, 1}, {0.1, 0.0, 0.1, 0.2}};
    private static final double[][] ACTUAL = new double[3][4];
    private static final double[] WEIGHTS = {0.3, 0.2, 0.9};

    public static void main(String[] args) {
        for (int k = 0; k < 100_000; k++) {
            for (int i = 0; i < INPUT_SIZE; i++) {
                for (int j = 0; j < NUMBER_OF_OUTS; j++) {
                    double pred = INPUT[i] * WEIGHTS[j];
                    ACTUAL[j][i] = pred;
                    double delta = pred - OUTPUT[j][i];
                    double error = Math.pow(delta, 2);
                    double weightDelta = ALPHA * delta * INPUT[i];
                    WEIGHTS[j] -= weightDelta;
                }
            }

            System.out.println("Calculation # " + k);
            printMatrix(OUTPUT, "TRAIN DATA");
            printMatrix(ACTUAL, "ACTUAL");
        }
    }

    private static void printMatrix(double[][] matrix, String name) {
        System.out.println(name);
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                System.out.print(String.format("%.5f", matrix[i][j]) + " ");
            }
            System.out.println();
        }
        System.out.println("-------------------------------");
    }
}
