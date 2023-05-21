package com.example.simpleai.onelayer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class SingleInputSingleOutput {
    private static final double TOLERANCE = 0.00001;

    public static void main(String[] args) {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(System.in))) {
            System.out.println("Provide separated by space: Input, Target, Start weight, Alpha");
            String[] inp = reader.readLine().split(" ");
            double input = Double.parseDouble(inp[0]);
            double target = Double.parseDouble(inp[1]);
            double weight = Double.parseDouble(inp[2]);
            double alpha = Double.parseDouble(inp[3]);
            double error = 100.0;
            int counter = 0;

            while (error > TOLERANCE) {
                double pred = input * weight;
                double delta = pred - target;
                error = Math.pow(delta, 2);
                double weightDelta = alpha * delta * input;
                weight = weight - weightDelta;

                printSemiResult(weight, error, counter, pred);

                counter++;
                if (counter == 10_000) {
                    break;
                }
            }

        } catch (IOException ignored) {}
    }

    private static void printSemiResult(double weight, double error, int counter, double pred) {
        System.out.println("Pred:         " + String.format("%.5f", pred));
        System.out.println("Error:        " + String.format("%.5f", error));
        System.out.println("Weight:       " + String.format("%.5f", weight));
        System.out.println("Counter:      " + counter);
        System.out.println("-----------------------");
    }
}
