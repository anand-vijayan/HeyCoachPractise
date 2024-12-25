package org.dto;

public class Item {
    public int value;
    public int weight;
    public double ratio;

    // Constructor to initialize an Item
    public Item(int value, int weight) {
        this.value = value;
        this.weight = weight;
        this.ratio = (double) value / weight;  // Calculate value to weight ratio
    }
}
