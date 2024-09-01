package org.dto;

public class Pair<K, V> {
    private K key;
    private V value;

    // Constructor to initialize a Pair with key and value
    public Pair(K key, V value) {
        this.key = key;
        this.value = value;
    }

    // Getter method for key
    public K getKey() {
        return key;
    }

    // Getter method for value
    public V getValue() {
        return value;
    }
}