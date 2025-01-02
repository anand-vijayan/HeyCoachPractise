package org.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class Pair<K, V> {
    private K key;
    private V value;
    public K vertex;
    public V weight;

    // Constructor to initialize a Pair with key and value
    public Pair(K key, V value) {
        this.key = key;
        this.value = value;
    }
}