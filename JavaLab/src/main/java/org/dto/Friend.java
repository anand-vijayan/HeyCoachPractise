package org.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class Friend {
    public int Id;
    public int EatingCapacity;
    public int Budget;
    public boolean Visited;
}
