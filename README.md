# MATLAB-Jordan-Networks

Implementation of a Jordan Recurrent Neural Network (JRNN) from scratch in MATLAB for non-linear time-series prediction and system identification.

## Overview
This project demonstrates the implementation of a Jordan Recurrent Neural Network. Unlike standard feedforward networks, the Jordan network incorporates a feedback loop from the output layer to the hidden layer, allowing the network to maintain an "internal state" or memory of previous outputs, making it suitable for dynamic system modeling.

## Key Difference: Elman vs. Jordan
* **Elman Network:** Feedback connection originates from the **hidden layer**.
* **Jordan Network:** Feedback connection originates from the **output layer**.

This implementation utilizes backpropagation and the chain rule for weight updates in the recurrent layer without relying on external toolboxes.

## Features
- Implementation from scratch in MATLAB.
- Dynamic system identification for non-linear mapping.
- Online learning with context-state updates.
- Visualization of training, testing, and checking performance.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
