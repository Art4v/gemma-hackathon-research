# gemma-hackathon-research

## Problem

Writing fast TPU kernels (Pallas) by hand is slow, specialist work, and small open models can't do it reliably.

## Goal

Fine-tune a small Gemma model to turn plain JAX code into Pallas TPU kernels that are correct and fast on a real TPU, across JAXBench, and show it clearly beats the untrained model on tasks it never saw.
