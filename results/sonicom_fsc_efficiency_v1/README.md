# FSC efficiency benchmark

Validation-only benchmark for FSC-Q14@Q14, FSC-Q26@Q26, and FSC-Q50@Q50. Pure model latency excludes checkpoint loading, input disk I/O, and output serialization; it uses FP32, batch size one, five warm-ups, 30 timed runs, and CUDA synchronization around each timed region. End-to-end latency includes input loading and prediction.h5 serialization. `UniqueDeployedParameters` counts all three members because their checkpoints are independent; frozen does not imply shared. Test subjects were not read.
