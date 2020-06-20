#########################
# Variables
#########################
NVCC=/usr/local/cuda-10.2/bin/nvcc
BUILD=build
SRC=src
FLAGS=-std=c++11 -O3 $(shell pkg-config --cflags opencv4)
LIBS=$(shell pkg-config --libs opencv4)

#########################
# Main
#########################
predict: $(BUILD)/main.o $(BUILD)/util.o $(BUILD)/LeNet5_cpu.o $(BUILD)/LeNet5_cuda.o $(BUILD)/LeNet5.o
	$(NVCC) $(FLAGS) $(LIBS) -o $@ $^

$(BUILD)/main.o: $(SRC)/main.cpp $(BUILD)/util.o $(BUILD)/LeNet5.o $(BUILD)/LeNet5_cpu.o $(BUILD)/LeNet5_cuda.o
	$(NVCC) $(FLAGS) $(LIBS) -o $@ -c $< 

$(BUILD)/util.o: $(SRC)/util.cpp
	$(NVCC) $(FLAGS) $(LIBS) -o $@ -c $< 

$(BUILD)/LeNet5_cpu.o: $(SRC)/LeNet5_cpu.cpp $(BUILD)/LeNet5.o
	$(NVCC) $(FLAGS) $(LIBS) -o $@ -c $<

$(BUILD)/LeNet5_cuda.o: $(SRC)/LeNet5_cuda.cu $(BUILD)/LeNet5.o
	$(NVCC) $(FLAGS) $(LIBS) -o $@ -c $<

$(BUILD)/LeNet5.o: $(SRC)/LeNet5.cpp
	$(NVCC) $(FLAGS) $(LIBS) -o $@ -c $<

run_on_server: predict
	mkdir -p result
	condor_submit predict_b1.cmd
	condor_submit predict_b128.cmd
#########################
# Hello World
#########################
hello_cpu: $(SRC)/hello.cpp
	$(NVCC) $(FLAGS) $(LIBS) -o $@ $< 
	./$@

hello_cuda: $(SRC)/hello.cu
	$(NVCC) $(FLAGS) $(LIBS) -o $@ $< 

hello_run_on_server: hello_cuda
	condor_submit $<.cmd
#########################
# Util
#########################
format:
	clang-format -i -style=Google $(SRC)/*.cu $(SRC)/*.cpp

clean:
	rm -rf hello_cuda hello_cpu predict result/* tmp/*.log tmp/*.bmp $(BUILD)/*.o

queue:
	condor_q
