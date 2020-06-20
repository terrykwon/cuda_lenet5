############################################
##
## Join Tables Condor command file
##
############################################

executable	 = predict
output		 = result/lenet5_b1.out
error		 = result/lenet5_b1.err
log		     = result/lenet5_b1.log
request_cpus = 1
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = model/values.txt,/nfs/data/cifar10/test_batch.bin
transfer_output_files   = tmp
arguments	              = test_batch.bin 0 1 tmp/cifar10_test_%d_%s.bmp values.txt
queue
